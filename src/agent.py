"""Agente Valor Food orquestado con LangGraph.

Flujo por turno (ver docs/arquitectura.md):

    preparar ─► planificar ─┬─► aclarar ──────────────────────────► memorizar ─► FIN
                  ▲         └─► agente ─┬─► herramientas ─► evaluar ─┐
                  │                     │                            │
                  │                     └─► memorizar (respuesta)    │
                  └──────── replanificar (fallo o cambio) ◄──────────┤
                                        agente ◄──── continuar ──────┘

- preparar: recupera contexto documental (RAG) y memoria de largo plazo.
- planificar: plan priorizado (LLM con salida estructurada o heurística).
- agente: LLM con herramientas que ejecuta el plan (ciclo ReAct).
- herramientas: ejecuta y valida llamadas, marca el avance del plan.
- evaluar: detecta fallos y cambios de condiciones y decide si replanificar.
- memorizar: cierra el turno, resume conversaciones largas y guarda episodios.
"""
import json
import operator
import re
from datetime import datetime
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from .operations import load_inventory
from .planner import make_plan
from .prompts import SUMMARY_PROMPT, build_messages
from .tools import make_tools
from .web_search import search_web

MAX_CALLS_PER_ROUND = 6
# Dependencias que el orquestador hace cumplir: no se escribe una orden sin leer el stock en el mismo turno.
PRECONDITIONS = {'generar_orden_compra': ('consultar_inventario', 'priorizar_reposicion')}
# Temas de memoria de largo plazo relevantes para cada herramienta (recuperación orientada a la tarea).
TOOL_TOPICS = {
    'generar_orden_compra': ('ordenes_compra', 'reposicion', 'general'),
    'priorizar_reposicion': ('reposicion', 'ordenes_compra', 'general'),
    'consultar_inventario': ('reposicion', 'general'),
    'comparar_ventas': ('ventas', 'general'),
    'buscar_documentos': ('navegacion', 'general'),
}


class AgentState(TypedDict, total=False):
    # Persisten entre turnos del mismo thread_id (memoria de corto plazo).
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    inventory_snapshot: dict
    reports: Annotated[list, operator.add]
    # Se reinician en cada turno.
    question: str
    context: list
    memories: list
    plan: dict
    plan_source: str
    observations: list
    decisions: list
    trace: list
    rounds: int
    replans: int
    replan_requested: bool
    plan_checked: bool
    web_unavailable: bool
    usage: dict


def _text(message):
    content = message.content
    if isinstance(content, str):
        return content
    return '\n'.join(b.get('text', '') for b in content if isinstance(b, dict))


def _history_text(messages, limit=8):
    lines = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append('Usuario: ' + _text(message)[:400])
        elif isinstance(message, AIMessage) and _text(message).strip():
            lines.append('Agente: ' + _text(message)[:400])
    return '\n'.join(lines[-limit:])


def _before_turn(messages):
    """Mensajes anteriores al turno actual (todo lo previo al último mensaje del usuario)."""
    for position in range(len(messages) - 1, -1, -1):
        if isinstance(messages[position], HumanMessage):
            return messages[:position]
    return []


def unsupported_citations(answer, trace, context=()):
    """Citas [WEB-n] y URL de la respuesta que no provienen de resultados reales de este turno."""
    web_ids, urls = set(), set()
    for entry in trace:
        result = entry['result']
        if entry['tool'] == 'buscar_web' and isinstance(result, dict):
            for hit in result.get('results', []):
                web_ids.add(hit['web_id'])
                urls.add(hit['url'].rstrip('/.'))
        if entry['tool'] == 'buscar_documentos' and isinstance(result, list):
            urls.update(str(h.get('source', '')).rstrip('/.') for h in result)
    urls.update(str(c.get('source', '')).rstrip('/.') for c in context)
    # Se normalizan guiones tipográficos (‐ ‑ ‒ – —) que algunos modelos usan en lugar de '-'.
    cited_ids = {re.sub(r'[\u2010-\u2014]', '-', c) for c in re.findall(r'WEB[-\u2010-\u2014]\d+', answer)}
    cited_urls = {u.rstrip('/.),]*') for u in re.findall(r'https?://[^\s)\]>*`"]+', answer)}
    return sorted(cited_ids - web_ids) + sorted(u for u in cited_urls if u not in urls)


def _log(state, node, decision, detail=''):
    return state.get('decisions', []) + [{'nodo': node, 'decision': decision, 'detalle': detail}]


def _stock_by_name(tool, result):
    if not isinstance(result, dict):
        return {}
    if tool == 'consultar_inventario':
        return {i['name']: i['stock'] for i in result.get('items', [])}
    if tool == 'priorizar_reposicion':
        return {i['name']: i['stock'] for i in result.get('priorities', [])}
    if tool == 'registrar_recepcion' and result.get('updated'):
        return {result['name']: result['stock_after']}
    if tool == 'generar_orden_compra':
        return {l['name']: l['stock_at_order'] for l in result.get('lines', [])}
    return {}


class ValorFoodAgent:
    def __init__(self, llm, retriever, memory, checkpointer=None, web_searcher=search_web, technique='zero_shot',
                 max_rounds=6, max_replans=2, summary_trigger=20, keep_messages=8):
        self.llm, self.retriever, self.memory, self.technique = llm, retriever, memory, technique
        self.max_rounds, self.max_replans = max_rounds, max_replans
        self.summary_trigger, self.keep_messages = summary_trigger, keep_messages
        self.tool_context = {}
        self.tools = {t.name: t for t in make_tools(retriever, memory, web_searcher, self.tool_context)}
        self.model = llm.bind_tools(list(self.tools.values()))
        self.graph = self._build().compile(checkpointer=checkpointer or InMemorySaver())

    # ------------------------------------------------------------------ nodos
    def preparar(self, state):
        question = _text(state['messages'][-1])
        decisions = []
        try:
            context = self.retriever.search(question, 'internal')
        except Exception as exc:  # La recuperación no debe impedir responder.
            context = []
            decisions.append({'nodo': 'preparar', 'decision': 'RAG no disponible', 'detalle': str(exc)[:200]})
        try:
            memories = self.memory.search(question, k=4, min_similarity=0.35)
        except Exception as exc:
            memories = []
            decisions.append({'nodo': 'preparar', 'decision': 'Memoria de largo plazo no disponible', 'detalle': str(exc)[:200]})
        # Percepción del entorno: compara el stock actual con el último observado en este hilo.
        snapshot, observations = dict(state.get('inventory_snapshot') or {}), []
        if snapshot:
            current = {i['name']: i['stock'] for i in load_inventory()}
            changed = [(n, snapshot[n], current[n]) for n in snapshot if n in current and current[n] != snapshot[n]]
            if changed:
                detail = '; '.join(f'{n}: {before} → {after}' for n, before, after in changed)
                observations.append(f'CAMBIO DE CONDICIONES: el stock cambió desde la última consulta de esta conversación ({detail}). '
                                    'No reutilices cifras de mensajes anteriores: vuelve a consultar el inventario.')
                decisions.append({'nodo': 'preparar', 'decision': 'Cambio en el entorno detectado', 'detalle': detail})
                snapshot.update({n: after for n, _, after in changed})
        decisions.append({'nodo': 'preparar', 'decision': 'Contexto recuperado',
                          'detalle': f'{len(context)} fragmentos internos, {len(memories)} recuerdos, '
                                     f'{len(state["messages"]) - 1} mensajes previos en el hilo'})
        return {'question': question, 'context': context, 'memories': memories, 'plan': {}, 'plan_source': '',
                'observations': observations, 'decisions': decisions, 'trace': [], 'rounds': 0, 'replans': 0,
                'replan_requested': False, 'web_unavailable': False, 'plan_checked': False,
                'usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0},
                'summary': state.get('summary', ''), 'inventory_snapshot': snapshot}

    def planificar(self, state):
        previous = state.get('plan') or {}
        done = [s for s in previous.get('pasos', []) if s['estado'] != 'pendiente']
        plan, source, error = make_plan(
            self.llm, state['question'], _history_text(_before_turn(state['messages'])), state.get('summary', ''),
            state.get('memories'), state.get('observations'), previous.get('pasos'))
        # Se conservan los pasos ya ejecutados y se renumeran los nuevos a continuación.
        for number, step in enumerate(plan['pasos'], len(done) + 1):
            step['id'] = number
        plan['pasos'] = done + plan['pasos']
        replanning = state.get('replan_requested', False)
        label = 'Replanificación' if replanning else 'Plan generado'
        memories, decisions = list(state.get('memories') or []), list(state.get('decisions', []))
        topics = sorted({t for st in plan['pasos'] if st['estado'] == 'pendiente' for t in TOOL_TOPICS.get(st['herramienta'], ())})
        task_memories = []
        if topics:
            try:
                # Las preferencias del tema de la tarea se recuperan aunque la pregunta no las mencione,
                # ordenadas por similitud semántica con la pregunta (filtro de metadatos + ranking vectorial).
                known = {m['id'] for m in memories}
                task_memories = [m | {'motivo': 'tarea'} for m in self.memory.search(
                    state['question'], k=5, min_similarity=-1, kinds=['preferencia'], topics=topics) if m['id'] not in known]
                memories += task_memories
            except Exception as exc:
                decisions.append({'nodo': 'planificar', 'decision': 'Memoria de largo plazo no disponible', 'detalle': str(exc)[:200]})
        detail = f'origen={source}; pasos=' + ' → '.join(f"{s['herramienta']}({s['prioridad']})" for s in plan['pasos'] if s['estado'] == 'pendiente')
        if plan.get('requiere_aclaracion'):
            detail += f"; aclaración: {plan['requiere_aclaracion']}"
        if error:
            detail += f'; el LLM no entregó plan válido ({error}), se usó la heurística'
        decisions.append({'nodo': 'planificar', 'decision': label, 'detalle': detail})
        if task_memories:
            decisions.append({'nodo': 'planificar', 'decision': 'Memoria orientada a la tarea',
                              'detalle': f'{len(task_memories)} preferencias de temas {", ".join(topics)}'})
        return {'plan': plan, 'plan_source': source, 'replan_requested': False, 'memories': memories,
                'replans': state.get('replans', 0) + (1 if replanning else 0), 'decisions': decisions}

    def aclarar(self, state):
        question = state['plan']['requiere_aclaracion']
        return {'messages': [AIMessage(content=question)],
                'decisions': _log(state, 'aclarar', 'Se pide un dato faltante antes de actuar', question)}

    def agente(self, state):
        messages = build_messages(self.technique, state['messages'], plan=state.get('plan'),
                                  observations=state.get('observations'), summary=state.get('summary', ''),
                                  memories=state.get('memories'), context=state.get('context'))
        decisions = state.get('decisions', [])
        if state.get('rounds', 0) >= self.max_rounds:
            # Límite de rondas: se fuerza una respuesta con lo obtenido en vez de fallar.
            messages.append(SystemMessage(content='Se alcanzó el límite de pasos. Responde ahora con la información obtenida y declara lo que quedó pendiente.'))
            reply = self.llm.invoke(messages)
            decisions = decisions + [{'nodo': 'agente', 'decision': 'Límite de rondas alcanzado', 'detalle': 'Respuesta forzada sin herramientas'}]
        else:
            reply = self.model.invoke(messages)
            pending = [s for s in (state.get('plan') or {}).get('pasos', []) if s['estado'] == 'pendiente' and s['prioridad'] == 'alta']
            if not getattr(reply, 'tool_calls', None) and pending and not state.get('plan_checked'):
                # Control de adherencia al plan: una sola vez por turno se piden los pasos imprescindibles omitidos.
                names = ', '.join(s['herramienta'] for s in pending)
                reply = self.model.invoke(messages + [SystemMessage(
                    content=f'Antes de responder: quedan pasos de prioridad ALTA sin ejecutar ({names}). '
                            'Ejecútalos ahora o, si ya no aplican por las observaciones, responde explicando por qué.')])
                decisions = decisions + [{'nodo': 'agente', 'decision': 'Verificación de plan', 'detalle': f'Pasos alta pendientes: {names}'}]
            if not getattr(reply, 'tool_calls', None) and not _text(reply).strip():
                reply = self.llm.invoke(messages + [SystemMessage(content='Tu respuesta anterior llegó vacía. Responde al usuario ahora.')])
            if not getattr(reply, 'tool_calls', None):
                # Verificador de fidelidad: toda cita web o URL debe provenir de un resultado real de este turno.
                invalid = unsupported_citations(_text(reply), state.get('trace', []), state.get('context', []))
                if invalid:
                    reply = self.llm.invoke(messages + [reply, SystemMessage(
                        content=f'Tu respuesta cita fuentes que NO aparecen en los resultados de herramientas de este turno: {", ".join(invalid)}. '
                                'Reescríbela completa sin esas citas ni URL. No afirmes haber buscado si no se ejecutó buscar_web; '
                                'presenta cualquier orientación general como conocimiento del modelo no verificado con fuentes.')])
                    still = unsupported_citations(_text(reply), state.get('trace', []), state.get('context', []))
                    decisions = decisions + [{'nodo': 'agente', 'decision': 'Citas no respaldadas corregidas' if not still else 'Citas no respaldadas persistentes',
                                              'detalle': ', '.join(invalid) + (f' | persisten: {", ".join(still)}' if still else '')}]
                    if still:
                        reply = AIMessage(content=_text(reply) + '\n\n⚠️ Advertencia del verificador: esta respuesta contiene referencias '
                                          f'que no provienen de resultados reales ({", ".join(still)}). No las consideres fuentes verificadas.')
        if not getattr(reply, 'tool_calls', None) and not _text(reply).strip():
            raise RuntimeError('El modelo no devolvió texto.')
        usage = dict(state.get('usage') or {})
        for key, value in (getattr(reply, 'usage_metadata', None) or {}).items():
            if key in ('input_tokens', 'output_tokens', 'total_tokens'):
                usage[key] = usage.get(key, 0) + value
        checked = state.get('plan_checked', False) or any(d['decision'] == 'Verificación de plan' for d in decisions)
        return {'messages': [reply], 'rounds': state.get('rounds', 0) + 1, 'usage': usage, 'decisions': decisions,
                'plan_checked': checked}

    def herramientas(self, state):
        calls = state['messages'][-1].tool_calls
        plan = {**state['plan'], 'pasos': [dict(s) for s in state['plan'].get('pasos', [])]}
        trace, decisions, outputs = list(state.get('trace', [])), list(state.get('decisions', [])), []
        for position, call in enumerate(calls):
            name, args = call['name'], call['args']
            if position >= MAX_CALLS_PER_ROUND:
                result = {'error': 'Demasiadas herramientas en una ronda; se omitió esta llamada.'}
            elif name not in self.tools:
                result = {'error': 'Herramienta no permitida.'}
            elif name in PRECONDITIONS and not any(
                    t['tool'] in PRECONDITIONS[name] and not (isinstance(t['result'], dict) and t['result'].get('error'))
                    for t in trace):
                result = {'error': f'Precondición no cumplida: antes de {name} ejecuta {" o ".join(PRECONDITIONS[name])} '
                                   'en este turno para trabajar con stock actual.'}
            elif name == 'buscar_web' and state.get('web_unavailable'):
                result = {'error': 'Omitida: la búsqueda web falló antes en este turno. Usa fuentes internas o informa el límite.'}
            else:
                try:
                    result = self.tools[name].invoke(args)
                except (ValueError, TypeError) as exc:  # Incluye errores de validación Pydantic.
                    result = {'error': str(exc)[:500]}
            failed = isinstance(result, dict) and bool(result.get('error'))
            step = next((s for s in plan['pasos'] if s['herramienta'] == name and s['estado'] == 'pendiente'), None)
            if step:
                step['estado'] = 'fallido' if failed else 'completado'
            else:
                decisions.append({'nodo': 'herramientas', 'decision': 'Acción no planificada',
                                  'detalle': f'El ejecutor agregó {name} para completar la tarea'})
            trace.append({'round': state['rounds'], 'tool': name, 'arguments': args, 'result': result})
            outputs.append(ToolMessage(content=json.dumps(result, ensure_ascii=False, default=str), tool_call_id=call['id']))
        return {'messages': outputs, 'trace': trace, 'plan': plan, 'decisions': decisions}

    def evaluar(self, state):
        """Evaluador determinista: compara resultados con el entorno y decide cómo seguir."""
        latest = [t for t in state['trace'] if t['round'] == state['rounds']]
        observations, decisions = list(state.get('observations', [])), list(state.get('decisions', []))
        snapshot = dict(state.get('inventory_snapshot') or {})
        replan, web_unavailable = False, state.get('web_unavailable', False)
        received = {t['result']['name'] for t in latest
                    if t['tool'] == 'registrar_recepcion' and isinstance(t['result'], dict) and t['result'].get('updated')}

        for entry in latest:
            tool, result = entry['tool'], entry['result']
            if isinstance(result, dict) and result.get('error'):
                observations.append(f'FALLO en {tool}: {result["error"]}')
                decisions.append({'nodo': 'evaluar', 'decision': f'Fallo detectado en {tool}', 'detalle': 'Se solicita replanificar con alternativas'})
                replan = True
                if tool == 'buscar_web':
                    web_unavailable = True
                    observations.append('No hay resultados web en este turno: no cites [WEB-n] ni URL externas. '
                                        'Si das orientación general, decláralo como conocimiento no verificado.')
                continue
            current = _stock_by_name(tool, result)
            changed = [(n, snapshot[n], s) for n, s in current.items() if n in snapshot and snapshot[n] != s and n not in received]
            if changed:
                detail = '; '.join(f'{n}: {before} → {after}' for n, before, after in changed)
                observations.append(f'CAMBIO DE CONDICIONES: el stock cambió desde la última consulta ({detail}). '
                                    'Recalcula con los datos actuales y no reutilices cifras anteriores.')
                decisions.append({'nodo': 'evaluar', 'decision': 'Cambio en el entorno detectado', 'detalle': detail})
                replan = True
            snapshot.update(current)
            if tool == 'consultar_inventario' and isinstance(result, dict):
                missing = [i['name'] for i in result.get('items', []) if i.get('buy') is None]
                if missing:
                    observations.append(f'Sin stock mínimo definido: {", ".join(missing)}. No calcules reposición para ellos; infórmalo.')
                    decisions.append({'nodo': 'evaluar', 'decision': 'Dato faltante en inventario', 'detalle': ', '.join(missing)})
                if not result.get('items'):
                    observations.append(f'Ingrediente no encontrado con {entry["arguments"]}. Pide al usuario que lo aclare.')
                    decisions.append({'nodo': 'evaluar', 'decision': 'Ingrediente inexistente', 'detalle': json.dumps(entry['arguments'], ensure_ascii=False)})
            if tool == 'generar_orden_compra' and isinstance(result, dict) and result.get('rejected'):
                detail = '; '.join(f"{r['name']}: {r['reason']}" for r in result['rejected'])
                observations.append(f'Líneas rechazadas por validación: {detail}. Explícalo al usuario.')
                decisions.append({'nodo': 'evaluar', 'decision': 'Validación de escritura rechazó líneas', 'detalle': detail})
            warnings = [f"{l['name']}: {l['warning']}" for l in result.get('lines', []) if l.get('warning')] \
                if tool == 'generar_orden_compra' and isinstance(result, dict) else []
            if warnings:
                observations.append('Cantidades por sobre lo necesario: ' + '; '.join(warnings) + ' Infórmalo al usuario.')
                decisions.append({'nodo': 'evaluar', 'decision': 'Cantidad sobre lo necesario', 'detalle': '; '.join(warnings)})
            if tool == 'comparar_ventas' and isinstance(result, dict) and not result.get('months'):
                observations.append(f'No hay ventas registradas para {result.get("year")}. No inventes cifras.')
                decisions.append({'nodo': 'evaluar', 'decision': 'Sin datos para el periodo', 'detalle': str(result.get('year'))})

        can_replan = replan and state.get('replans', 0) < self.max_replans
        if replan and not can_replan:
            decisions.append({'nodo': 'evaluar', 'decision': 'Límite de replanificaciones', 'detalle': 'Se continúa con el plan vigente'})
        if not latest:
            decisions.append({'nodo': 'evaluar', 'decision': 'Sin resultados nuevos', 'detalle': ''})
        return {'observations': observations, 'decisions': decisions, 'inventory_snapshot': snapshot,
                'web_unavailable': web_unavailable, 'replan_requested': can_replan}

    def memorizar(self, state):
        decisions = list(state.get('decisions', []))
        pending = [s for s in state.get('plan', {}).get('pasos', []) if s['estado'] == 'pendiente']
        if pending:
            decisions.append({'nodo': 'memorizar', 'decision': 'Pasos no ejecutados',
                              'detalle': ', '.join(f"{s['herramienta']}({s['prioridad']})" for s in pending)})
        update = {}
        messages = state['messages']
        if len(messages) > self.summary_trigger:
            cut = len(messages) - self.keep_messages
            while cut < len(messages) and not isinstance(messages[cut], HumanMessage):
                cut += 1  # Se corta en el inicio de un turno para no separar llamadas y resultados de herramientas.
            old = messages[:cut]
            conversation = _history_text(old, limit=len(old))
            try:
                summary = _text(self.llm.invoke([HumanMessage(content=SUMMARY_PROMPT.format(
                    previous=state.get('summary') or 'Ninguno', conversation=conversation))])).strip()
                if not summary:
                    raise RuntimeError('Resumen vacío')
                update['summary'] = summary
                update['messages'] = [RemoveMessage(id=m.id) for m in old]
                memory_result = self.memory.add('episodio', summary[:500], self.tool_context.get('thread_id'))
                decisions.append({'nodo': 'memorizar', 'decision': 'Ventana de contexto compactada',
                                  'detalle': f'{len(old)} mensajes resumidos; episodio {memory_result["id"]} guardado en memoria de largo plazo'})
            except Exception as exc:
                decisions.append({'nodo': 'memorizar', 'decision': 'No se pudo resumir', 'detalle': f'{type(exc).__name__}: {str(exc)[:150]}'})
        final = messages[-1]
        report = {'timestamp': datetime.now().isoformat(timespec='seconds'), 'question': state['question'],
                  'answer': _text(final), 'technique': self.technique, 'plan': state.get('plan'),
                  'plan_source': state.get('plan_source'), 'decisions': decisions,
                  'observations': state.get('observations', []), 'trace': state.get('trace', []),
                  'context': state.get('context', []), 'memories': state.get('memories', []),
                  'rounds': state.get('rounds', 0), 'replans': state.get('replans', 0),
                  'usage': state.get('usage', {}), 'simulated_data': True,
                  'validation': 'Respuesta pendiente de revisión humana.'}
        return {**update, 'decisions': decisions, 'reports': [report]}

    # ------------------------------------------------------------------ rutas
    def _after_plan(self, state):
        return 'aclarar' if state['plan'].get('requiere_aclaracion') else 'agente'

    def _after_agent(self, state):
        return 'herramientas' if getattr(state['messages'][-1], 'tool_calls', None) else 'memorizar'

    def _after_evaluation(self, state):
        return 'planificar' if state.get('replan_requested') else 'agente'

    def _build(self):
        graph = StateGraph(AgentState)
        for name in ('preparar', 'planificar', 'aclarar', 'agente', 'herramientas', 'evaluar', 'memorizar'):
            graph.add_node(name, getattr(self, name))
        graph.add_edge(START, 'preparar')
        graph.add_edge('preparar', 'planificar')
        graph.add_conditional_edges('planificar', self._after_plan, ['aclarar', 'agente'])
        graph.add_edge('aclarar', 'memorizar')
        graph.add_conditional_edges('agente', self._after_agent, ['herramientas', 'memorizar'])
        graph.add_edge('herramientas', 'evaluar')
        graph.add_conditional_edges('evaluar', self._after_evaluation, ['planificar', 'agente'])
        graph.add_edge('memorizar', END)
        return graph

    # ------------------------------------------------------------- interfaz
    def ask(self, question, thread_id):
        if not isinstance(question, str) or not question.strip() or len(question) > 1000:
            raise ValueError('Escribe una pregunta de 1 a 1000 caracteres.')
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError('Falta el identificador de conversación.')
        self.tool_context['thread_id'] = thread_id
        state = self.graph.invoke({'messages': [HumanMessage(content=question.strip())]},
                                  config={'configurable': {'thread_id': thread_id}, 'recursion_limit': 80})
        return {**state['reports'][-1], 'thread_id': thread_id}

    def state(self, thread_id):
        return self.graph.get_state({'configurable': {'thread_id': thread_id}}).values

    def mermaid(self):
        return self.graph.get_graph().draw_mermaid()

"""Pruebas del grafo LangGraph con un LLM simulado (sin API).

Cada prueba documenta un comportamiento evaluado: flujo de herramientas
(IE1), memoria de corto plazo (IE3), memoria semántica de largo plazo (IE4)
y planificación/adaptación ante condiciones cambiantes (IE5).
"""
import json
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from conftest import ScriptedLLM, call, step
from src import operations as ops
from src.agent import ValorFoodAgent


def agent_for(llm, retriever, memory, **kwargs):
    return ValorFoodAgent(llm, retriever, memory, **kwargs)


def decisions(report):
    return [d['decision'] for d in report['decisions']]


def test_full_replenishment_flow_reads_reasons_and_writes(retriever, memory, isolated_runtime):
    """Consulta → razonamiento → escritura en un solo turno, siguiendo el plan priorizado."""
    plan = {'objetivo': 'Preparar reposición', 'pasos': [
        step('generar_orden_compra', 'media'), step('priorizar_reposicion', 'media'), step('consultar_inventario', 'alta')]}
    llm = ScriptedLLM(plans=[plan], replies=[
        call('consultar_inventario', {}),
        call('priorizar_reposicion', {'excluir': ['Pan']}),
        call('generar_orden_compra', {'excluir': ['Pan']}),
        AIMessage(content='Generé el borrador con queso y carne [inventario simulado].'),
    ])
    report = agent_for(llm, retriever, memory).ask('Revisa el stock y prepara la orden sin pan', 'hilo-1')
    # El plan quedó ordenado por fase aunque el LLM lo propuso desordenado.
    assert [s['herramienta'] for s in report['plan']['pasos']] == ['consultar_inventario', 'priorizar_reposicion', 'generar_orden_compra']
    assert all(s['estado'] == 'completado' for s in report['plan']['pasos'])
    assert [t['tool'] for t in report['trace']] == ['consultar_inventario', 'priorizar_reposicion', 'generar_orden_compra']
    order = report['trace'][-1]['result']
    assert order['created'] and {l['name'] for l in order['lines']} == {'Queso', 'Carne'}
    assert (isolated_runtime / 'ordenes' / f"{order['order_id']}.csv").exists()
    # La escritura queda registrada como decisión en la memoria de largo plazo.
    assert memory.search('borrador orden queso', kinds=['decision'], min_similarity=0.1)
    assert 'Validación de escritura rechazó líneas' in decisions(report)  # Salsa sin mínimo, Pan excluido.
    assert report['plan_source'] == 'llm'


def test_short_term_memory_keeps_conversation_continuity(retriever, memory):
    """El año dado en el turno 2 resuelve la aclaración pedida en el turno 1 del mismo hilo."""
    def plan_turn_two(messages):
        assert 'mejor mes' in messages[0].content  # El planificador ve el turno anterior.
        return {'objetivo': 'Comparar ventas 2026', 'pasos': [step('comparar_ventas')]}

    def final(messages):
        assert any(isinstance(m, HumanMessage) and 'mejor mes' in m.content for m in messages)
        assert json.loads(messages[-1].content)['best_months'][0]['month'] == '2026-03'
        return AIMessage(content='Marzo de 2026 fue el mes con mayor facturación [ventas simuladas].')

    llm = ScriptedLLM(
        plans=[{'objetivo': 'Mejor mes', 'requiere_aclaracion': '¿De qué año?', 'pasos': []}, plan_turn_two],
        replies=[call('comparar_ventas', {'year': 2026}), final])
    agent = agent_for(llm, retriever, memory)
    first = agent.ask('¿Cuál fue el mejor mes?', 'hilo-ventas')
    assert first['answer'] == '¿De qué año?' and first['trace'] == []
    second = agent.ask('Del 2026', 'hilo-ventas')
    assert 'Marzo' in second['answer']
    state = agent.state('hilo-ventas')
    assert [type(m).__name__ for m in state['messages']][:3] == ['HumanMessage', 'AIMessage', 'HumanMessage']
    assert len(state['reports']) == 2
    # Otro hilo no comparte la memoria de corto plazo.
    assert agent.state('otro-hilo') == {}


def test_long_term_memory_crosses_conversations(retriever, memory):
    """Una preferencia guardada en un hilo se recupera semánticamente en otro hilo nuevo."""
    def check_memory_in_prompt(messages):
        system = messages[0]
        assert isinstance(system, SystemMessage) and 'panadería' in system.content
        return call('generar_orden_compra', {'excluir': ['Pan']})

    llm = ScriptedLLM(
        plans=[{'objetivo': 'Guardar preferencia', 'pasos': [step('guardar_preferencia')]},
               {'objetivo': 'Orden', 'pasos': [step('consultar_inventario'), step('generar_orden_compra', 'media')]}],
        replies=[call('guardar_preferencia', {'texto': 'No incluir pan en las órdenes de compra: lo trae la panadería'}),
                 AIMessage(content='Preferencia guardada.'),
                 call('consultar_inventario', {}), check_memory_in_prompt,
                 AIMessage(content='Borrador generado sin pan por tu preferencia [memoria].')])
    agent = agent_for(llm, retriever, memory)
    agent.ask('Recuerda que no incluyas pan en las órdenes, lo trae la panadería', 'hilo-a')
    report = agent.ask('Genera la orden de compra de pan y lo demás', 'hilo-b')
    assert report['memories'] and report['memories'][0]['kind'] == 'preferencia'
    order = report['trace'][-1]['result']
    assert 'Pan' not in {l['name'] for l in order['lines']}


def test_stock_change_between_turns_is_perceived_before_planning(retriever, memory):
    """Si el stock cambia fuera del agente entre turnos, el cambio se percibe antes de planificar."""
    def plan_with_change(messages):
        assert 'CAMBIO DE CONDICIONES' in messages[0].content and 'Queso: 2 → 7' in messages[0].content
        return {'objetivo': 'Orden de queso', 'pasos': [step('consultar_inventario'), step('generar_orden_compra')]}

    def stale_order(messages):
        assert 'CAMBIO DE CONDICIONES' in messages[0].content  # El ejecutor también recibe la observación.
        return call('generar_orden_compra', {'lineas': [{'nombre': 'Queso', 'cantidad': 6}]})

    llm = ScriptedLLM(
        plans=[{'objetivo': 'Queso', 'pasos': [step('consultar_inventario')]}, plan_with_change],
        replies=[call('consultar_inventario', {'name': 'queso'}), AIMessage(content='Faltan 6 kg.'),
                 stale_order,  # El modelo intenta escribir con la cifra antigua y sin releer el stock.
                 call('consultar_inventario', {'name': 'queso'}),
                 call('generar_orden_compra', {'lineas': [{'nombre': 'Queso'}]}),
                 AIMessage(content='El stock cambió: ahora solo falta 1 kg y generé ese borrador.')])
    agent = agent_for(llm, retriever, memory)
    agent.ask('¿Cuánto queso falta?', 'hilo-stock')
    ops.register_receipt('Queso', 5)  # Llega mercadería registrada en el ERP, fuera de la conversación.
    report = agent.ask('Genera la orden del queso', 'hilo-stock')
    assert report['decisions'][0]['decision'] == 'Cambio en el entorno detectado'
    # La precondición bloquea la escritura con datos viejos; luego se escribe con el stock actual.
    assert report['trace'][0]['result']['error'].startswith('Precondición no cumplida')
    assert report['trace'][-1]['result']['lines'][0]['quantity'] == 1.0
    assert agent.state('hilo-stock')['inventory_snapshot']['Queso'] == 7


def test_stock_change_during_turn_triggers_replanning(retriever, memory):
    """Si el entorno cambia mientras el agente trabaja, el evaluador lo detecta y replanifica."""
    def change_then_reread(messages):
        ops.register_receipt('Carne', 4)  # Otro usuario registra una recepción a mitad del turno.
        return call('priorizar_reposicion', {})

    def replan(messages):
        assert 'CAMBIO DE CONDICIONES' in messages[0].content and 'Carne: 3 → 7' in messages[0].content
        return {'objetivo': 'Recalcular', 'pasos': [step('generar_orden_compra')]}

    llm = ScriptedLLM(
        plans=[{'objetivo': 'Orden', 'pasos': [step('consultar_inventario'), step('priorizar_reposicion'),
                                               step('generar_orden_compra', 'media')]}, replan],
        replies=[call('consultar_inventario', {}), change_then_reread, call('generar_orden_compra', {}),
                 AIMessage(content='La carne cambió durante la revisión; la orden usa el stock nuevo.')])
    report = agent_for(llm, retriever, memory).ask('Prepara la orden de compra', 'hilo-cambio')
    assert 'Cambio en el entorno detectado' in decisions(report)
    assert 'Replanificación' in decisions(report) and report['replans'] == 1
    carne = next(l for l in report['trace'][-1]['result']['lines'] if l['name'] == 'Carne')
    assert carne['quantity'] == 2.0


def test_task_oriented_memory_recalls_preferences_not_mentioned(retriever, memory):
    """Una preferencia del tema 'ordenes_compra' se recupera para una orden aunque la pregunta no la mencione."""
    memory.add('preferencia', 'El pan nunca se incluye porque lo trae la panadería', topic='ordenes_compra')
    memory.add('preferencia', 'Mostrar la facturación en millones', topic='ventas')
    llm = ScriptedLLM(plans=[{'objetivo': 'Orden', 'pasos': [step('consultar_inventario'), step('generar_orden_compra')]}],
                      replies=[call('consultar_inventario', {}), call('generar_orden_compra', {'excluir': ['Pan']}),
                               AIMessage(content='Listo.')])
    report = agent_for(llm, retriever, memory).ask('Hazme el pedido de la semana', 'hilo-tarea')
    recalled = [m for m in report['memories'] if m.get('motivo') == 'tarea']
    assert [m['topic'] for m in recalled] == ['ordenes_compra']
    assert 'Memoria orientada a la tarea' in decisions(report)


def test_quantity_above_need_is_flagged(retriever, memory):
    llm = ScriptedLLM(plans=[{'objetivo': 'x', 'pasos': [step('consultar_inventario'), step('generar_orden_compra')]}],
                      replies=[call('consultar_inventario', {}),
                               call('generar_orden_compra', {'lineas': [{'nombre': 'Queso', 'cantidad': 10}]}),
                               AIMessage(content='Generada con advertencia.')])
    report = agent_for(llm, retriever, memory).ask('Pide 10 kg de queso', 'hilo-w')
    assert 'Cantidad sobre lo necesario' in decisions(report)


def test_web_failure_is_detected_and_not_retried(retriever, memory):
    """Si la búsqueda web falla, el agente replanifica y bloquea reintentos en el mismo turno."""
    def no_web(query):
        raise ConnectionError('sin conexión')

    def replan(messages):
        assert 'FALLO en buscar_web' in messages[0].content
        return {'objetivo': 'Responder con fuentes internas', 'pasos': [step('buscar_documentos')]}

    llm = ScriptedLLM(
        plans=[{'objetivo': 'Buenas prácticas', 'pasos': [step('buscar_web')]}, replan],
        replies=[call('buscar_web', {'query': 'almacenar carne', 'reason': 'No hay documentación interna'}),
                 call('buscar_web', {'query': 'almacenar carne 2', 'reason': 'Reintento'}),
                 call('buscar_documentos', {'query': 'carne'}),
                 AIMessage(content='La búsqueda externa no está disponible; con las fuentes internas no puedo responderlo.')])
    report = agent_for(llm, retriever, memory, web_searcher=no_web).ask('¿Cómo almaceno carne?', 'hilo-web')
    results = [t['result'] for t in report['trace'] if t['tool'] == 'buscar_web']
    assert 'ConnectionError' in results[0]['error'] and results[1]['error'].startswith('Omitida')
    assert 'Fallo detectado en buscar_web' in decisions(report)
    assert report['replans'] >= 1


def test_planner_falls_back_to_heuristics(retriever, memory):
    llm = ScriptedLLM(plans=[RuntimeError('salida no estructurada')],
                      replies=[call('consultar_inventario', {}), AIMessage(content='Stock revisado.')])
    report = agent_for(llm, retriever, memory).ask('¿Qué stock está crítico?', 'hilo-h')
    assert report['plan_source'] == 'heuristica'
    assert report['plan']['pasos'][0]['herramienta'] == 'consultar_inventario'


def test_missing_minimum_and_unplanned_actions_are_reported(retriever, memory):
    llm = ScriptedLLM(plans=[{'objetivo': 'Salsa', 'pasos': [step('consultar_inventario'), step('priorizar_reposicion', 'baja')]}],
                      replies=[call('consultar_inventario', {'name': 'salsa'}), call('buscar_documentos', {'query': 'mínimo'}),
                               AIMessage(content='La salsa no tiene mínimo definido.')])
    report = agent_for(llm, retriever, memory).ask('¿Cuánta salsa repongo?', 'hilo-s')
    assert 'Dato faltante en inventario' in decisions(report)
    assert 'Acción no planificada' in decisions(report)
    assert 'Pasos no ejecutados' in decisions(report)  # priorizar_reposicion quedó pendiente.


def test_round_limit_forces_answer(retriever, memory):
    def endless(messages):
        if isinstance(messages[-1], SystemMessage) and 'límite' in messages[-1].content:
            return AIMessage(content='Respondo con lo obtenido hasta ahora.')
        return call('consultar_inventario', {}, call_id=f'c{len(messages)}')
    llm = ScriptedLLM(plans=[{'objetivo': 'x', 'pasos': []}], replies=[endless] * 10)
    report = agent_for(llm, retriever, memory, max_rounds=3).ask('stock', 'hilo-limite')
    assert report['rounds'] == 4 and 'Límite de rondas alcanzado' in decisions(report)
    assert report['answer'] == 'Respondo con lo obtenido hasta ahora.'


def test_long_conversation_is_summarized_into_memory(retriever, memory):
    """Al superar la ventana se resumen los mensajes antiguos y el episodio pasa a memoria de largo plazo."""
    replies = [AIMessage(content=f'Respuesta {i}') for i in range(3)]
    replies.append(AIMessage(content='- El usuario consultó queso y ventas de 2026.'))  # Resumen.
    llm = ScriptedLLM(replies=replies)
    agent = agent_for(llm, retriever, memory, summary_trigger=5, keep_messages=2)
    for i in range(3):
        report = agent.ask(f'Pregunta {i} sobre queso', 'hilo-largo')
    state = agent.state('hilo-largo')
    assert state['summary'].startswith('- El usuario consultó')
    assert len(state['messages']) == 2 and isinstance(state['messages'][0], HumanMessage)
    assert 'Ventana de contexto compactada' in decisions(report)
    assert memory.all()[0]['kind'] == 'episodio'


def test_invalid_inputs(retriever, memory):
    agent = agent_for(ScriptedLLM(), retriever, memory)
    with pytest.raises(ValueError): agent.ask('', 'h')
    with pytest.raises(ValueError): agent.ask('hola', '')
    assert 'preparar' in agent.mermaid()


def test_plan_adherence_check_requests_skipped_high_priority_steps(retriever, memory):
    """Si el ejecutor intenta responder omitiendo un paso de prioridad alta, el grafo se lo exige una vez."""
    def nudged(messages):
        assert 'prioridad ALTA' in messages[-1].content and 'priorizar_reposicion' in messages[-1].content
        return call('priorizar_reposicion', {})
    llm = ScriptedLLM(plans=[{'objetivo': 'x', 'pasos': [step('consultar_inventario'), step('priorizar_reposicion')]}],
                      replies=[call('consultar_inventario', {}), AIMessage(content='Respuesta apresurada'), nudged,
                               AIMessage(content='Queso y carne son críticos.')])
    report = agent_for(llm, retriever, memory).ask('¿Qué es lo más urgente de reponer?', 'hilo-p')
    assert 'Verificación de plan' in decisions(report)
    assert all(s['estado'] == 'completado' for s in report['plan']['pasos'])
    assert report['answer'] == 'Queso y carne son críticos.'


def test_fabricated_web_citations_are_detected_and_rewritten(retriever, memory):
    """Si la web falla y el modelo inventa [WEB-n] o URL, el verificador exige reescribir la respuesta."""
    def no_web(query):
        raise ConnectionError('sin conexión')

    def rewrite(messages):
        assert 'WEB-1' in messages[-1].content and 'https://www.fda.gov/food' in messages[-1].content
        return AIMessage(content='La búsqueda web no está disponible; como orientación general no verificada, refrigera bajo 4 °C.')

    llm = ScriptedLLM(plans=[{'objetivo': 'x', 'pasos': [step('buscar_web')]}, {'objetivo': 'x', 'pasos': []}],
                      replies=[call('buscar_web', {'query': 'carne', 'reason': 'Sin documentos internos'}),
                               AIMessage(content='Refrigera bajo 4 °C [WEB-1] (https://www.fda.gov/food).'), rewrite])
    report = agent_for(llm, retriever, memory, web_searcher=no_web).ask('¿Cómo guardo carne?', 'hilo-citas')
    assert 'Citas no respaldadas corregidas' in decisions(report)
    assert 'WEB-1' not in report['answer']
    assert any('No hay resultados web' in o for o in report['observations'])


def test_real_web_citations_are_accepted():
    from src.agent import unsupported_citations
    trace = [{'tool': 'buscar_web', 'result': {'results': [{'web_id': 'WEB-1', 'url': 'https://example.com/a', 'title': 't'}]}}]
    assert unsupported_citations('Dato [WEB-1] (https://example.com/a).', trace) == []
    assert unsupported_citations('Dato [WEB-2] https://otro.cl/x', trace) == ['WEB-2', 'https://otro.cl/x']
    assert unsupported_citations('Dato [WEB‑1] y [WEB‑3]', trace) == ['WEB-3']  # Guion no separable.

import json
import os
import uuid
import streamlit as st
from src import operations as ops
from src.agent import ValorFoodAgent
from src.memory import LongTermMemory, open_checkpointer, list_threads
from src.prompts import TECHNIQUES
from src.retrieval import configured_retriever
from src.settings import DATA, load_settings, get_llm, get_embeddings, embedding_signature, model_name

st.set_page_config(page_title='Valor Food Agente', page_icon='🍽️', layout='wide')
load_settings()


@st.cache_resource
def embeddings():
    return get_embeddings()


@st.cache_resource
def checkpointer():
    return open_checkpointer()


@st.cache_resource
def long_term_memory():
    return LongTermMemory(embeddings, embedding_signature())


@st.cache_resource
def agent(technique):
    return ValorFoodAgent(get_llm(), configured_retriever(embeddings()), long_term_memory(), checkpointer(), technique=technique)


def thread_values(thread_id):
    saved = checkpointer().get_tuple({'configurable': {'thread_id': thread_id}})
    return saved.checkpoint['channel_values'] if saved else {}


def new_thread():
    st.session_state['thread_id'] = 'conv-' + uuid.uuid4().hex[:6]


def show_report(report):
    with st.expander('Cómo lo resolvió el agente: plan, decisiones y herramientas'):
        steps = report.get('plan', {}).get('pasos', [])
        st.markdown(f"**Plan** (origen: `{report.get('plan_source') or '—'}`, replanificaciones: {report.get('replans', 0)})")
        if steps:
            st.dataframe(steps, hide_index=True, use_container_width=True)
        else:
            st.caption('Sin pasos: respuesta directa o aclaración.')
        st.markdown('**Decisiones del grafo**')
        st.dataframe(report.get('decisions', []), hide_index=True, use_container_width=True)
        if report.get('observations'):
            st.markdown('**Observaciones del evaluador**')
            for item in report['observations']:
                st.warning(item)
        if report.get('memories'):
            st.markdown('**Memoria de largo plazo recuperada**')
            st.dataframe(report['memories'], hide_index=True, use_container_width=True)
        for i, item in enumerate(report.get('trace', []), 1):
            st.markdown(f"**{i}. {item['tool']}** · ronda {item['round']}")
            result = item['result']
            if item['tool'] == 'buscar_web' and isinstance(result, dict):
                st.caption('Motivo del respaldo externo: ' + str(result.get('reason', '—')))
                if result.get('error'):
                    st.warning(result['error'])
                for hit in result.get('results', []):
                    st.markdown(f"{hit['web_id']} · [{hit['title']}]({hit['url']})")
            else:
                st.json({'argumentos': item['arguments'], 'resultado': result}, expanded=False)
        st.caption(f"Tokens del ejecutor: {report.get('usage', {})} · {report.get('validation', '')}")
        st.download_button('Descargar evidencia JSON', json.dumps(report, ensure_ascii=False, indent=2, default=str),
                           file_name=f"turno-{report.get('timestamp', 'x').replace(':', '')}.json", mime='application/json',
                           key='dl-' + report.get('timestamp', '') + str(len(report.get('trace', []))))


if 'thread_id' not in st.session_state:
    new_thread()

st.title('Valor Food Agente')
st.caption('Prototipo académico independiente · Matías Medina · DuocUC · ISY0101 · 006D')
st.info('Inventario, ventas y órdenes son SIMULADOS. El agente no se conecta a Crispy House ni ejecuta compras.')

with st.sidebar:
    st.header('Conversación')
    threads = list_threads(checkpointer())
    options = list(dict.fromkeys([st.session_state['thread_id']] + threads))
    st.session_state['thread_id'] = st.selectbox('Hilo (memoria de corto plazo)', options,
                                                 index=options.index(st.session_state['thread_id']))
    st.button('Nueva conversación', on_click=new_thread)
    technique = st.selectbox('Técnica de prompt', TECHNIQUES)
    st.divider()
    st.write('Modelo:', os.getenv('LLM_PROVIDER', 'groq'), '·', model_name())
    st.write('Embeddings:', embedding_signature())
    st.write('Búsqueda web:', 'activa' if os.getenv('WEB_SEARCH_ENABLED', 'true').lower() in ('1', 'true', 'yes', 'si', 'sí', 'on') else 'desactivada')
    st.caption('Orquestación con LangGraph: preparar → planificar → agente ⇄ herramientas → evaluar → memorizar.')

chat, env_tab, memory_tab, help_tab = st.tabs(['Agente', 'Entorno simulado', 'Memoria', 'Instrucciones'])

with chat:
    thread_id = st.session_state['thread_id']
    values = thread_values(thread_id)
    if values.get('summary'):
        st.caption('Resumen de la parte antigua de esta conversación (memoria de corto plazo compactada):')
        st.markdown(values['summary'])
    for report in values.get('reports', []):
        with st.chat_message('user'):
            st.write(report['question'])
        with st.chat_message('assistant'):
            st.write(report['answer'])
            show_report(report)
    if not values.get('reports'):
        st.write('Ejemplos: «Revisa el stock y prepárame una orden de compra con lo crítico» · '
                 '«Recuerda que el pan no se incluye en las órdenes» · «¿Cuál fue el mejor mes?» · '
                 '«Llegaron 5 kg de queso» · «¿Dónde veo el stock crítico?»')
    question = st.chat_input('Escribe tu mensaje')
    if question is not None:
        if not question.strip():
            st.warning('Escribe una pregunta.')
        else:
            try:
                with st.spinner('Planificando y ejecutando…'):
                    agent(technique).ask(question, thread_id)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f'No se pudo completar la consulta ({type(exc).__name__}): {exc}')

with env_tab:
    st.caption('Cambia aquí las condiciones del entorno para observar cómo el agente detecta cambios y replanifica. '
               'Estos cálculos se hacen en Python, sin LLM.')
    st.subheader('Inventario simulado (editable)')
    edited = st.data_editor(ops.load_inventory(), hide_index=True, use_container_width=True, key='inventory_editor',
                            disabled=['id', 'name', 'unit'])
    col1, col2 = st.columns(2)
    if col1.button('Guardar cambios de stock'):
        try:
            ops.save_inventory([{**row, 'minimum': None if row['minimum'] is None or row['minimum'] != row['minimum'] else row['minimum']}
                                for row in edited])
            st.success('Inventario actualizado. El agente detectará el cambio en su próxima consulta.')
        except ValueError as exc:
            st.error(str(exc))
    if col2.button('Restablecer datos de demostración'):
        ops.reset_inventory()
        st.rerun()
    st.subheader('Prioridad de reposición')
    st.dataframe(ops.prioritize()['priorities'], hide_index=True, use_container_width=True)
    st.subheader('Facturación simulada')
    year = st.number_input('Año', min_value=2000, max_value=2100, value=2026)
    result_sales = ops.sales(int(year))
    st.dataframe(result_sales['months'], hide_index=True, use_container_width=True)
    st.caption(result_sales['coverage'])
    st.subheader('Borradores de órdenes de compra generados por el agente')
    orders = ops.list_orders()
    if not orders:
        st.caption('Aún no hay órdenes.')
    for order in orders:
        with st.expander(f"{order['id']} · {order['created_at']}"):
            st.caption(order['status'])
            st.dataframe(order['lines'], hide_index=True, use_container_width=True)
            if order['rejected']:
                st.write('Rechazadas:', order['rejected'])

with memory_tab:
    st.subheader('Memoria de largo plazo (compartida entre conversaciones)')
    st.caption('Preferencias, decisiones y episodios resumidos. Se recuperan por similitud semántica (FAISS, coseno).')
    records = long_term_memory().all()
    if records:
        st.dataframe(records, hide_index=True, use_container_width=True)
        col1, col2 = st.columns(2)
        selected = col1.selectbox('Recuerdo', [r['id'] for r in records])
        if col1.button('Eliminar recuerdo'):
            long_term_memory().delete(selected)
            st.rerun()
        if col2.button('Vaciar memoria de largo plazo'):
            long_term_memory().clear()
            st.rerun()
    else:
        st.caption('Sin recuerdos todavía.')
    st.subheader('Memoria de corto plazo del hilo actual')
    values = thread_values(st.session_state['thread_id'])
    st.write('Mensajes en ventana:', len(values.get('messages', [])), '· Turnos:', len(values.get('reports', [])))
    st.write('Último stock observado por el agente:', values.get('inventory_snapshot') or '—')

with help_tab:
    st.markdown(
        '1. Copia `.env.example` como `.env` y configura `GROQ_API_KEY`.\n'
        '2. Instala dependencias con `pip install -r requirements.txt`.\n'
        '3. Ejecuta `python create_vector_index.py`.\n'
        '4. Conversa en **Agente**. Cada respuesta muestra el plan, las decisiones y la traza.\n'
        '5. Modifica el stock en **Entorno simulado** y vuelve a preguntar para ver la replanificación.\n'
        '6. Revisa la **Memoria** de largo plazo y abre una conversación nueva para comprobar que persiste.')
    st.write('Fuentes vectoriales del prototipo:')
    for doc in json.loads((DATA / 'documents.json').read_text(encoding='utf-8')):
        st.write(f"{doc['id']} — {doc['title']} ({doc['type']})")
    st.caption('README.md y docs/ contienen la arquitectura, decisiones de diseño y evidencia de pruebas.')

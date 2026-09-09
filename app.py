import json
import os
import streamlit as st
from src.settings import load_settings, DATA, get_llm
from src.operations import inventory, sales
from src.retrieval import configured_retriever
from src.agent import Assistant

st.set_page_config(page_title='Valor Food Asistente', page_icon='🍽️', layout='wide')
load_settings()
st.title('Valor Food Asistente')
st.caption('Prototipo académico independiente · Matías Medina · DuocUC · 006D')
st.info('El inventario y las ventas son simulados. Esta aplicación no se conecta a Crispy House.')
with st.sidebar:
    st.header('Configuración')
    st.write('Proveedor de generación:', os.getenv('LLM_PROVIDER', 'openai'))
    st.write('Recuperación interna:', os.getenv('EMBEDDING_PROVIDER', 'local'), '+ FAISS')
    st.write('Respaldo externo:', 'búsqueda web global' if os.getenv('WEB_SEARCH_ENABLED', 'true').lower() in ('1','true','yes','si','sí','on') else 'desactivado')
    technique = st.selectbox('Técnica de prompt', ['zero_shot', 'few_shot'])
    st.caption('Primero se consultan las fuentes internas. La búsqueda web se usa solo como respaldo cuando falta información interna.')
    st.caption('Cada pregunta es independiente. La traza permite revisar datos, documentos y fuentes.')
chat, data_tab, help_tab = st.tabs(['Consultar con IA', 'Revisar datos sin IA', 'Instrucciones'])
with chat:
    st.write('Ejemplos: ¿Dónde veo el stock crítico? · ¿Cuánto queso falta para el mínimo? · ¿Cuál fue el mes con mayor facturación en 2026?')
    with st.form('question_form'):
        question = st.text_area('Escribe tu pregunta', max_chars=1000)
        submit = st.form_submit_button('Consultar')
    if submit:
        st.session_state.pop('last_result', None)
        if not question.strip():
            st.warning('Escribe una pregunta.')
        else:
            try:
                with st.spinner('Recuperando información y consultando el modelo…'):
                    result = Assistant(get_llm(), configured_retriever()).ask(question, technique)
                st.session_state['last_result'] = result
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f'No se pudo completar la consulta ({type(exc).__name__}): {exc}')
    if 'last_result' in st.session_state:
        result = st.session_state['last_result']
        st.subheader('Respuesta')
        st.write(result['answer'])
        for i, item in enumerate(result['trace']):
            with st.expander(f"{i+1}. {item['tool']}"):
                if item['tool'] == 'buscar_documentos' and isinstance(item['result'], list):
                    if not item['result']:
                        st.write('Sin fragmentos internos que superen el umbral configurado.')
                    for hit in item['result']:
                        st.markdown(f"**{hit['title']}**")
                        st.caption(f"{hit['chunk_id']} · interna · Similitud {hit['similarity']}")
                        st.write(hit['text'])
                        st.write('Fuente:', hit['source'])
                elif item['tool'] == 'buscar_web' and isinstance(item['result'], dict):
                    web_result = item['result']
                    st.caption('Motivo del respaldo externo: ' + str(web_result.get('reason', item.get('arguments', {}).get('reason', 'No indicado'))))
                    if web_result.get('error'):
                        st.warning(web_result['error'])
                    if not web_result.get('results'):
                        st.write('La búsqueda web no devolvió resultados utilizables.')
                    for hit in web_result.get('results', []):
                        st.markdown(f"**{hit['web_id']} · {hit['title']}**")
                        st.write(hit.get('snippet', ''))
                        st.write('Fuente externa:', hit['url'])
                else:
                    st.json(item)
        st.caption(result['validation'])
        st.json(result['usage'], expanded=False)
        st.download_button('Descargar evidencia JSON', json.dumps(result, ensure_ascii=False, indent=2), file_name='consulta-valor-food.json', mime='application/json')
with data_tab:
    st.caption('Estas tablas se calculan en Python, sin LLM ni llamadas de API. No son evidencia de generación con IA.')
    st.subheader('Inventario simulado y reposición hasta el mínimo')
    st.dataframe(inventory()['items'], hide_index=True, use_container_width=True)
    st.subheader('Facturación simulada')
    year = st.number_input('Año', min_value=2000, max_value=2100, value=2026)
    result_sales = sales(int(year))
    st.dataframe(result_sales['months'], hide_index=True, use_container_width=True)
    st.write('Meses con mayor facturación:', result_sales['best_months'])
    st.caption(result_sales['coverage'])
with help_tab:
    st.markdown('1. Copia `.env.example` como `.env` y configura la clave del LLM que usarás.\n2. Instala/actualiza dependencias con `pip install -r requirements.txt`.\n3. Ejecuta `python create_vector_index.py`.\n4. Utiliza **Consultar con IA**. El asistente prioriza fuentes internas y solo busca en Internet si no son suficientes.\n5. Revisa siempre la traza y las fuentes.')
    st.write('Las fuentes vectoriales del prototipo son:')
    for doc in json.loads((DATA / 'documents.json').read_text(encoding='utf-8')):
        st.write(f"{doc['id']} — {doc['title']} ({doc['type']})")
    st.caption('README.md contiene instalación, pruebas, limitaciones y pasos para GitHub.')

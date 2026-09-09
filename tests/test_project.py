import hashlib
import json
import math
from pathlib import Path
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from src.operations import inventory, sales
from src.retrieval import Retriever, chunks_from_file, DOCUMENTS
from src.agent import Assistant, make_tools
from src.prompts import make_prompt

class TestEmbeddings(Embeddings):
    """Vectores sintéticos por temas para pruebas mecánicas; NO embeddings semánticos reales."""
    __test__ = False
    def embed_query(self, text):
        text = text.lower()
        groups = [('stock', 'pdf', 'inicio', 'alerta'), ('oms', 'agua', 'alimentos', 'limpieza'), ('ventas', 'paid', 'facturación')]
        v = [float(sum(text.count(word) for word in group)) for group in groups]
        # Dimensión independiente para consultas sin vocabulario conocido.
        return v + [1. if not any(v) else 0.]
    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

@pytest.fixture
def retriever(tmp_path):
    Retriever.build(TestEmbeddings(), tmp_path, model='synthetic-test')
    return Retriever.load(TestEmbeddings(), tmp_path, model='synthetic-test')

def test_inventory():
    assert inventory('Queso')['items'][0]['buy'] == 6
    assert inventory('Harina')['items'][0]['buy'] == 0
    assert inventory('Aceite')['items'][0]['buy'] == 0
    assert inventory('Salsa')['items'][0]['buy'] is None
    assert inventory('inexistente')['items'] == []

def test_sales():
    result = sales(2026)
    assert result['best_months'] == [{'month': '2026-03', 'total': 1150000}]
    assert result['months'][1]['total'] == 900000
    assert len(result['months']) == 6
    assert sales(2025)['months'] == []
    with pytest.raises(ValueError): sales(True)
    with pytest.raises(ValueError): sales('2026')

def test_chunks_have_provenance():
    chunks = chunks_from_file()
    assert {d.metadata['type'] for d in chunks} == {'internal', 'external'}
    assert all(d.metadata['source'] and d.metadata['chunk_id'] and len(d.page_content) <= 600 for d in chunks)

def test_faiss_roundtrip_and_filter(retriever):
    hits = retriever.search('stock pdf')
    assert hits and hits[0]['id'] == 'VF-STOCK'
    external = retriever.search('OMS limpieza alimentos', 'external')
    assert external and all(x['type'] == 'external' for x in external)
    assert retriever.search('astronomía saturno') == []
    assert all(-1 <= x['similarity'] <= 1 for x in hits)

def test_changed_model_rejected(tmp_path):
    Retriever.build(TestEmbeddings(), tmp_path, model='synthetic-test')
    with pytest.raises(ValueError, match='modelo'):
        Retriever.load(TestEmbeddings(), tmp_path, model='other')

def test_changed_documents_rejected(tmp_path):
    corpus = tmp_path / 'docs.json'
    corpus.write_bytes(DOCUMENTS.read_bytes())
    dest = tmp_path / 'index'
    Retriever.build(TestEmbeddings(), dest, model='test', documents=corpus)
    corpus.write_text(corpus.read_text()+' ')
    with pytest.raises(ValueError, match='documentos'):
        Retriever.load(TestEmbeddings(), dest, model='test', documents=corpus)

def test_modified_index_rejected(tmp_path):
    Retriever.build(TestEmbeddings(), tmp_path)
    manifest = json.loads((tmp_path/'manifest.json').read_text())
    (tmp_path/manifest['index_file']).write_bytes(b'invalid')
    with pytest.raises(ValueError, match='alterado'):
        Retriever.load(TestEmbeddings(), tmp_path)

def test_tools_validate(retriever):
    tools = {t.name:t for t in make_tools(retriever)}
    assert tools['consultar_inventario'].invoke({'name':'queso'})['items'][0]['buy'] == 6
    with pytest.raises(ValueError): tools['comparar_ventas'].invoke({'year':'2026'})
    assert tools['buscar_documentos'].invoke({'query':'stock'})
    fake_web = lambda q: {'query': q, 'source_type': 'web', 'results': [{'web_id':'WEB-1','title':'Ejemplo','url':'https://example.com','snippet':'texto'}]}
    web_tools = {t.name:t for t in make_tools(retriever, fake_web)}
    web = web_tools['buscar_web'].invoke({'query':'tema externo','reason':'No existe información interna suficiente.'})
    assert web['results'][0]['web_id'] == 'WEB-1'
    assert web['reason'].startswith('No existe')

def test_prompt_techniques():
    zero = make_prompt('zero_shot').format_messages(context='[]', question='consulta')
    few = make_prompt('few_shot').format_messages(context='[]', question='consulta')
    assert len(few) == len(zero) + 4
    assert 'SIMULADOS' in zero[0].content
    with pytest.raises(ValueError): make_prompt('inventado')

class FakeLLM:
    def __init__(self): self.rounds = 0
    def bind_tools(self, tools): return self
    def invoke(self, messages):
        self.rounds += 1
        if self.rounds == 1:
            return AIMessage(content='', tool_calls=[{'name':'consultar_inventario','args':{'name':'queso'},'id':'call-1','type':'tool_call'}])
        assert json.loads(messages[-1].content)['items'][0]['buy'] == 6
        return AIMessage(content='Datos simulados: faltan 6 kg de queso [inventario simulado].')

def test_agent_tool_loop(retriever):
    model = FakeLLM()
    result = Assistant(model,retriever).ask('¿Cuánto queso falta?')
    assert model.rounds == 2
    assert result['trace'][0]['tool'] == 'buscar_documentos'
    assert result['trace'][1]['tool'] == 'consultar_inventario'
    assert '6 kg' in result['answer']
    assert result['simulated_data'] is True

def test_agent_no_fake_fallback(retriever):
    class Broken(FakeLLM):
        def invoke(self,messages): raise RuntimeError('proveedor no disponible')
    with pytest.raises(RuntimeError, match='proveedor'):
        Assistant(Broken(),retriever).ask('stock')

def test_agent_step_limit(retriever):
    class Endless(FakeLLM):
        def invoke(self,messages): return AIMessage(content='',tool_calls=[{'name':'consultar_inventario','args':{},'id':str(len(messages)),'type':'tool_call'}])
    with pytest.raises(RuntimeError, match='cuatro'):
        Assistant(Endless(),retriever).ask('stock')

"""Pruebas de componentes: operaciones, RAG, memoria de largo plazo, herramientas y planificación."""
import json
import pytest
from conftest import TestEmbeddings, HashEmbeddings, step
from src import operations as ops
from src.memory import LongTermMemory
from src.planner import heuristic_plan, order_steps
from src.prompts import build_messages
from src.retrieval import Retriever, chunks_from_file, DOCUMENTS
from src.settings import DATA
from src.tools import make_tools, CATEGORIES


# ---------------------------------------------------------------- operaciones
def test_inventory():
    assert ops.inventory('Queso')['items'][0]['buy'] == 6
    assert ops.inventory('Harina')['items'][0]['buy'] == 0
    assert ops.inventory('Aceite')['items'][0]['buy'] == 0
    assert ops.inventory('Salsa')['items'][0]['buy'] is None
    assert ops.inventory('inexistente')['items'] == []


def test_sales():
    result = ops.sales(2026)
    assert result['best_months'] == [{'month': '2026-03', 'total': 1150000}]
    assert result['months'][1]['total'] == 900000
    assert len(result['months']) == 6
    assert ops.sales(2025)['months'] == []
    with pytest.raises(ValueError): ops.sales(True)
    with pytest.raises(ValueError): ops.sales('2026')


def test_prioritize_orders_by_criticality():
    result = ops.prioritize()
    names = [x['name'] for x in result['priorities']]
    levels = [x['level'] for x in result['priorities']]
    # Queso 25 %, Carne 33 %, Pan 40 % del mínimo: críticos ordenados por cobertura.
    assert names[:3] == ['Queso', 'Carne', 'Pan']
    assert levels == sorted(levels, key=['critico', 'bajo', 'ok'].index)
    assert result['without_minimum'] == ['Salsa']
    assert ops.prioritize(['pan'])['excluded'] == ['Pan']


def test_purchase_order_validates_and_writes(isolated_runtime):
    result = ops.create_purchase_order(exclude=['Pan'], note='prueba')
    assert result['created'] and result['order_id'].startswith('OC-')
    assert {(l['name'], l['quantity']) for l in result['lines']} == {('Queso', 6.0), ('Carne', 6.0)}
    reasons = {r['name']: r['reason'] for r in result['rejected']}
    assert 'Excluido' in reasons['Pan'] and 'mínimo' in reasons['Salsa']
    saved = json.loads((isolated_runtime / 'ordenes' / f"{result['order_id']}.json").read_text(encoding='utf-8'))
    assert saved['status'].startswith('BORRADOR')
    assert (isolated_runtime / 'ordenes' / f"{result['order_id']}.csv").exists()
    explicit = ops.create_purchase_order([{'name': 'Queso', 'quantity': 999}, {'name': 'Caviar', 'quantity': 1}])
    assert not explicit['created'] and len(explicit['rejected']) == 2
    assert ops.list_orders()[0]['id'] == result['order_id']


def test_receipt_writes_runtime_only(isolated_runtime):
    seed = (DATA / 'inventory.json').read_text(encoding='utf-8')
    result = ops.register_receipt('queso', 5)
    assert result['stock_before'] == 2 and result['stock_after'] == 7 and result['buy_now'] == 1
    assert ops.inventory('Queso')['items'][0]['stock'] == 7
    assert (DATA / 'inventory.json').read_text(encoding='utf-8') == seed
    with pytest.raises(ValueError): ops.register_receipt('queso', -1)
    ops.reset_inventory()
    assert ops.inventory('Queso')['items'][0]['stock'] == 2


# ------------------------------------------------------------------------ RAG
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
    corpus.write_text(corpus.read_text() + ' ')
    with pytest.raises(ValueError, match='documentos'):
        Retriever.load(TestEmbeddings(), dest, model='test', documents=corpus)


def test_modified_index_rejected(tmp_path):
    Retriever.build(TestEmbeddings(), tmp_path)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    (tmp_path / manifest['index_file']).write_bytes(b'invalid')
    with pytest.raises(ValueError, match='alterado'):
        Retriever.load(TestEmbeddings(), tmp_path)


# ----------------------------------------------------- memoria de largo plazo
def test_long_term_memory_semantic_search_and_persistence(tmp_path):
    path = tmp_path / 'mem.json'
    memory = LongTermMemory(HashEmbeddings, 'hash-test', path)
    first = memory.add('preferencia', 'No incluir pan en las órdenes de compra porque lo trae la panadería')
    memory.add('decision', 'Se generó el borrador OC-1 con queso 6 kg')
    assert first['action'] == 'creado'
    # Consolidación: el mismo recuerdo no se duplica.
    again = memory.add('preferencia', 'No incluir pan en las órdenes de compra porque lo trae la panadería.')
    assert again['action'] == 'actualizado' and len(memory.all()) == 2
    reloaded = LongTermMemory(HashEmbeddings, 'hash-test', path)
    hits = reloaded.search('¿Qué hago con el pan en la orden de compra?', min_similarity=0.2)
    assert hits[0]['id'] == first['id'] and hits[0]['kind'] == 'preferencia'
    assert reloaded.search('pan orden', kinds=['decision'], min_similarity=0.0)[0]['kind'] == 'decision'
    assert reloaded.search('astronomía saturno júpiter') == []
    assert reloaded.delete(first['id']) and len(reloaded.all()) == 1
    with pytest.raises(ValueError): reloaded.add('otro', 'texto válido')


def test_memory_reembeds_when_model_changes(tmp_path):
    path = tmp_path / 'mem.json'
    LongTermMemory(HashEmbeddings, 'modelo-a', path).add('preferencia', 'Comprar carne los lunes')
    changed = LongTermMemory(HashEmbeddings, 'modelo-b', path)
    assert json.loads(path.read_text(encoding='utf-8'))['model'] == 'modelo-b'
    assert changed.search('carne lunes', min_similarity=0.2)


# -------------------------------------------------------------- herramientas
def test_tools_validate(retriever, memory):
    tools = {t.name: t for t in make_tools(retriever, memory)}
    assert set(tools) == set(CATEGORIES)
    assert tools['consultar_inventario'].invoke({'name': 'queso'})['items'][0]['buy'] == 6
    with pytest.raises(ValueError): tools['comparar_ventas'].invoke({'year': '2026'})
    with pytest.raises(ValueError): tools['generar_orden_compra'].invoke({'lineas': [{'nombre': 'Queso', 'cantidad': -2}]})
    assert tools['buscar_documentos'].invoke({'query': 'stock'})
    order = tools['generar_orden_compra'].invoke({'lineas': [{'nombre': 'Queso'}], 'nota': 'x'})
    assert order['created'] and order['memory']['kind'] == 'decision'
    assert tools['recordar'].invoke({'query': 'borrador queso'})[0]['kind'] == 'decision'
    assert tools['guardar_preferencia'].invoke({'texto': 'El dueño prefiere órdenes los lunes'})['action'] == 'creado'
    fake_web = lambda q: {'query': q, 'source_type': 'web', 'results': [{'web_id': 'WEB-1', 'title': 'Ejemplo', 'url': 'https://example.com', 'snippet': 'texto'}]}
    web = {t.name: t for t in make_tools(retriever, memory, fake_web)}['buscar_web'].invoke(
        {'query': 'tema externo', 'reason': 'No existe información interna suficiente.'})
    assert web['results'][0]['web_id'] == 'WEB-1' and web['reason'].startswith('No existe')

    def broken(q): raise TimeoutError('sin red')
    failed = {t.name: t for t in make_tools(retriever, memory, broken)}['buscar_web'].invoke({'query': 'x', 'reason': 'falta info'})
    assert 'TimeoutError' in failed['error']


# --------------------------------------------------------------- planificación
def test_plan_is_ordered_by_phase_then_priority():
    plan = order_steps([step('generar_orden_compra', 'alta'), step('buscar_web', 'alta'),
                        step('priorizar_reposicion', 'media'), step('recordar', 'baja'),
                        step('consultar_inventario', 'alta')])
    assert [s['herramienta'] for s in plan] == ['consultar_inventario', 'recordar', 'buscar_web',
                                               'priorizar_reposicion', 'generar_orden_compra']
    assert [s['id'] for s in plan] == [1, 2, 3, 4, 5] and all(s['estado'] == 'pendiente' for s in plan)


def test_heuristic_planner_uses_history_and_asks_when_needed():
    ask = heuristic_plan('¿Cuál fue el mejor mes?')
    assert ask['requiere_aclaracion'] and not ask['pasos']
    resolved = heuristic_plan('¿y el mejor mes?', history_text='Usuario: quiero revisar 2026')
    assert resolved['pasos'][0]['herramienta'] == 'comparar_ventas'
    order = order_steps(heuristic_plan('Genera la orden de compra con lo crítico')['pasos'])
    tools = [s['herramienta'] for s in order]
    assert tools[0] == 'consultar_inventario' and tools[-1] == 'generar_orden_compra'


def test_prompt_techniques():
    zero = build_messages('zero_shot', [], plan={'pasos': []})
    few = build_messages('few_shot', [])
    assert len(few) == len(zero) + 4
    assert 'SIMULADOS' in zero[0].content and 'PLAN ACTUAL' in zero[0].content
    with pytest.raises(ValueError): build_messages('inventado', [])

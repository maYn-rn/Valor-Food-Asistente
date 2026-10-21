import re
import unicodedata
import zlib
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from src.memory import LongTermMemory
from src.retrieval import Retriever


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


class HashEmbeddings(Embeddings):
    """Bolsa de raíces (5 primeras letras) proyectada a 128 dimensiones. Determinista y sin red."""

    def embed_query(self, text):
        text = ''.join(c for c in unicodedata.normalize('NFD', text.lower()) if unicodedata.category(c) != 'Mn')
        vector = [0.0] * 128
        for word in re.findall(r'[a-z0-9]{3,}', text):
            vector[zlib.crc32(word[:5].encode()) % 128] += 1.0
        return vector

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


class ScriptedLLM:
    """LLM simulado: entrega planes y respuestas en el orden programado.

    Cada elemento puede ser un valor o una función(messages) que permite
    verificar lo que el grafo envió al modelo. Un plan que sea una excepción se lanza.
    """

    def __init__(self, plans=(), replies=()):
        self.plans, self.replies = list(plans), list(replies)
        self.plan_calls, self.reply_calls = [], []

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, method=None):
        llm = self

        class Planner:
            def invoke(self, messages):
                llm.plan_calls.append(messages)
                item = llm.plans.pop(0) if llm.plans else {'objetivo': 'responder', 'pasos': []}
                item = item(messages) if callable(item) else item
                if isinstance(item, Exception):
                    raise item
                return schema.model_validate(item)
        return Planner()

    def invoke(self, messages):
        self.reply_calls.append(messages)
        if not self.replies:
            return AIMessage(content='Respuesta final de prueba.')
        item = self.replies.pop(0)
        return item(messages) if callable(item) else item


def call(name, args, call_id=None):
    return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': call_id or f'call-{name}', 'type': 'tool_call'}])


def step(tool, priority='alta', goal='paso de prueba'):
    return {'herramienta': tool, 'objetivo': goal, 'prioridad': priority}


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    """Cada prueba usa su propia carpeta runtime: el inventario semilla nunca se modifica."""
    monkeypatch.setenv('VALOR_RUNTIME_DIR', str(tmp_path / 'runtime'))
    return tmp_path / 'runtime'


@pytest.fixture
def retriever(tmp_path):
    Retriever.build(TestEmbeddings(), tmp_path / 'index', model='synthetic-test')
    return Retriever.load(TestEmbeddings(), tmp_path / 'index', model='synthetic-test')


@pytest.fixture
def memory(tmp_path):
    return LongTermMemory(HashEmbeddings, 'hash-test', tmp_path / 'memoria.json')

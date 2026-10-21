"""Memoria del agente.

- Corto plazo: historial de cada conversación (thread_id) persistido por el
  checkpointer SQLite de LangGraph. Sobrevive a recargas de la interfaz.
- Largo plazo: recuerdos semánticos (preferencias, decisiones y episodios
  resumidos) que se comparten entre conversaciones y se recuperan por similitud
  coseno con FAISS. Se guardan en JSON con sus vectores (sin pickle).
"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver
from .settings import runtime_dir
from .vectors import flat_index, cosine_search

KINDS = ('preferencia', 'decision', 'episodio')
TOPICS = ('ordenes_compra', 'reposicion', 'ventas', 'navegacion', 'general')
DEDUP_SIMILARITY = 0.90


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def open_checkpointer(path=None):
    path = Path(path or runtime_dir() / 'conversaciones.sqlite')
    # check_same_thread=False: Streamlit atiende cada interacción en hilos distintos.
    saver = SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
    saver.setup()
    return saver


def list_threads(checkpointer):
    seen = []
    for checkpoint in checkpointer.list(None):
        thread = checkpoint.config['configurable']['thread_id']
        if thread not in seen:
            seen.append(thread)
    return seen


class LongTermMemory:
    def __init__(self, embeddings_factory, signature, path=None):
        """embeddings_factory se invoca solo al buscar o escribir (carga perezosa del modelo)."""
        self.path = Path(path or runtime_dir() / 'memoria_largo_plazo.json')
        self.signature = signature
        self._factory, self._embeddings, self._index = embeddings_factory, None, None
        self.records = self._read()

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = self._factory()
        return self._embeddings

    def _read(self):
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding='utf-8'))
        records = data.get('records', [])
        if records and data.get('model') != self.signature:
            # Cambió el modelo de embeddings: se re-vectorizan los recuerdos existentes.
            vectors = self.embeddings.embed_documents([r['text'] for r in records])
            for record, vector in zip(records, vectors):
                record['vector'] = [float(x) for x in vector]
            self.records = records
            self._write()
        return records

    def _write(self):
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps({'model': self.signature, 'records': self.records}, ensure_ascii=False, indent=1),
                        encoding='utf-8')
        temp.replace(self.path)
        self._index = None

    def _ranked(self, vector, k):
        if not self.records:
            return []
        if self._index is None:
            self._index = flat_index([r['vector'] for r in self.records])
        return cosine_search(self._index, vector, k)

    def add(self, kind, text, thread_id=None, topic='general'):
        if kind not in KINDS:
            raise ValueError(f'Tipo de recuerdo inválido; usa {", ".join(KINDS)}.')
        if topic not in TOPICS:
            raise ValueError(f'Tema inválido; usa {", ".join(TOPICS)}.')
        if not isinstance(text, str) or not 3 <= len(text.strip()) <= 500:
            raise ValueError('El recuerdo debe tener entre 3 y 500 caracteres.')
        text = text.strip()
        vector = [float(x) for x in self.embeddings.embed_query(text)]
        # Consolidación: un recuerdo casi idéntico del mismo tipo se actualiza en vez de duplicarse.
        for position, similarity in self._ranked(vector, 3):
            record = self.records[position]
            if record['kind'] == kind and similarity >= DEDUP_SIMILARITY:
                record.update(text=text, vector=vector, topic=topic, updated_at=now(), thread_id=thread_id or record.get('thread_id'))
                self._write()
                return {'action': 'actualizado', 'id': record['id'], 'kind': kind, 'topic': topic, 'text': text}
        record = {'id': 'M-' + uuid.uuid4().hex[:8], 'kind': kind, 'topic': topic, 'text': text, 'thread_id': thread_id,
                  'created_at': now(), 'updated_at': now(), 'vector': vector}
        self.records.append(record)
        self._write()
        return {'action': 'creado', 'id': record['id'], 'kind': kind, 'topic': topic, 'text': text}

    def search(self, query, k=4, min_similarity=0.35, kinds=None, topics=None):
        """Búsqueda semántica con filtros opcionales de metadatos (tipo y tema)."""
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            raise ValueError('Consulta de memoria inválida.')
        if not self.records:
            return []
        hits = []
        for position, similarity in self._ranked(self.embeddings.embed_query(query), len(self.records)):
            record = self.records[position]
            if kinds and record['kind'] not in kinds:
                continue
            if topics and record.get('topic', 'general') not in topics:
                continue
            if similarity >= min_similarity:
                hits.append({k: record[k] for k in ('id', 'kind', 'text', 'created_at')}
                            | {'topic': record.get('topic', 'general'), 'similarity': round(similarity, 4)})
            if len(hits) == k:
                break
        return hits

    def all(self):
        return [{k: v for k, v in r.items() if k != 'vector'} for r in self.records]

    def delete(self, record_id):
        before = len(self.records)
        self.records = [r for r in self.records if r['id'] != record_id]
        if len(self.records) != before:
            self._write()
        return before != len(self.records)

    def clear(self):
        self.records = []
        self._write()

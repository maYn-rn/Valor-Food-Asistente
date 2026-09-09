import hashlib
import json
import os
import uuid
from pathlib import Path
import faiss
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .settings import DATA, INDEX, load_settings, get_embeddings, embedding_signature

DOCUMENTS = DATA / 'documents.json'

def corpus_hash(path=DOCUMENTS):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def chunks_from_file(path=DOCUMENTS):
    records = json.loads(Path(path).read_text(encoding='utf-8'))
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100,
                                             separators=['\n\n', '\n', '. ', ' ', ''])
    chunks = []
    for row in records:
        parts = splitter.split_text(row['text'])
        for i, text in enumerate(parts, 1):
            chunks.append(Document(page_content=text, metadata={k: row[k] for k in ('id', 'type', 'title', 'source', 'date')}))
            chunks[-1].metadata['chunk_id'] = f"{row['id']}-{i}"
    return chunks

class Retriever:
    def __init__(self, store, top_k=3, min_similarity=.25):
        if not 1 <= top_k <= 10 or not -1 <= min_similarity <= 1:
            raise ValueError('Parámetros de recuperación fuera de rango.')
        self.store, self.top_k, self.min_similarity = store, top_k, min_similarity

    def search(self, query, source_type=None):
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            raise ValueError('Consulta documental inválida.')
        if source_type not in (None, 'internal', 'external'):
            raise ValueError('Tipo de fuente inválido.')
        result = self.store.similarity_search_with_score(query, k=self.top_k,
                  filter={'type': source_type} if source_type else None, fetch_k=self.store.index.ntotal)
        hits = []
        for doc, distance in result:
            # Índice L2 con vectores unitarios: coseno = 1 - distancia cuadrada / 2.
            similarity = float(max(-1, min(1, 1 - float(distance) / 2)))
            if similarity >= self.min_similarity:
                hits.append({**doc.metadata, 'text': doc.page_content, 'similarity': round(similarity, 5)})
        return hits

    @classmethod
    def build(cls, embeddings, destination=INDEX, model='test', documents=DOCUMENTS):
        chunks = chunks_from_file(documents)
        store = FAISS.from_documents(chunks, embeddings, normalize_L2=True)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        records = [store.docstore.search(store.index_to_docstore_id[i]) for i in range(store.index.ntotal)]
        # Se evita pickle: FAISS binario local más metadatos JSON. Publicar manifest al final.
        name = 'index-' + uuid.uuid4().hex + '.faiss'
        faiss.write_index(store.index, str(destination / name))
        metadata = {'model': model, 'dimension': store.index.d, 'corpus_sha256': corpus_hash(documents),
                    'index_file': name, 'index_sha256': hashlib.sha256((destination / name).read_bytes()).hexdigest(),
                    'chunk_size': 600, 'chunk_overlap': 100,
                    'documents': [{'text': d.page_content, 'metadata': d.metadata} for d in records]}
        temp = destination / 'manifest.tmp'
        temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(destination / 'manifest.json')
        for old in destination.glob('index-*.faiss'):
            if old.name != name:
                old.unlink()
        return cls(store)

    @classmethod
    def load(cls, embeddings, destination=INDEX, model='test', documents=DOCUMENTS, top_k=3, min_similarity=.25):
        destination = Path(destination)
        manifest_file = destination / 'manifest.json'
        if not manifest_file.exists():
            raise ValueError('Falta el índice. Ejecuta python create_vector_index.py.')
        metadata = json.loads(manifest_file.read_text(encoding='utf-8'))
        if metadata['model'] != model or metadata['corpus_sha256'] != corpus_hash(documents):
            raise ValueError('Cambió el modelo de embeddings o los documentos. Reconstruye el índice.')
        filename = metadata['index_file']
        if Path(filename).name != filename or not filename.endswith('.faiss'):
            raise ValueError('Nombre de índice inválido.')
        index_path = destination / filename
        if hashlib.sha256(index_path.read_bytes()).hexdigest() != metadata['index_sha256']:
            raise ValueError('Índice alterado o incompleto. Reconstrúyelo.')
        index = faiss.read_index(str(index_path))
        docs = {str(i): Document(page_content=r['text'], metadata=r['metadata']) for i, r in enumerate(metadata['documents'])}
        if index.ntotal != len(docs) or index.d != metadata['dimension']:
            raise ValueError('Índice y metadatos no coinciden.')
        store = FAISS(embeddings, index, InMemoryDocstore(docs), {i: str(i) for i in range(len(docs))}, normalize_L2=True)
        return cls(store, top_k=top_k, min_similarity=min_similarity)

def configured_retriever():
    load_settings()
    return Retriever.load(get_embeddings(), model=embedding_signature(),
                          top_k=int(os.getenv('TOP_K', '3')), min_similarity=float(os.getenv('MIN_SIMILARITY', '.25')))

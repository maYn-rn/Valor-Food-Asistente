import hashlib
import json
import os
import uuid
from pathlib import Path
import faiss
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .settings import DATA, INDEX, load_settings, get_embeddings, embedding_signature
from .vectors import flat_index, cosine_search

DOCUMENTS = DATA / 'documents.json'
CHUNK_SIZE, CHUNK_OVERLAP = 600, 100


def corpus_hash(path=DOCUMENTS):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chunks_from_file(path=DOCUMENTS):
    records = json.loads(Path(path).read_text(encoding='utf-8'))
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                             separators=['\n\n', '\n', '. ', ' ', ''])
    chunks = []
    for row in records:
        parts = splitter.split_text(row['text'])
        for i, text in enumerate(parts, 1):
            chunks.append(Document(page_content=text, metadata={k: row[k] for k in ('id', 'type', 'title', 'source', 'date')}))
            chunks[-1].metadata['chunk_id'] = f"{row['id']}-{i}"
    return chunks


class Retriever:
    """Recuperación semántica de documentación (RAG) sobre un índice FAISS coseno."""

    def __init__(self, embeddings, index, docs, top_k=3, min_similarity=.25):
        if not 1 <= top_k <= 10 or not -1 <= min_similarity <= 1:
            raise ValueError('Parámetros de recuperación fuera de rango.')
        if index.ntotal != len(docs):
            raise ValueError('Índice y metadatos no coinciden.')
        self.embeddings, self.index, self.docs = embeddings, index, docs
        self.top_k, self.min_similarity = top_k, min_similarity

    def search(self, query, source_type=None):
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            raise ValueError('Consulta documental inválida.')
        if source_type not in (None, 'internal', 'external'):
            raise ValueError('Tipo de fuente inválido.')
        # Se ordena todo el corpus (pequeño) y luego se filtra por tipo de fuente.
        ranked = cosine_search(self.index, self.embeddings.embed_query(query), self.index.ntotal)
        hits = []
        for position, similarity in ranked:
            doc = self.docs[position]
            if source_type and doc.metadata['type'] != source_type:
                continue
            if similarity >= self.min_similarity:
                hits.append({**doc.metadata, 'text': doc.page_content, 'similarity': round(similarity, 5)})
            if len(hits) == self.top_k:
                break
        return hits

    @classmethod
    def build(cls, embeddings, destination=INDEX, model='test', documents=DOCUMENTS):
        chunks = chunks_from_file(documents)
        index = flat_index(embeddings.embed_documents([d.page_content for d in chunks]))
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        # Se evita pickle: FAISS binario local más metadatos JSON. Publicar manifest al final.
        name = 'index-' + uuid.uuid4().hex + '.faiss'
        faiss.write_index(index, str(destination / name))
        metadata = {'model': model, 'metric': 'cosine', 'dimension': index.d, 'corpus_sha256': corpus_hash(documents),
                    'index_file': name, 'index_sha256': hashlib.sha256((destination / name).read_bytes()).hexdigest(),
                    'chunk_size': CHUNK_SIZE, 'chunk_overlap': CHUNK_OVERLAP,
                    'documents': [{'text': d.page_content, 'metadata': d.metadata} for d in chunks]}
        temp = destination / 'manifest.tmp'
        temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(destination / 'manifest.json')
        for old in destination.glob('index-*.faiss'):
            if old.name != name:
                old.unlink()
        return cls(embeddings, index, chunks)

    @classmethod
    def load(cls, embeddings, destination=INDEX, model='test', documents=DOCUMENTS, top_k=3, min_similarity=.25):
        destination = Path(destination)
        manifest_file = destination / 'manifest.json'
        if not manifest_file.exists():
            raise ValueError('Falta el índice. Ejecuta python create_vector_index.py.')
        metadata = json.loads(manifest_file.read_text(encoding='utf-8'))
        if metadata.get('metric') != 'cosine':
            raise ValueError('Índice de una versión anterior. Reconstruye el índice con python create_vector_index.py.')
        if metadata['model'] != model or metadata['corpus_sha256'] != corpus_hash(documents):
            raise ValueError('Cambió el modelo de embeddings o los documentos. Reconstruye el índice.')
        filename = metadata['index_file']
        if Path(filename).name != filename or not filename.endswith('.faiss'):
            raise ValueError('Nombre de índice inválido.')
        index_path = destination / filename
        if hashlib.sha256(index_path.read_bytes()).hexdigest() != metadata['index_sha256']:
            raise ValueError('Índice alterado o incompleto. Reconstrúyelo.')
        index = faiss.read_index(str(index_path))
        docs = [Document(page_content=r['text'], metadata=r['metadata']) for r in metadata['documents']]
        if index.d != metadata['dimension']:
            raise ValueError('Índice y metadatos no coinciden.')
        return cls(embeddings, index, docs, top_k=top_k, min_similarity=min_similarity)


def configured_retriever(embeddings=None):
    load_settings()
    return Retriever.load(embeddings or get_embeddings(), model=embedding_signature(),
                          top_k=int(os.getenv('TOP_K', '3')), min_similarity=float(os.getenv('MIN_SIMILARITY', '.25')))

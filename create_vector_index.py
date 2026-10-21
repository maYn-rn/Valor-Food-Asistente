from src.settings import get_embeddings, embedding_signature, INDEX, load_settings
from src.retrieval import Retriever

def main():
    load_settings()
    try:
        retriever = Retriever.build(get_embeddings(), model=embedding_signature())
        print(f'Índice FAISS creado: {retriever.index.ntotal} fragmentos en {INDEX}.')
    except ValueError as exc:
        raise SystemExit(str(exc))
    except Exception as exc:
        raise SystemExit(f'No se pudo crear el índice ({type(exc).__name__}): {exc}')
if __name__ == '__main__':
    main()

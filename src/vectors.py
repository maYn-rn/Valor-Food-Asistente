"""Utilidades FAISS compartidas por la recuperación documental y la memoria de largo plazo.

Se usa FAISS directamente (IndexFlatIP sobre vectores unitarios = similitud coseno)
en lugar del wrapper de langchain-community, que está en proceso de retiro.
"""
import faiss
import numpy as np


def unit(vectors):
    """Normaliza filas a norma 1. Un vector nulo queda en cero (similitud 0 con todo)."""
    matrix = np.asarray(vectors, dtype='float32')
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def flat_index(vectors):
    matrix = unit(vectors)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def cosine_search(index, query_vector, k):
    """Devuelve [(posición, similitud)] ordenados de mayor a menor similitud."""
    if index.ntotal == 0:
        return []
    scores, ids = index.search(unit(query_vector), min(k, index.ntotal))
    return [(int(i), float(max(-1.0, min(1.0, s)))) for i, s in zip(ids[0], scores[0]) if i >= 0]

# Búsqueda semántica sobre el recetario ya ingerido (ver rag/ingesta.py).
# Usado tanto por el agente de recetas de CrewAI (como tool) como por el nodo
# de recetas de LangGraph.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb

from rag.config import CHROMA_COLLECTION, CHROMA_DIR
from rag.embeddings import embed_textos


def buscar_receta(consulta: str, k: int = 3) -> list[str]:
    """Devuelve hasta k fragmentos de receta relevantes para la consulta.
    Lista vacía si el recetario no se ha ingerido todavía (ver rag/ingesta.py)
    o si no hay ningún fragmento almacenado."""
    cliente = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        coleccion = cliente.get_collection(CHROMA_COLLECTION)
    except Exception:
        return []

    total = coleccion.count()
    if total == 0:
        return []

    (embedding_consulta,) = embed_textos([consulta])
    resultado = coleccion.query(query_embeddings=[embedding_consulta], n_results=min(k, total))
    return resultado["documents"][0]

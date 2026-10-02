# Búsqueda semántica sobre el recetario ingerido. Solo acceso a datos: juzgar y
# reformular lo hace orquestacion_langgraph/rag_agentico/.

import chromadb

from rag.config import CHROMA_COLLECTION, CHROMA_DIR
from rag.embeddings import embed_textos


def buscar_receta_detallado(consulta: str, k: int = 3) -> list[dict]:
    """Como buscar_receta(), con fuente y distancia por fragmento.

    La distancia es solo informativa, no un umbral (ver rag_agentico/README.md).
    Lista vacía si el recetario no se ha ingerido.
    """
    cliente = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        coleccion = cliente.get_collection(CHROMA_COLLECTION)
    except Exception:
        return []

    total = coleccion.count()
    if total == 0:
        return []

    (embedding_consulta,) = embed_textos([consulta])
    resultado = coleccion.query(
        query_embeddings=[embedding_consulta],
        n_results=min(k, total),
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "texto": texto,
            "fuente": (metadatos or {}).get("fuente", "desconocida"),
            "distancia": round(float(distancia), 4),
        }
        for texto, metadatos, distancia in zip(
            resultado["documents"][0],
            resultado["metadatas"][0],
            resultado["distances"][0],
        )
    ]


def fragmentos_de_fuente(fuente: str) -> list[dict]:
    """Todos los fragmentos de un archivo, en orden, para reconstruir la receta
    completa. Lista vacía si la colección no existe."""
    cliente = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        coleccion = cliente.get_collection(CHROMA_COLLECTION)
    except Exception:
        return []

    resultado = coleccion.get(
        where={"fuente": fuente},
        include=["documents", "metadatas"],
    )

    fragmentos = [
        {
            "texto": texto,
            "fuente": (metadatos or {}).get("fuente", fuente),
            # Índices viejos no guardaban `orden`.
            "orden": (metadatos or {}).get("orden", 0),
        }
        for texto, metadatos in zip(resultado["documents"], resultado["metadatas"])
    ]
    return sorted(fragmentos, key=lambda f: f["orden"])


def buscar_receta(consulta: str, k: int = 3) -> list[str]:
    """Hasta k fragmentos relevantes, solo texto (lo usa la tool de CrewAI)."""
    return [f["texto"] for f in buscar_receta_detallado(consulta, k)]

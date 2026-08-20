# Búsqueda semántica sobre el recetario ya ingerido (ver rag/ingesta.py).
#
# Esta es la capa de ACCESO A DATOS: solo consulta la base vectorial y
# devuelve lo que encuentra. No decide si el resultado sirve, no reformula la
# consulta y no habla con el LLM — de eso se encarga la capa de orquestación
# (ver orquestacion_langgraph/rag_agentico/).
#
# Consumidores:
#   - orquestacion_crewai/agentes.py  -> buscar_receta() como tool
#   - orquestacion_langgraph/rag_agentico/nodos.py -> buscar_receta_detallado()

import chromadb

from rag.config import CHROMA_COLLECTION, CHROMA_DIR
from rag.embeddings import embed_textos


def buscar_receta_detallado(consulta: str, k: int = 3) -> list[dict]:
    """Igual que buscar_receta() pero devuelve, por fragmento, además del texto:
    el archivo del que salió y la distancia vectorial que reportó ChromaDB.

    La distancia es informativa: se registra en la traza para poder analizarla
    en el notebook, pero NO se usa como umbral de corte. Un umbral numérico
    tendría que calibrarse contra un conjunto de consultas etiquetadas, y ese
    trabajo no está hecho; quien decide si un fragmento sirve es el nodo
    evaluador (un LLM), no un número elegido a dedo.

    Lista vacía si el recetario no se ha ingerido todavía (correr
    'python -m rag.ingesta') o si no hay ningún fragmento almacenado.
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


def buscar_receta(consulta: str, k: int = 3) -> list[str]:
    """Devuelve hasta k fragmentos de receta relevantes para la consulta, solo
    como texto. Lo usa la tool de CrewAI, donde el framework espera un string.
    Para la versión con metadatos ver buscar_receta_detallado()."""
    return [f["texto"] for f in buscar_receta_detallado(consulta, k)]

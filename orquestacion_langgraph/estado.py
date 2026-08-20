# Estado que viaja entre los nodos del grafo principal.

from typing import Optional, TypedDict


class EstadoConversacion(TypedDict):
    consulta: str
    intencion: Optional[str]
    razonamiento: Optional[str]
    respuesta: Optional[str]

    # Traza interna del subgrafo de RAG agéntico, cuando la consulta pasó por
    # el nodo de recetas. Se propaga hacia arriba solo para poder inspeccionarla
    # (demo y notebook comparativo); ningún nodo del grafo principal la lee para
    # tomar decisiones. Es None para las demás intenciones.
    traza_rag: Optional[list[dict]]

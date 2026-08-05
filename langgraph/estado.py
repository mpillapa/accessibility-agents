# Estado que viaja entre los nodos del grafo.

from typing import Optional, TypedDict


class EstadoConversacion(TypedDict):
    consulta: str
    intencion: Optional[str]
    razonamiento: Optional[str]
    respuesta: Optional[str]

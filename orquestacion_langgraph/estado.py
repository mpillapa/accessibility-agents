# Estado que viaja entre los nodos del grafo principal.

from typing import Optional, TypedDict


class EstadoConversacion(TypedDict):
    consulta: str
    intencion: Optional[str]
    razonamiento: Optional[str]
    respuesta: Optional[str]

    # --- Entrada por voz (None cuando la consulta entra como texto) ---

    # Audio a transcribir. Su presencia es lo que hace que el grafo arranque
    # por el nodo de voz en vez de ir directo al Orchestrator.
    ruta_audio: Optional[str]

    # Qué devolvió el ASR, incluidas las señales que NO son el texto:
    # `sin_voz`, probabilidad de idioma, duración. Se guarda completo aunque la
    # transcripción se descarte, porque es la evidencia de por qué se descartó.
    transcripcion: Optional[dict]

    # Por qué no se pudo confiar en la transcripción, o None si sí se pudo.
    # Cuando no es None, el grafo NO rutea: pide que repitan. Ver
    # orquestacion_langgraph/voz.py para el porqué de esta regla.
    entrada_descartada: Optional[str]

    # Traza interna del subgrafo de RAG agéntico, cuando la consulta pasó por
    # el nodo de recetas. Se propaga hacia arriba solo para poder inspeccionarla
    # (demo y notebook comparativo); ningún nodo del grafo principal la lee para
    # tomar decisiones. Es None para las demás intenciones.
    traza_rag: Optional[list[dict]]

# Estado que viaja entre los nodos del grafo principal.

from typing import Optional, TypedDict


class EstadoConversacion(TypedDict):
    consulta: str
    intencion: Optional[str]
    razonamiento: Optional[str]
    respuesta: Optional[str]

    # Perfil de medicacion/datos/perfiles.json. Si es None se usa
    # agentes.PERFIL_POR_DEFECTO (solo prototipo; ver medicacion/README.md).
    id_perfil: Optional[str]

    # Entrada por voz (None si la consulta entra como texto).
    # Si hay audio, el grafo arranca por el nodo de voz.
    ruta_audio: Optional[str]

    # Salida completa del ASR; se guarda aunque se descarte, como evidencia.
    transcripcion: Optional[dict]

    # Motivo del descarte; si no es None, el grafo pide que repitan (voz.py).
    entrada_descartada: Optional[str]

    # Traza del subgrafo de RAG, solo para inspección; ningún nodo la lee.
    traza_rag: Optional[list[dict]]

    # Tarea medicamento × comida. Cada rama paralela escribe su propio campo:
    # si escribieran el mismo, LangGraph no sabría cuál conservar.
    medicamentos_vigentes: Optional[list[str]]
    recetas_encontradas: Optional[dict]
    # Resultado de interacciones/reglas.py.
    cruce: Optional[dict]

# Construye el StateGraph: Orchestrator -> (edge condicional) -> especialista -> END.
# Esto es lo que en CrewAI se resolvía con allow_delegation/tools; aquí el flujo
# de datos entre nodos y las transiciones quedan explícitas en el grafo.

from langgraph.graph import StateGraph, START, END

from orquestacion_langgraph.estado import EstadoConversacion
from orquestacion_langgraph.agentes import (
    nodo_orchestrator,
    ruta_siguiente_nodo,
    nodo_medicacion,
    nodo_recetas,
    nodo_familia_stub,
    nodo_emergencia,
    nodo_small_talk,
)
from orquestacion_langgraph.voz import (
    RAMA_CONTINUAR,
    RAMA_DESCARTAR,
    nodo_transcribir_voz,
    ruta_tras_transcribir,
    nodo_no_se_entendio,
)


def _estado_inicial(consulta: str, ruta_audio: str | None = None) -> dict:
    return {
        "consulta": consulta,
        "intencion": None,
        "razonamiento": None,
        "respuesta": None,
        "traza_rag": None,
        "ruta_audio": ruta_audio,
        "transcripcion": None,
        "entrada_descartada": None,
    }


# Por dónde entra el grafo: si hay audio, hay que transcribirlo y verificarlo
# antes de clasificar nada. Si la consulta ya viene como texto, va directo al
# Orchestrator (comportamiento original, intacto).
def _ruta_de_entrada(estado: EstadoConversacion) -> str:
    return "entra por voz" if estado.get("ruta_audio") else "entra por texto"


def construir_grafo():
    grafo = StateGraph(EstadoConversacion)

    grafo.add_node("transcribir_voz", nodo_transcribir_voz)
    grafo.add_node("no_se_entendio", nodo_no_se_entendio)
    grafo.add_node("orchestrator", nodo_orchestrator)
    grafo.add_node("medicacion", nodo_medicacion)
    grafo.add_node("recetas", nodo_recetas)
    grafo.add_node("familia", nodo_familia_stub)
    grafo.add_node("emergencia", nodo_emergencia)
    grafo.add_node("small_talk", nodo_small_talk)

    # Las claves de estos mapas son las etiquetas que aparecen en el diagrama:
    # describen POR QUÉ se toma cada rama, no a dónde va.
    grafo.add_conditional_edges(START, _ruta_de_entrada, {
        "entra por voz": "transcribir_voz",
        "entra por texto": "orchestrator",
    })

    # El guardrail del ASR, explícito en la topología: una transcripción que no
    # se pudo verificar NO llega al Orchestrator. Ver voz.py.
    grafo.add_conditional_edges("transcribir_voz", ruta_tras_transcribir, {
        RAMA_CONTINUAR: "orchestrator",
        RAMA_DESCARTAR: "no_se_entendio",
    })
    grafo.add_edge("no_se_entendio", END)

    grafo.add_conditional_edges("orchestrator", ruta_siguiente_nodo, {
        "MEDICATION_HEALTH": "medicacion",
        "RECIPE_MULTIMEDIA": "recetas",
        "FAMILY_COMMUNICATION": "familia",
        "EMERGENCY": "emergencia",
        "SMALL_TALK": "small_talk",
    })
    grafo.add_edge("medicacion", END)
    grafo.add_edge("recetas", END)
    grafo.add_edge("familia", END)
    grafo.add_edge("emergencia", END)
    grafo.add_edge("small_talk", END)

    return grafo.compile()


def procesar_consulta(consulta: str, ruta_audio: str | None = None) -> dict:
    import time

    app = construir_grafo()
    inicio = time.time()
    resultado = app.invoke(_estado_inicial(consulta, ruta_audio))
    latencia = time.time() - inicio

    return {
        # Con entrada por voz, la consulta la escribe el ASR: se devuelve la del
        # estado final, no la que se pasó por parámetro (que va vacía).
        "consulta": resultado.get("consulta") or consulta,
        "intencion": resultado["intencion"],
        "razonamiento": resultado.get("razonamiento"),
        "respuesta": resultado["respuesta"],
        "traza_rag": resultado.get("traza_rag"),
        "transcripcion": resultado.get("transcripcion"),
        "entrada_descartada": resultado.get("entrada_descartada"),
        "latencia_segundos": round(latencia, 2),
    }


def procesar_audio(ruta_audio: str) -> dict:
    """Entrada por voz: transcribe el audio y lo procesa como una consulta.

    Si el ASR no da una transcripción confiable, el grafo NO clasifica: pide
    que repitan y devuelve `entrada_descartada` con el motivo. Ver
    orquestacion_langgraph/voz.py.
    """
    return procesar_consulta(consulta="", ruta_audio=ruta_audio)


# Igual que procesar_consulta, pero usando app.stream() en vez de app.invoke().
# stream_mode="updates" entrega, nodo por nodo, SOLO lo que ese nodo escribió
# en el estado compartido — es la comunicación real entre agentes: el
# Orchestrator no le "pasa un mensaje" al especialista, escribe en el estado
# y el especialista lee de ahí. Sirve para mostrar ese intercambio en vivo,
# equivalente a verbose=True en CrewAI pero explícito por nodo.
def procesar_consulta_verbose(consulta: str) -> dict:
    import time

    app = construir_grafo()
    estado = _estado_inicial(consulta)
    estado_acumulado = dict(estado)

    inicio = time.time()
    for actualizacion in app.stream(estado, stream_mode="updates"):
        for nodo, cambios in actualizacion.items():
            print(f"[NODO: {nodo}]")
            for clave, valor in cambios.items():
                print(f"    {clave} -> {valor!r}")
            print()
            estado_acumulado.update(cambios)
    latencia = time.time() - inicio

    return {
        "consulta": consulta,
        "intencion": estado_acumulado["intencion"],
        "razonamiento": estado_acumulado.get("razonamiento"),
        "respuesta": estado_acumulado["respuesta"],
        "traza_rag": estado_acumulado.get("traza_rag"),
        "latencia_segundos": round(latencia, 2),
    }


# Igual que traza_por_nodo(), pero ENTREGANDO cada paso apenas ocurre en vez de
# esperar a que el grafo termine. Lo consume la interfaz web para mostrar por
# dónde va el sistema mientras trabaja.
#
# Por qué existe: una consulta con RAG tarda entre 6 y 30 segundos, y el 88% se
# va en el nodo `generar`. Sin retroalimentación la interfaz parece colgada, que
# fue exactamente el problema al presentar la demo. Mostrar el avance no acelera
# nada, pero convierte una espera opaca en uno donde se ve qué está pasando.
#
# Entrega dicts {"nodo": str, "cambios": dict}; el último trae el estado final
# acumulado bajo la clave "estado_final".
def procesar_consulta_en_vivo(consulta: str, ruta_audio: str | None = None):
    import time

    app = construir_grafo()
    estado_acumulado = _estado_inicial(consulta, ruta_audio)

    inicio = time.time()
    for actualizacion in app.stream(_estado_inicial(consulta, ruta_audio), stream_mode="updates"):
        for nodo, cambios in actualizacion.items():
            estado_acumulado.update(cambios)
            yield {"nodo": nodo, "cambios": cambios}

    yield {
        "nodo": None,
        "cambios": {},
        "estado_final": {
            "consulta": estado_acumulado.get("consulta") or consulta,
            "intencion": estado_acumulado.get("intencion"),
            "razonamiento": estado_acumulado.get("razonamiento"),
            "respuesta": estado_acumulado.get("respuesta"),
            "traza_rag": estado_acumulado.get("traza_rag"),
            "transcripcion": estado_acumulado.get("transcripcion"),
            "entrada_descartada": estado_acumulado.get("entrada_descartada"),
            "latencia_segundos": round(time.time() - inicio, 2),
        },
    }


# traza_por_nodo() devuelve el mismo intercambio que imprime
# procesar_consulta_verbose(), pero COMO DATOS (lista de dicts), sin imprimir,
# para que el notebook comparativo pueda renderizar el cruce de información
# entre agentes (qué escribió cada nodo en el estado compartido).
def traza_por_nodo(consulta: str) -> list[dict]:
    app = construir_grafo()
    traza = []
    for actualizacion in app.stream(_estado_inicial(consulta), stream_mode="updates"):
        for nodo, cambios in actualizacion.items():
            traza.append({"nodo": nodo, "cambios": cambios})
    return traza


# Solo la decisión de clasificación del Orchestrator (una llamada al LLM), sin
# ejecutar el especialista. Es el ruteo NATIVO de LangGraph: es exactamente el
# valor que el edge condicional (ruta_siguiente_nodo) usa para decidir a qué
# nodo saltar. Comparable con crewai.agentes.clasificar_consulta().
def clasificar_consulta(consulta: str) -> str:
    return nodo_orchestrator(_estado_inicial(consulta))["intencion"]

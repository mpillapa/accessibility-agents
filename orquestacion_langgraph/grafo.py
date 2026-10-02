# Construye el StateGraph: Orchestrator -> (edge condicional) -> especialista -> END,
# y expone las formas de ejecutarlo (invoke, stream, traza por nodo).

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
    nodo_medicacion_cruce,
    nodo_recetas_cruce,
    nodo_integrador,
    RAMAS_CRUCE,
)
from orquestacion_langgraph.voz import (
    RAMA_CONTINUAR,
    RAMA_DESCARTAR,
    nodo_transcribir_voz,
    ruta_tras_transcribir,
    nodo_no_se_entendio,
)


def _estado_inicial(consulta: str, ruta_audio: str | None = None,
                    id_perfil: str | None = None) -> dict:
    return {
        "consulta": consulta,
        "id_perfil": id_perfil,
        "intencion": None,
        "razonamiento": None,
        "respuesta": None,
        "traza_rag": None,
        "ruta_audio": ruta_audio,
        "transcripcion": None,
        "entrada_descartada": None,
        "medicamentos_vigentes": None,
        "recetas_encontradas": None,
        "cruce": None,
    }


# Con audio, primero se transcribe y verifica; con texto, directo al Orchestrator.
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
    grafo.add_node("medicacion_cruce", nodo_medicacion_cruce)
    grafo.add_node("recetas_cruce", nodo_recetas_cruce)
    grafo.add_node("integrador", nodo_integrador)

    # Las claves son las etiquetas del diagrama: dicen por qué se toma la rama.
    grafo.add_conditional_edges(START, _ruta_de_entrada, {
        "entra por voz": "transcribir_voz",
        "entra por texto": "orchestrator",
    })

    # Guardrail del ASR: lo no verificado no llega al Orchestrator (voz.py).
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
        # Fan-out en paralelo.
        RAMAS_CRUCE[0]: "medicacion_cruce",
        RAMAS_CRUCE[1]: "recetas_cruce",
    })
    # El integrador espera a las dos ramas.
    grafo.add_edge(["medicacion_cruce", "recetas_cruce"], "integrador")
    grafo.add_edge("integrador", END)
    grafo.add_edge("medicacion", END)
    grafo.add_edge("recetas", END)
    grafo.add_edge("familia", END)
    grafo.add_edge("emergencia", END)
    grafo.add_edge("small_talk", END)

    return grafo.compile()


def procesar_consulta(consulta: str, ruta_audio: str | None = None,
                      id_perfil: str | None = None) -> dict:
    import time

    app = construir_grafo()
    inicio = time.time()
    resultado = app.invoke(_estado_inicial(consulta, ruta_audio, id_perfil))
    latencia = time.time() - inicio

    return {
        # Con voz, la consulta la escribe el ASR; el parámetro llega vacío.
        "consulta": resultado.get("consulta") or consulta,
        "intencion": resultado["intencion"],
        "razonamiento": resultado.get("razonamiento"),
        "respuesta": resultado["respuesta"],
        "traza_rag": resultado.get("traza_rag"),
        "transcripcion": resultado.get("transcripcion"),
        "entrada_descartada": resultado.get("entrada_descartada"),
        "cruce": resultado.get("cruce"),
        "latencia_segundos": round(latencia, 2),
    }


def procesar_audio(ruta_audio: str, id_perfil: str | None = None) -> dict:
    """Entrada por voz. Si la transcripción no es confiable, devuelve `entrada_descartada`."""
    return procesar_consulta(consulta="", ruta_audio=ruta_audio, id_perfil=id_perfil)


# Como procesar_consulta, pero imprime lo que escribe cada nodo en el estado
# (stream_mode="updates").
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


# Generador para la interfaz web: entrega {"nodo", "cambios"} por paso y al
# final un dict con "estado_final" (bitácora 11.5). `config` es el
# RunnableConfig que usa medicion/campana.py para LangSmith y tokens por nodo.
def procesar_consulta_en_vivo(consulta: str, ruta_audio: str | None = None,
                              id_perfil: str | None = None, config: dict | None = None):
    import time

    app = construir_grafo()
    estado_acumulado = _estado_inicial(consulta, ruta_audio, id_perfil)

    inicio = time.time()
    for actualizacion in app.stream(_estado_inicial(consulta, ruta_audio, id_perfil),
                                    config=config, stream_mode="updates"):
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
            "cruce": estado_acumulado.get("cruce"),
            "latencia_segundos": round(time.time() - inicio, 2),
        },
    }


# Lo mismo que imprime procesar_consulta_verbose(), como lista de dicts (notebook).
def traza_por_nodo(consulta: str) -> list[dict]:
    app = construir_grafo()
    traza = []
    for actualizacion in app.stream(_estado_inicial(consulta), stream_mode="updates"):
        for nodo, cambios in actualizacion.items():
            traza.append({"nodo": nodo, "cambios": cambios})
    return traza


# Solo la intención del Orchestrator, sin ejecutar el especialista.
def clasificar_consulta(consulta: str) -> str:
    return nodo_orchestrator(_estado_inicial(consulta))["intencion"]

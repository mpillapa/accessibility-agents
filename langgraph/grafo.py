# Construye el StateGraph: Orchestrator -> (edge condicional) -> especialista -> END.
# Esto es lo que en CrewAI se resolvía con allow_delegation/tools; aquí el flujo
# de datos entre nodos y las transiciones quedan explícitas en el grafo.

from langgraph.graph import StateGraph, START, END

from estado import EstadoConversacion
from agentes import (
    nodo_orchestrator,
    ruta_siguiente_nodo,
    nodo_medicacion,
    nodo_recetas,
    nodo_familia_stub,
    nodo_emergencia_stub,
    nodo_small_talk,
)


def _estado_inicial(consulta: str) -> dict:
    return {"consulta": consulta, "intencion": None, "razonamiento": None, "respuesta": None}


def construir_grafo():
    grafo = StateGraph(EstadoConversacion)

    grafo.add_node("orchestrator", nodo_orchestrator)
    grafo.add_node("medicacion", nodo_medicacion)
    grafo.add_node("recetas", nodo_recetas)
    grafo.add_node("familia", nodo_familia_stub)
    grafo.add_node("emergencia", nodo_emergencia_stub)
    grafo.add_node("small_talk", nodo_small_talk)

    grafo.add_edge(START, "orchestrator")
    grafo.add_conditional_edges("orchestrator", ruta_siguiente_nodo, {
        "medicacion": "medicacion",
        "recetas": "recetas",
        "familia": "familia",
        "emergencia": "emergencia",
        "small_talk": "small_talk",
    })
    grafo.add_edge("medicacion", END)
    grafo.add_edge("recetas", END)
    grafo.add_edge("familia", END)
    grafo.add_edge("emergencia", END)
    grafo.add_edge("small_talk", END)

    return grafo.compile()


def procesar_consulta(consulta: str) -> dict:
    import time

    app = construir_grafo()
    inicio = time.time()
    resultado = app.invoke(_estado_inicial(consulta))
    latencia = time.time() - inicio

    return {
        "consulta": consulta,
        "intencion": resultado["intencion"],
        "razonamiento": resultado.get("razonamiento"),
        "respuesta": resultado["respuesta"],
        "latencia_segundos": round(latencia, 2),
    }


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
        "latencia_segundos": round(latencia, 2),
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

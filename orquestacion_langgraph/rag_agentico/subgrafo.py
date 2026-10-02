# Construcción del subgrafo de RAG agéntico, con el ciclo reformular -> recuperar.
# Diagrama del flujo y reglas en README.md de este paquete.

from langgraph.graph import StateGraph, START, END

from orquestacion_langgraph.rag_agentico.estado import EstadoRAG
from orquestacion_langgraph.rag_agentico.nodos import (
    nodo_decidir_busqueda,
    nodo_evaluar_relevancia,
    nodo_expandir_contexto,
    nodo_generar,
    nodo_recuperar,
    nodo_reformular,
    nodo_responder_sin_recetario,
    nodo_sin_resultado,
    ruta_tras_decidir,
    ruta_tras_evaluar,
)


def construir_subgrafo_rag(con_generacion: bool = True):
    """Con `con_generacion=False` termina tras expandir el contexto, sin redactar.

    Lo usa la tarea medicamento × comida; se ahorra `generar` (~88% del tiempo, bitácora 11).
    """
    grafo = StateGraph(EstadoRAG)

    grafo.add_node("decidir_busqueda", nodo_decidir_busqueda)
    grafo.add_node("recuperar", nodo_recuperar)
    grafo.add_node("evaluar_relevancia", nodo_evaluar_relevancia)
    grafo.add_node("reformular", nodo_reformular)
    grafo.add_node("expandir_contexto", nodo_expandir_contexto)
    grafo.add_node("generar", nodo_generar)
    grafo.add_node("sin_resultado", nodo_sin_resultado)
    grafo.add_node("responder_sin_recetario", nodo_responder_sin_recetario)

    grafo.add_edge(START, "decidir_busqueda")
    grafo.add_conditional_edges("decidir_busqueda", ruta_tras_decidir, {
        "recuperar": "recuperar",
        "responder_sin_recetario": "responder_sin_recetario",
    })

    grafo.add_edge("recuperar", "evaluar_relevancia")
    grafo.add_conditional_edges("evaluar_relevancia", ruta_tras_evaluar, {
        "generar": "expandir_contexto",
        "reformular": "reformular",
        "sin_resultado": "sin_resultado",
    })

    # El arco que cierra el ciclo.
    grafo.add_edge("reformular", "recuperar")

    if con_generacion:
        grafo.add_edge("expandir_contexto", "generar")
        grafo.add_edge("generar", END)
    else:
        grafo.add_edge("expandir_contexto", END)
    grafo.add_edge("sin_resultado", END)
    grafo.add_edge("responder_sin_recetario", END)

    return grafo.compile()


def _estado_inicial(consulta: str) -> dict:
    return {
        "consulta": consulta,
        "necesita_recetario": None,
        "consulta_busqueda": None,
        "intentos": 0,
        "fragmentos": None,
        "fragmentos_utiles": None,
        "fragmentos_contexto": None,
        "respuesta": None,
        "hubo_resultado": None,
        "traza": [],
    }


def consultar_recetario(consulta: str, con_generacion: bool = True) -> dict:
    """Punto de entrada del subgrafo: respuesta, traza y metadatos.

    `fuentes` son las recetas que aprobó el evaluador. Con `con_generacion=False`,
    `respuesta` es None."""
    resultado = construir_subgrafo_rag(con_generacion).invoke(_estado_inicial(consulta))
    utiles = resultado.get("fragmentos_utiles") or []
    hubo_resultado = resultado["hubo_resultado"]
    if hubo_resultado is None and resultado.get("necesita_recetario"):
        # Sin generación nadie escribe hubo_resultado: se deduce del filtro.
        hubo_resultado = bool(utiles)

    return {
        "respuesta": resultado["respuesta"],
        "hubo_resultado": hubo_resultado,
        "intentos": resultado["intentos"],
        "fuentes": list(dict.fromkeys(f["fuente"] for f in utiles)),
        "traza": resultado["traza"],
    }

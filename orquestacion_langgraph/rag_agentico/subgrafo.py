# Construcción del subgrafo de RAG agéntico.
#
# Forma del grafo:
#
#   START -> decidir_busqueda
#              |-- (no necesita) --> responder_sin_recetario --> END
#              `-- (sí necesita) --> recuperar
#                                       |
#                                       v
#                                  evaluar_relevancia
#                                       |
#              .------------------------+------------------------.
#              |                        |                        |
#         (hay útiles)          (sin útiles,            (sin útiles,
#              |                 quedan intentos)        sin intentos)
#              v                        |                        |
#      expandir_contexto            reformular              sin_resultado
#              |                        |                        |
#              v                        `--> recuperar           v
#           generar                          (ciclo)            END
#              |
#              v
#             END
#
# El ciclo reformular -> recuperar es lo que distingue esto de un RAG lineal:
# el grafo puede volver sobre sus pasos. En un pipeline fijo, una búsqueda que
# falla por vocabulario termina en una respuesta mala o inventada.
#
# expandir_contexto está entre el filtro y el generador porque el filtro deja
# pasar fragmentos sueltos: de una receta troceada en cinco párrafos puede
# aprobar uno, y redactar con ese solo da un paso aislado en vez de la receta.

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


def construir_subgrafo_rag():
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

    grafo.add_edge("expandir_contexto", "generar")
    grafo.add_edge("generar", END)
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


def consultar_recetario(consulta: str) -> dict:
    """Punto de entrada del subgrafo. Devuelve la respuesta más la traza del
    ciclo, para que el grafo principal pueda pasarla hacia arriba y el
    notebook comparativo pueda mostrar qué hizo el RAG paso a paso."""
    resultado = construir_subgrafo_rag().invoke(_estado_inicial(consulta))

    return {
        "respuesta": resultado["respuesta"],
        "hubo_resultado": resultado["hubo_resultado"],
        "intentos": resultado["intentos"],
        "traza": resultado["traza"],
    }

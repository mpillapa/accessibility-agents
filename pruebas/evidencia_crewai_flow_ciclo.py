# Evidencia: un Flow de CrewAI SÍ expresa el ciclo de reformulación del RAG
# agéntico, con estado tipado que se acumula entre intentos.
#
# Uso (desde la raíz del repo):
#   python -m pruebas.evidencia_crewai_flow_ciclo
#
# No necesita LLM, VPN ni GPU: la "evaluación" rechaza los dos primeros intentos
# a propósito para forzar el ciclo.
#
# POR QUÉ EXISTE
# --------------
# La tesis partía de que CrewAI "solo soporta arquitecturas jerárquicas y
# secuenciales y no maneja bien el estado" (bitácora 4.1, dicho por los
# tutores). Eso vale para la abstracción `Crew` (Process.sequential /
# hierarchical). CrewAI 1.15.5 trae además `Flow`, con @start, @listen y @router,
# y este script muestra que un Flow recorre
#
#   decidir -> recuperar -> evaluar -> reformular -> recuperar -> ... -> generar
#
# que es la misma estructura del subgrafo de orquestacion_langgraph/rag_agentico.
#
# Detalle observado al escribirlo: el valor que devuelve un @listen NO dispara
# otra rama; solo lo hace un @router. Para cerrar el ciclo hizo falta un @router
# extra (`volver_a_recuperar`) que en LangGraph es una sola línea
# (`add_edge("reformular", "recuperar")`). Es una diferencia de ergonomía, no de
# capacidad.
#
# No se cuenta entre las pruebas automáticas del proyecto: es evidencia de una
# afirmación de la tesis, no una prueba del sistema.

import sys

from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, or_, router, start

INTENTOS_HASTA_ACEPTAR = 3


class EstadoRAG(BaseModel):
    consulta: str = "el arrocito dulce"
    intento: int = 0
    traza: list[str] = []
    resultado: str = ""


class FlowRAG(Flow[EstadoRAG]):
    @start()
    def decidir(self):
        self.state.traza.append("decidir")

    @listen(or_(decidir, "reformulada"))
    def recuperar(self):
        self.state.intento += 1
        self.state.traza.append(f"recuperar#{self.state.intento}")

    @router(recuperar)
    def evaluar(self):
        self.state.traza.append("evaluar")
        return "sirve" if self.state.intento >= INTENTOS_HASTA_ACEPTAR else "no_sirve"

    @listen("no_sirve")
    def reformular(self):
        self.state.consulta += " (reformulada)"
        self.state.traza.append("reformular")

    @router(reformular)
    def volver_a_recuperar(self):
        return "reformulada"

    @listen("sirve")
    def generar(self):
        self.state.traza.append("generar")
        self.state.resultado = "OK"


def main():
    flow = FlowRAG()
    flow.kickoff()
    traza = " -> ".join(flow.state.traza)
    print(f"\ntraza: {traza}")

    esperada = ["decidir"]
    for i in range(1, INTENTOS_HASTA_ACEPTAR + 1):
        esperada += [f"recuperar#{i}", "evaluar"]
        if i < INTENTOS_HASTA_ACEPTAR:
            esperada.append("reformular")
    esperada.append("generar")

    ok = flow.state.traza == esperada and flow.state.resultado == "OK"
    print("RESULTADO:", "el Flow recorrió el ciclo con estado acumulado" if ok
          else f"NO coincide con lo esperado: {esperada}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

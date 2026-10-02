# Evidencia (bitácora 4.1): un Flow de CrewAI expresa el ciclo de reformulación
# del RAG agéntico, con estado tipado que se acumula entre intentos.
#
# Uso (desde la raíz del repo):
#   python -m pruebas.evidencia_crewai_flow_ciclo
#
# No necesita LLM, VPN ni GPU: la "evaluación" rechaza los dos primeros intentos
# a propósito para forzar el ciclo. No es parte de las pruebas automáticas.

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

    # Un @listen no dispara otra rama; solo un @router cierra el ciclo.
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

# Pruebas de la lógica del subgrafo de RAG agéntico, con dobles del LLM y de ChromaDB.
# Uso: python -m pruebas.prueba_ciclo_rag
# No requiere VPN. Sin pytest para no sumar dependencias al prototipo.

import sys

from orquestacion_langgraph.rag_agentico import nodos
from orquestacion_langgraph.rag_agentico.estado import MAX_INTENTOS_RECUPERACION
from orquestacion_langgraph.rag_agentico.subgrafo import consultar_recetario


class RespuestaFalsa:
    def __init__(self, content):
        self.content = content


class LLMFalso:
    """Responde según el tipo de prompt. `veredictos`: cola de 'SI'/'NO', uno por fragmento."""

    def __init__(self, decision="BUSCAR", veredictos=None, reformulacion="arroz con leche"):
        self.decision = decision
        self.veredictos = list(veredictos or [])
        self.reformulacion = reformulacion
        self.llamadas = []

    def invoke(self, prompt):
        if "DECISION: BUSCAR" in prompt:
            self.llamadas.append("decidir")
            return RespuestaFalsa(f"Razono brevemente.\nDECISION: {self.decision}")

        if "VEREDICTO: SI" in prompt:
            self.llamadas.append("evaluar")
            veredicto = self.veredictos.pop(0) if self.veredictos else "NO"
            return RespuestaFalsa(f"Razono brevemente.\nVEREDICTO: {veredicto}")

        if "Reescribe la consulta" in prompt:
            self.llamadas.append("reformular")
            return RespuestaFalsa(self.reformulacion)

        self.llamadas.append("generar")
        return RespuestaFalsa("Receta explicada paso a paso.")


class BusquedaFalsa:
    """Devuelve fragmentos fijos y cuenta las llamadas."""

    def __init__(self, fragmentos_por_llamada):
        self.fragmentos_por_llamada = fragmentos_por_llamada
        self.llamadas = 0
        self.consultas = []

    def __call__(self, consulta, k=3):
        self.consultas.append(consulta)
        i = min(self.llamadas, len(self.fragmentos_por_llamada) - 1)
        self.llamadas += 1
        return self.fragmentos_por_llamada[i]


class FuenteFalsa:
    """Doble de rag.buscar.fragmentos_de_fuente."""

    def __init__(self, por_fuente):
        self.por_fuente = por_fuente
        self.consultadas = []

    def __call__(self, fuente):
        self.consultadas.append(fuente)
        return self.por_fuente.get(fuente, [])


FRAGMENTO = [{"texto": "Arroz con leche: hervir el arroz...", "fuente": "arroz.txt",
              "distancia": 0.2, "orden": 1}]

RECETA_COMPLETA = {
    "arroz.txt": [
        {"texto": "Arroz con leche", "fuente": "arroz.txt", "orden": 0},
        {"texto": "Arroz con leche: hervir el arroz...", "fuente": "arroz.txt", "orden": 1},
        {"texto": "Agregar la canela y la leche condensada.", "fuente": "arroz.txt", "orden": 2},
        {"texto": "Servir frío, espolvoreado con canela.", "fuente": "arroz.txt", "orden": 3},
    ]
}


def preparar(llm_falso, busqueda_falsa, fuente_falsa=None):
    nodos.llm = llm_falso
    nodos.buscar_receta_detallado = busqueda_falsa
    nodos.fragmentos_de_fuente = fuente_falsa or FuenteFalsa(RECETA_COMPLETA)


def caso_camino_directo():
    """Encuentra algo relevante en el primer intento: no debe reformular."""
    llm = LLMFalso(veredictos=["SI"])
    busqueda = BusquedaFalsa([FRAGMENTO])
    preparar(llm, busqueda)

    r = consultar_recetario("léeme la receta del arroz con leche")

    assert r["hubo_resultado"] is True, "debió responder con la receta"
    assert r["intentos"] == 1, f"debió buscar una sola vez, buscó {r['intentos']}"
    assert "reformular" not in llm.llamadas, "no debía reformular si encontró algo útil"
    assert [p["nodo"] for p in r["traza"]] == [
        "decidir_busqueda", "recuperar", "evaluar_relevancia",
        "expandir_contexto", "generar",
    ], f"camino inesperado: {[p['nodo'] for p in r['traza']]}"
    return "camino directo (encuentra en el primer intento)"


def caso_expande_a_la_receta_completa():
    """El filtro aprueba fragmentos sueltos; sin expansión sale un paso aislado (bitácora 3.1)."""
    llm = LLMFalso(veredictos=["SI"])
    busqueda = BusquedaFalsa([FRAGMENTO])
    fuente = FuenteFalsa(RECETA_COMPLETA)
    preparar(llm, busqueda, fuente)

    r = consultar_recetario("receta del arroz con leche")

    paso_expansion = next(p for p in r["traza"] if p["nodo"] == "expandir_contexto")
    assert paso_expansion["fragmentos_aprobados"] == 1, "el filtro debía aprobar 1 fragmento"
    assert paso_expansion["fragmentos_en_contexto"] == 4, (
        f"debió expandir a los 4 fragmentos de la receta, expandió a "
        f"{paso_expansion['fragmentos_en_contexto']}"
    )
    assert fuente.consultadas == ["arroz.txt"], f"consultó {fuente.consultadas}"

    paso_generar = next(p for p in r["traza"] if p["nodo"] == "generar")
    assert paso_generar["fragmentos_usados"] == 4, "el generador debía recibir la receta completa"
    return "expande de 1 fragmento aprobado a la receta completa (4)"


def caso_reformula_y_acierta():
    """La receta está, pero el usuario la nombró de otra manera."""
    llm = LLMFalso(veredictos=["NO", "SI"], reformulacion="arroz con leche")
    busqueda = BusquedaFalsa([FRAGMENTO, FRAGMENTO])
    preparar(llm, busqueda)

    r = consultar_recetario("eso dulce del arrocito que hacía mi mamá")

    assert r["hubo_resultado"] is True, "debió terminar respondiendo la receta"
    assert r["intentos"] == 2, f"debió buscar dos veces, buscó {r['intentos']}"
    assert "reformular" in llm.llamadas, "debió reformular tras el veredicto NO"
    assert busqueda.consultas[1] == "arroz con leche", (
        f"la segunda búsqueda debió usar la consulta reformulada, usó {busqueda.consultas[1]!r}"
    )
    return "reformula y acierta (2 intentos)"


def caso_se_rinde_sin_inventar():
    """Sin nada relevante: respeta el tope y no le pide al LLM que redacte."""
    llm = LLMFalso(veredictos=["NO"] * 10)
    busqueda = BusquedaFalsa([FRAGMENTO])
    preparar(llm, busqueda)

    r = consultar_recetario("quiero preparar sushi")

    assert r["hubo_resultado"] is False, "debió admitir que no encontró la receta"
    assert r["intentos"] == MAX_INTENTOS_RECUPERACION, (
        f"debió parar en {MAX_INTENTOS_RECUPERACION} intentos, hizo {r['intentos']}"
    )
    assert "generar" not in llm.llamadas, (
        "no debía pedirle al LLM que redacte: es la vía por la que se inventan recetas"
    )
    assert r["traza"][-1]["nodo"] == "sin_resultado"
    return f"se rinde sin inventar (tope de {MAX_INTENTOS_RECUPERACION} intentos)"


def caso_no_consulta_recetario():
    """Un agradecimiento no necesita el recetario: no debe tocar la base."""
    llm = LLMFalso(decision="RESPONDER_DIRECTO")
    busqueda = BusquedaFalsa([FRAGMENTO])
    preparar(llm, busqueda)

    r = consultar_recetario("gracias mijito, ya me salió rico")

    assert busqueda.llamadas == 0, "no debía buscar en el recetario"
    assert r["traza"][-1]["nodo"] == "responder_sin_recetario"
    return "decide no consultar el recetario"


CASOS = [
    caso_camino_directo,
    caso_expande_a_la_receta_completa,
    caso_reformula_y_acierta,
    caso_se_rinde_sin_inventar,
    caso_no_consulta_recetario,
]


def main():
    print("Pruebas del ciclo de RAG agéntico (con dobles, sin VPN)\n")
    fallos = 0
    for caso in CASOS:
        try:
            print(f"  OK    {caso()}")
        except AssertionError as e:
            fallos += 1
            print(f"  FALLA {caso.__name__}: {e}")
        except Exception as e:
            fallos += 1
            print(f"  ERROR {caso.__name__}: {type(e).__name__}: {e}")

    print(f"\n{len(CASOS) - fallos}/{len(CASOS)} pruebas pasaron")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())

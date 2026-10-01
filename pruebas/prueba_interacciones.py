# Pruebas de la tarea medicamento × comida: el cruce y el camino por el grafo.
#
# Uso (desde la raíz del repo):
#   python -m pruebas.prueba_interacciones
#
# NO requieren VPN ni LLM. El cruce (interacciones/reglas.py) es código puro y
# se prueba contra la verdad de referencia escrita a mano
# (interacciones/datos/casos_prueba.json). El grafo se prueba con dobles en
# lugar del LLM y de ChromaDB: lo que se verifica es el CAMINO (orquestador →
# dos ramas en paralelo → integrador) y que el integrador cruce lo que las
# ramas le pasan, no la calidad de la redacción.
#
# La integridad de los datos es de pruebas/prueba_datos_interacciones.py.
#
# Escrito sin pytest, igual que el resto de pruebas/ (ver prueba_ciclo_rag.py).

import sys
from datetime import date

from interacciones import reglas
from interacciones.agente import TEXTO_NO_EVALUABLE, prompt_integrador, redactar
from interacciones.datos import cargar_casos

HOY = date.fromisoformat(cargar_casos()["_fecha_referencia"])


# --- El cruce contra la verdad de referencia --------------------------------

def prueba_casos_de_desarrollo():
    """Los 11 casos escritos a mano, con la fuente dada (sin RAG)."""
    for caso in cargar_casos()["desarrollo"]["casos"]:
        r = reglas.cruzar(caso["usuario"], [caso["fuente"]], HOY)
        esperado = caso["esperado"]
        assert r["evaluable"] == esperado["evaluable"], f"{caso['id']}: evaluable={r['evaluable']} ({r['motivo']})"
        assert r["hay_interaccion"] == esperado["hay_interaccion"], caso["id"]
        assert reglas.pares(r) == {tuple(p) for p in esperado["interacciones"]}, f"{caso['id']}: {reglas.pares(r)}"
    return "el cruce coincide con los 11 casos de desarrollo escritos a mano"


def prueba_casos_de_campana():
    """Las 20 combinaciones usuario × frase, con cada fuente aceptada."""
    total = 0
    for frase in cargar_casos()["campana"]["frases"]:
        for usuario, esperado in frase["esperado_por_usuario"].items():
            for fuente in frase["fuentes_aceptadas"]:
                r = reglas.cruzar(usuario, [fuente], HOY)
                assert reglas.pares(r) == {tuple(p) for p in esperado["interacciones"]}, f"{frase['id']}/{usuario}/{fuente}"
                total += 1
    return f"el cruce coincide en las {total} combinaciones de la campaña (incluye las dos fuentes de llapingacho)"


# --- Reglas que los perfiles reales no ejercitan -------------------------------

def _con_perfil_sintetico(funcion_renal: str):
    """Reemplaza perfil y prescripción por los de una persona inventada que toma
    losartán. Ningún perfil real combina losartán con función renal reducida,
    así que la regla del potasio no se prueba de otra forma."""
    perfil = {"id": "sintetico", "condiciones": ["hipertension"], "alergias": [], "funcion_renal": funcion_renal}
    receta = {"id": "rx-sint", "id_perfil": "sintetico", "vigente_hasta": None,
              "indicaciones": [{"medicamento": "Losartan"}]}
    reglas.obtener_perfil = lambda _id: perfil
    reglas.prescripciones_de = lambda _id: [receta]


def _restaurar():
    from medicacion import datos
    reglas.obtener_perfil = datos.obtener_perfil
    reglas.prescripciones_de = datos.prescripciones_de


def prueba_potasio_solo_con_funcion_renal_reducida():
    try:
        _con_perfil_sintetico("reducida")
        r = reglas.cruzar("sintetico", ["bolon de verde.png"], HOY)
        assert reglas.pares(r) == {("Losartan", "alto_potasio")}, reglas.pares(r)
        assert r["veredicto"] == reglas.VEREDICTO_PRECAUCION
        _con_perfil_sintetico("normal")
        r = reglas.cruzar("sintetico", ["bolon de verde.png"], HOY)
        assert not r["hay_interaccion"], "con función renal normal el potasio no debe alertar"
    finally:
        _restaurar()
    return "losartán × potasio alerta con función renal reducida y no con normal (perfil sintético)"


def prueba_veredicto_es_el_mas_grave():
    assert reglas.cruzar("rosa", ["receta5.jpeg"], HOY)["veredicto"] == reglas.VEREDICTO_EVITAR  # metformina: evitar
    assert reglas.cruzar("manuel", ["receta7.jpeg"], HOY)["veredicto"] == reglas.VEREDICTO_PRECAUCION
    assert reglas.cruzar("jorge", ["sopa_de_verduras.txt"], HOY)["veredicto"] == reglas.VEREDICTO_SIN_INTERACCION
    assert reglas.cruzar("luis", ["receta5.jpeg"], HOY)["veredicto"] == reglas.VEREDICTO_NO_EVALUABLE
    return "el veredicto toma la severidad más grave (evitar > precaución > sin interacción)"


def prueba_no_evaluable_sin_receta_reconocida():
    r = reglas.cruzar("rosa", ["no_existe.jpg"], HOY)
    assert not r["evaluable"] and r["motivo"] == reglas.MOTIVO_SIN_RECETA
    assert r["fuentes_desconocidas"] == ["no_existe.jpg"]
    r = reglas.cruzar("rosa", [], HOY)
    assert r["motivo"] == reglas.MOTIVO_SIN_RECETA, "si el RAG no encontró nada, no se puede decir que no hay problema"
    return "sin receta reconocida no es evaluable (no se responde 'no hay problema')"


def prueba_prescripcion_vencida_no_cuenta():
    """rx-manuel-002 (Amoxicilina) venció el 2026-09-27."""
    assert "Amoxicilina" in reglas.medicamentos_vigentes("manuel", date(2026, 9, 20))
    assert "Amoxicilina" not in reglas.medicamentos_vigentes("manuel", date(2026, 10, 1))
    return "solo cuentan las prescripciones vigentes a la fecha"


def prueba_varias_fuentes_se_cruzan_todas():
    r = reglas.cruzar("elena", ["sopa_de_verduras.txt", "tigrillo.png"], HOY)
    assert reglas.pares(r) == {("Melatonina", "cafeina")}
    assert {i["fuente"] for i in r["interacciones"]} == {"tigrillo.png"}, "cada interacción dice de qué receta sale"
    return "con varias recetas se cruzan todas y cada interacción indica su fuente"


def prueba_usa_los_medicamentos_de_la_rama():
    """El integrador usa lo que calculó la rama de medicación del grafo."""
    r = reglas.cruzar("manuel", ["receta5.jpeg"], HOY, medicamentos=["Sertralina"])
    assert reglas.pares(r) == {("Sertralina", "alcohol")}
    return "el cruce usa los medicamentos que le pasa la rama de medicación"


# --- Redacción ------------------------------------------------------------------

class LLMQueNoDebeLlamarse:
    def invoke(self, prompt):
        raise AssertionError("no debía llamarse al LLM")


def prueba_no_evaluable_no_llama_al_llm():
    for usuario, fuentes, motivo in [("luis", ["receta5.jpeg"], reglas.MOTIVO_SIN_PRESCRIPCION),
                                      ("rosa", [], reglas.MOTIVO_SIN_RECETA)]:
        salida = redactar("¿puedo comer esto?", reglas.cruzar(usuario, fuentes, HOY), LLMQueNoDebeLlamarse())
        assert not salida["uso_llm"]
        assert salida["respuesta"].startswith(TEXTO_NO_EVALUABLE[motivo])
    return "los casos no evaluables responden con texto fijo, sin LLM"


def prueba_el_prompt_lleva_solo_el_cruce():
    cruce = reglas.cruzar("elena", ["tigrillo.png"], HOY)
    prompt = prompt_integrador("¿puedo comer tigrillo?", cruce)
    assert "Melatonina con cafeina" in prompt and "acompañamiento" in prompt
    assert "NO agregues interacciones" in prompt
    assert "Sertralina con" not in prompt, "Sertralina no interactúa con este plato"
    return "el prompt del integrador lleva el resultado del cruce y prohíbe agregar otros"


# --- El camino por el grafo, con dobles -----------------------------------------

class Respuesta:
    def __init__(self, content):
        self.content = content


class LLMGuionado:
    """Responde según qué pide el prompt. Registra qué prompts recibió."""

    def __init__(self, intencion):
        self.intencion = intencion
        self.recibidos = []

    def invoke(self, prompt):
        if "CATEGORIA:" in prompt:
            self.recibidos.append("orchestrator")
            return Respuesta(f"Razono.\nCATEGORIA: {self.intencion}")
        if "DECISION: BUSCAR" in prompt:
            self.recibidos.append("decidir")
            return Respuesta("Pide ingredientes.\nDECISION: BUSCAR")
        if "VEREDICTO: SI" in prompt:
            self.recibidos.append("evaluar")
            return Respuesta("Es la receta.\nVEREDICTO: SI")
        if "Resultado del cruce" in prompt:
            self.recibidos.append("integrador")
            return Respuesta("Mejor evítelo.")
        self.recibidos.append("otro")
        return Respuesta("Respuesta genérica.")


def _grafo_con_dobles(intencion, fuente="receta5.jpeg"):
    from orquestacion_langgraph import agentes, grafo
    from orquestacion_langgraph.rag_agentico import nodos

    llm = LLMGuionado(intencion)
    agentes.llm = llm
    nodos.llm = llm
    fragmento = {"texto": "Ingredientes ... ron blanco", "fuente": fuente, "distancia": 0.2, "orden": 0}
    nodos.buscar_receta_detallado = lambda consulta, k=3: [fragmento]
    nodos.fragmentos_de_fuente = lambda f: [fragmento]
    return grafo, llm


def prueba_camino_del_grafo_en_paralelo():
    grafo, llm = _grafo_con_dobles("MEDICATION_FOOD_CHECK")
    app = grafo.construir_grafo()
    pasos = []
    for actualizacion in app.stream(grafo._estado_inicial("¿puedo tomar el come y bebe de badea?", id_perfil="rosa"),
                                    stream_mode="updates"):
        pasos.append(set(actualizacion))
    nodos_recorridos = [n for paso in pasos for n in paso]
    assert nodos_recorridos[0] == "orchestrator", nodos_recorridos
    assert set(nodos_recorridos[1:3]) == {"medicacion_cruce", "recetas_cruce"}, nodos_recorridos
    assert nodos_recorridos[-1] == "integrador", nodos_recorridos
    assert len(nodos_recorridos) == 4, f"no debe pasar por otros nodos: {nodos_recorridos}"
    assert "generar" not in llm.recibidos, "la rama de recetas no debe redactar la receta"

    final = app.invoke(grafo._estado_inicial("¿puedo tomar el come y bebe de badea?", id_perfil="rosa"))
    assert reglas.pares(final["cruce"]) == {("Metformina", "alcohol"), ("Aspirina", "alcohol")}
    assert final["respuesta"].startswith("Mejor evítelo.")
    return "orchestrator → {medicación, receta} en paralelo → integrador, sin redactar la receta"


def prueba_las_otras_intenciones_no_pasan_por_el_cruce():
    grafo, _ = _grafo_con_dobles("FAMILY_COMMUNICATION")
    final = grafo.construir_grafo().invoke(grafo._estado_inicial("avísale a mi hija", id_perfil="rosa"))
    assert final["intencion"] == "FAMILY_COMMUNICATION" and final["cruce"] is None
    return "las demás intenciones siguen su camino de siempre"


CASOS = [
    prueba_casos_de_desarrollo,
    prueba_casos_de_campana,
    prueba_potasio_solo_con_funcion_renal_reducida,
    prueba_veredicto_es_el_mas_grave,
    prueba_no_evaluable_sin_receta_reconocida,
    prueba_prescripcion_vencida_no_cuenta,
    prueba_varias_fuentes_se_cruzan_todas,
    prueba_usa_los_medicamentos_de_la_rama,
    prueba_no_evaluable_no_llama_al_llm,
    prueba_el_prompt_lleva_solo_el_cruce,
    prueba_camino_del_grafo_en_paralelo,
    prueba_las_otras_intenciones_no_pasan_por_el_cruce,
]


def main():
    print("Pruebas de la tarea medicamento × comida (sin VPN, sin LLM)\n")
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

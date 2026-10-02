# Pruebas del arnés de medición (medicion/).
#
# Uso (desde la raíz del repo):
#   python -m pruebas.prueba_medicion
#
# NO requieren VPN: el registro de tokens se prueba con un modelo falso sobre un
# grafo chico que reproduce la forma del real (un nodo con subgrafo adentro y
# dos ramas en paralelo). Escrito sin pytest, igual que el resto de pruebas/.

import csv
import sys
from pathlib import Path
from typing import TypedDict

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from interacciones.datos import cargar_ingredientes
from medicacion.datos import prescripciones_de
from medicion.casos import TAREAS, cargar_campana, frases_de, plan_de_ejecuciones
from medicion.criterios import evaluar, nombra
from medicion.estadistica import kruskal_wallis, media_de, wilson
from medicion.registro import RegistroEjecucion, ubicacion

RAIZ = Path(__file__).parent.parent
CATALOGO = ["Metformina", "Losartan", "Aspirina", "Ibuprofeno", "Acido folico"]


def _respuesta(entrada, salida, razonamiento):
    return AIMessage(
        content="ok",
        usage_metadata={"input_tokens": entrada, "output_tokens": salida, "total_tokens": entrada + salida,
                        "output_token_details": {"reasoning": razonamiento}},
        response_metadata={"model_name": "modelo-falso", "system_fingerprint": "fp-1", "finish_reason": "stop"},
    )


def prueba_ubicacion_desde_metadata():
    assert ubicacion({"langgraph_node": "orchestrator",
                      "langgraph_checkpoint_ns": "orchestrator:abc"}) == ("orchestrator", "orchestrator")
    assert ubicacion({"langgraph_node": "evaluar_relevancia",
                      "langgraph_checkpoint_ns": "recetas_cruce:a|evaluar_relevancia:b"}) == ("recetas_cruce", "evaluar_relevancia")
    assert ubicacion(None) == (None, None)
    return "el agente es el primer tramo del checkpoint_ns; el nodo, el que llamó al LLM"


def prueba_registro_por_agente_en_un_grafo():
    llm = GenericFakeChatModel(messages=iter([_respuesta(100, 50, 30), _respuesta(10, 5, 0), _respuesta(7, 3, 1)]))

    class E(TypedDict, total=False):
        x: str
        a: str
        b: str

    sub = StateGraph(E)
    sub.add_node("paso_interno", lambda e: {"x": llm.invoke("hola").content})
    sub.add_edge(START, "paso_interno")
    sub.add_edge("paso_interno", END)
    subgrafo = sub.compile()

    g = StateGraph(E)
    g.add_node("jefe", lambda e: {"x": llm.invoke("hola").content})
    g.add_node("rama_rag", lambda e: {"a": subgrafo.invoke({"x": ""})["x"]})
    g.add_node("rama_sin_llm", lambda e: {"b": "fijo"})
    g.add_edge(START, "jefe")
    g.add_conditional_edges("jefe", lambda e: ["rama_rag", "rama_sin_llm"])
    g.add_edge(["rama_rag", "rama_sin_llm"], END)

    registro = RegistroEjecucion()
    g.compile().invoke({"x": ""}, config={"callbacks": [registro]})
    r = registro.resumen()

    # Sin __start__: es un nodo interno de LangGraph, no un agente.
    assert set(r["agentes"]) == {"jefe", "rama_rag", "rama_sin_llm"}, r["agentes"]
    assert r["agentes"]["jefe"]["tokens_entrada"] == 100 and r["agentes"]["jefe"]["tokens_razonamiento"] == 30
    assert r["agentes"]["rama_rag"]["llamadas_llm"] == 1 and r["agentes"]["rama_rag"]["tokens_salida"] == 5
    assert r["agentes"]["rama_sin_llm"]["llamadas_llm"] == 0
    assert [s["nodo"] for s in r["subnodos"]] == ["paso_interno"] and r["subnodos"][0]["agente"] == "rama_rag"
    assert r["tokens_sistema"] == {"llamadas_llm": 2, "entrada": 110, "salida": 55, "razonamiento": 30}
    assert r["fingerprints"] == ["fp-1"] and r["modelos"] == ["modelo-falso"]
    return "tokens y tiempos se atribuyen al agente correcto, también dentro de un subgrafo y en paralelo"


def _ejecucion(**cambios):
    base = {"intencion": None, "camino": [], "respuesta": "", "traza_rag": None, "cruce": None,
            "error": None, "timeout": False}
    base.update(cambios)
    return base


def prueba_criterio_t1():
    definicion = {"intencion": "MEDICATION_HEALTH",
                  "medicamentos_por_usuario": {"rosa": ["Metformina", "Losartan", "Aspirina"]}}
    frase = {"id": "f"}
    bien = _ejecucion(intencion="MEDICATION_HEALTH", camino=["orchestrator", "medicacion"],
                      respuesta="A las 8: metformina y losartán. A las 13: ASPIRINA.")
    assert evaluar("T1", definicion, frase, "rosa", bien, CATALOGO)["exito"]

    falta = dict(bien, respuesta="A las 8: metformina y losartán.")
    r = evaluar("T1", definicion, frase, "rosa", falta, CATALOGO)
    assert not r["exito"] and r["detalle"]["faltan"] == ["Aspirina"]

    sobra = dict(bien, respuesta=bien["respuesta"] + " Si duele, ibuprofeno.")
    r = evaluar("T1", definicion, frase, "rosa", sobra, CATALOGO)
    assert not r["exito"] and r["detalle"]["ajenos"] == ["Ibuprofeno"]
    return "T1: exige todos los de la receta (sin importar tildes) y ninguno ajeno"


def prueba_criterio_t2():
    definicion = {"intencion": "RECIPE_MULTIMEDIA"}
    con_fuente = _ejecucion(intencion="RECIPE_MULTIMEDIA", camino=["orchestrator", "recetas"],
                            traza_rag=[{"nodo": "recuperar"}, {"nodo": "generar", "fuentes": ["llapingacho.jpg"]}])
    assert evaluar("T2", definicion, {"fuentes_aceptadas": ["llapingacho.jpg", "receta1.jpeg"]},
                   "rosa", con_fuente)["exito"]
    assert not evaluar("T2", definicion, {"fuentes_aceptadas": ["bolon de verde.png"]}, "rosa", con_fuente)["exito"]

    sin_resultado = dict(con_fuente, traza_rag=[{"nodo": "recuperar"}, {"nodo": "sin_resultado"}])
    assert evaluar("T2", definicion, {"fuentes_aceptadas": []}, "rosa", sin_resultado)["exito"]
    assert not evaluar("T2", definicion, {"fuentes_aceptadas": []}, "rosa", con_fuente)["exito"]
    return "T2: fuente aceptada, o ninguna fuente si el plato no está en el recetario"


def prueba_timeout_y_error_son_fallo():
    definicion = {"intencion": "EMERGENCY"}
    bien = _ejecucion(intencion="EMERGENCY", camino=["orchestrator", "emergencia"])
    assert evaluar("T3", definicion, {}, "rosa", bien)["exito"]
    assert not evaluar("T3", definicion, {}, "rosa", dict(bien, timeout=True))["exito"]
    assert not evaluar("T3", definicion, {}, "rosa", dict(bien, error="ConnectError"))["exito"]
    assert not evaluar("T3", definicion, {}, "rosa", dict(bien, intencion="SMALL_TALK",
                                                          camino=["orchestrator", "small_talk"]))["exito"]
    return "un timeout o una excepción es fallo aunque la intención haya salido bien"


def prueba_criterio_t6():
    frase = frases_de("T6")[0]
    usuario = next(iter(frase["esperado_por_usuario"]))
    esperado = frase["esperado_por_usuario"][usuario]
    cruce = {"evaluable": True, "fuentes_cruzadas": [frase["fuentes_aceptadas"][0]],
             "interacciones": [{"medicamento": m, "categoria": c} for m, c in esperado["interacciones"]]}
    ejecucion = _ejecucion(intencion="MEDICATION_FOOD_CHECK", cruce=cruce,
                           camino=["orchestrator", "recetas_cruce", "medicacion_cruce", "integrador"])
    assert evaluar("T6", {"intencion": "MEDICATION_FOOD_CHECK"}, frase, usuario, ejecucion)["exito"]
    mal_camino = dict(ejecucion, camino=["orchestrator", "medicacion"])
    assert not evaluar("T6", {"intencion": "MEDICATION_FOOD_CHECK"}, frase, usuario, mal_camino)["exito"]
    return "T6: el orden de las ramas paralelas no importa; saltarse una sí"


def prueba_casos_coherentes_con_los_datos():
    campana = cargar_campana()
    for tarea in TAREAS:
        assert len(frases_de(tarea, campana)) == 4, tarea

    # T1: lo esperado coincide con TODAS las recetas de cada persona (vigentes o no).
    esperado = campana["tareas"]["T1"]["medicamentos_por_usuario"]
    for usuario in campana["usuarios"]:
        en_receta = {i["medicamento"] for r in prescripciones_de(usuario) for i in r["indicaciones"]}
        assert set(esperado[usuario]) == en_receta, (usuario, esperado[usuario], en_receta)

    # T1, T3-T5: las frases están en dataset.csv, en la línea y con la etiqueta declaradas.
    with open(RAIZ / "dataset.csv", encoding="utf-8-sig") as f:
        lineas = list(csv.reader(f))
    for tarea in ["T1", "T3", "T4", "T5"]:
        for frase in campana["tareas"][tarea]["frases"]:
            etiqueta, texto = lineas[frase["linea_dataset"] - 1]
            assert texto == frase["consulta"], (frase["id"], texto)
            assert etiqueta == campana["tareas"][tarea]["intencion"], (frase["id"], etiqueta)

    # T2: las fuentes aceptadas existen en el índice del recetario.
    indexadas = set(cargar_ingredientes())
    for frase in campana["tareas"]["T2"]["frases"]:
        assert set(frase["fuentes_aceptadas"]) <= indexadas, frase["id"]
    return "las frases y la verdad de referencia cuadran con dataset.csv y prescripciones.json"


def prueba_plan_intercalado_por_repeticion():
    plan = plan_de_ejecuciones(TAREAS, ["rosa", "manuel", "carmen", "jorge", "elena"], 3)
    assert len(plan) == 360 and len({e["id"] for e in plan}) == 360
    assert [e["repeticion"] for e in plan] == sorted(e["repeticion"] for e in plan)
    primeras = [e["tarea"] for e in plan[:12]]
    assert len(set(primeras)) > 2, primeras
    assert plan == plan_de_ejecuciones(TAREAS, ["rosa", "manuel", "carmen", "jorge", "elena"], 3)
    piloto = plan_de_ejecuciones(TAREAS, ["rosa"], 1, frases_por_tarea=1)
    assert sorted(e["tarea"] for e in piloto) == TAREAS
    return "360 ejecuciones únicas, intercaladas, reproducibles y por bloques de repetición"


def prueba_nombra_palabra_completa():
    assert nombra("tome su Losartán", "Losartan")
    assert not nombra("hierros", "Hierro")
    assert nombra("ácido fólico diario", "Acido folico")
    return "la búsqueda de nombres ignora tildes y exige palabra completa"


def prueba_estadistica_contra_valores_conocidos():
    m, de = media_de([2, 4, 4, 4, 5, 5, 7, 9])
    assert m == 5 and abs(de - 2.138) < 1e-3  # DE muestral (n-1)
    bajo, alto = wilson(20, 20)
    assert abs(bajo - 0.8389) < 1e-3 and alto == 1.0  # 20/20 no da [1, 1]
    bajo, alto = wilson(0, 10)
    assert bajo == 0.0 and abs(alto - 0.2775) < 1e-3
    # Tres grupos sin solapamiento: H = 7.2, p = exp(-3.6) = 0.0273 (gl = 2).
    r = kruskal_wallis([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert abs(r["H"] - 7.2) < 1e-9 and r["gl"] == 2 and abs(r["p"] - 0.02732) < 1e-4
    # Grupos idénticos: H = 0, p = 1.
    r = kruskal_wallis([[1, 2], [1, 2], [1, 2], [1, 2], [1, 2]])
    assert abs(r["H"]) < 1e-9 and abs(r["p"] - 1) < 1e-9
    return "media ± DE muestral, Wilson y Kruskal-Wallis dan los valores de referencia"


CASOS = [
    prueba_ubicacion_desde_metadata,
    prueba_registro_por_agente_en_un_grafo,
    prueba_criterio_t1,
    prueba_criterio_t2,
    prueba_timeout_y_error_son_fallo,
    prueba_criterio_t6,
    prueba_casos_coherentes_con_los_datos,
    prueba_plan_intercalado_por_repeticion,
    prueba_nombra_palabra_completa,
    prueba_estadistica_contra_valores_conocidos,
]


def main():
    print("Pruebas del arnés de medición (sin VPN)\n")
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

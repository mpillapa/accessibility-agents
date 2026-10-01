# Evaluación de la tarea medicamento × comida con los modelos reales, por el
# grafo completo.
#
# Uso (desde la raíz del repo):
#   python -m pruebas.evaluar_interacciones_real
#   python -m pruebas.evaluar_interacciones_real --usuarios rosa elena --json salida.json
#
# REQUIERE el LLM, los embeddings y el índice del recetario.
#
# Corre las frases de la campaña (interacciones/datos/casos_prueba.json,
# sección "campana") para cada usuario y marca cada ejecución contra la verdad
# de referencia. Es ÉXITO si se cumple todo:
#   1. el Orchestrator eligió MEDICATION_FOOD_CHECK;
#   2. el camino fue orchestrator → {medicacion_cruce, recetas_cruce} → integrador;
#   3. el RAG encontró una de las fuentes aceptadas;
#   4. las interacciones son exactamente las esperadas para ese usuario.
# Si algo falla, se registra QUÉ falló, para leerlo antes de contarlo.
#
# Corre en serie: la latencia que reporta es la de una consulta sola, aunque
# sigue expuesta a la carga del servidor compartido (bitácora 11).

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from infraestructura.modelos import describir_resolucion
from interacciones.datos import cargar_casos
from interacciones.reglas import pares
from orquestacion_langgraph.grafo import procesar_consulta_en_vivo

CAMINO_ESPERADO = ["orchestrator", {"medicacion_cruce", "recetas_cruce"}, "integrador"]


def _camino_correcto(nodos: list[str]) -> bool:
    return (len(nodos) == 4 and nodos[0] == "orchestrator"
            and set(nodos[1:3]) == {"medicacion_cruce", "recetas_cruce"} and nodos[3] == "integrador")


def ejecutar(frase: dict, usuario: str) -> dict:
    esperado = frase["esperado_por_usuario"][usuario]
    nodos, final = [], None
    inicio = time.time()
    try:
        for paso in procesar_consulta_en_vivo(frase["consulta"], id_perfil=usuario):
            if paso["nodo"]:
                nodos.append(paso["nodo"])
            else:
                final = paso["estado_final"]
        error = None
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    segundos = round(time.time() - inicio, 2)

    cruce = (final or {}).get("cruce") or {}
    fuentes = cruce.get("fuentes_cruzadas", [])
    obtenidos = pares(cruce) if cruce else set()
    esperados = {tuple(p) for p in esperado["interacciones"]}
    chequeos = {
        "intencion": (final or {}).get("intencion") == "MEDICATION_FOOD_CHECK",
        "camino": _camino_correcto(nodos),
        "fuente": any(f in frase["fuentes_aceptadas"] for f in fuentes),
        "interacciones": cruce.get("evaluable", False) and obtenidos == esperados,
    }
    return {
        "frase": frase["id"], "usuario": usuario, "consulta": frase["consulta"],
        "exito": error is None and all(chequeos.values()),
        "chequeos": chequeos, "error": error, "segundos": segundos,
        "intencion": (final or {}).get("intencion"), "camino": nodos,
        "fuentes": fuentes, "fuentes_aceptadas": frase["fuentes_aceptadas"],
        "veredicto": cruce.get("veredicto"), "motivo_no_evaluable": cruce.get("motivo"),
        "interacciones_obtenidas": sorted(obtenidos), "interacciones_esperadas": sorted(esperados),
        "respuesta": (final or {}).get("respuesta"),
    }


def main():
    parser = argparse.ArgumentParser(description="Tarea medicamento × comida con modelos reales")
    parser.add_argument("--usuarios", nargs="+", help="por defecto, los de la campaña")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    campana = cargar_casos()["campana"]
    usuarios = args.usuarios or campana["usuarios"]
    resultados = []
    for frase in campana["frases"]:
        for usuario in usuarios:
            r = ejecutar(frase, usuario)
            resultados.append(r)
            fallidos = [k for k, v in r["chequeos"].items() if not v]
            estado = "OK   " if r["exito"] else f"FALLA {fallidos or r['error']}"
            print(f"  {estado} {r['frase']}/{usuario:7} {r['segundos']:5.1f}s  fuentes={r['fuentes']} "
                  f"veredicto={r['veredicto']}", flush=True)

    exitos = sum(r["exito"] for r in resultados)
    tiempos = sorted(r["segundos"] for r in resultados)
    print(f"\nÉxito: {exitos}/{len(resultados)} · latencia mediana {tiempos[len(tiempos) // 2]:.1f}s "
          f"(mín {tiempos[0]:.1f}, máx {tiempos[-1]:.1f})")
    for clave in ("intencion", "camino", "fuente", "interacciones"):
        print(f"  {clave:14} {sum(r['chequeos'][clave] for r in resultados)}/{len(resultados)}")

    if args.json:
        args.json.write_text(json.dumps({
            "fecha": datetime.now().isoformat(timespec="seconds"),
            "resolucion_de_modelos": describir_resolucion(),
            "resultados": resultados,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Guardado en {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

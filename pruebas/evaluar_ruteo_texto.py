# Ruteo del Orchestrator sobre las frases etiquetadas de dataset.csv (bitácora 20.3).
# Reporta accuracy, recall por intención, matriz de confusión y cómo se extrajo
# la intención (`por_defecto` = el modelo no cerró con CATEGORIA:).
#
# Uso (desde la raíz del repo):
#   python -m pruebas.evaluar_ruteo_texto --etiqueta base --json resultados/X.json
#   python -m pruebas.evaluar_ruteo_texto --por-intencion 10     # muestra rápida
#
# Requiere el LLM del servidor. Corre en paralelo: la latencia es orientativa;
# la del paper sale de la campaña, en serie.

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from infraestructura.modelos import describir_resolucion
from orquestacion_langgraph.agentes import INTENCIONES, clasificar_detallado, decidir_intencion

DATASET = Path(__file__).parent.parent / "dataset.csv"

# --solo-llm: sin la red de seguridad de emergencias. Por defecto, la decisión completa del grafo.
SOLO_LLM = False


def cargar_frases(por_intencion: int | None) -> list[dict]:
    with DATASET.open(encoding="utf-8-sig") as f:
        filas = list(csv.DictReader(f))
    if por_intencion:
        vistos: Counter = Counter()
        elegidas = []
        for fila in filas:
            if vistos[fila["intent"]] < por_intencion:
                elegidas.append(fila)
                vistos[fila["intent"]] += 1
        filas = elegidas
    return filas


def _clasificar(fila: dict) -> dict:
    inicio = time.time()
    try:
        d = (clasificar_detallado if SOLO_LLM else decidir_intencion)(fila["text"])
        error = None
    except Exception as e:
        d, error = {}, f"{type(e).__name__}: {e}"
    return {
        "texto": fila["text"],
        "esperada": fila["intent"],
        "obtenida": d.get("intencion"),
        "como_se_decidio": d.get("como_se_decidio"),
        "crudo": d.get("crudo"),
        "tokens": d.get("tokens"),
        "segundos": round(time.time() - inicio, 2),
        "error": error,
    }


def resumir(resultados: list[dict]) -> dict:
    validos = [r for r in resultados if not r["error"]]
    aciertos = sum(r["esperada"] == r["obtenida"] for r in validos)
    por_clase = defaultdict(lambda: [0, 0])
    confusion: dict[str, Counter] = defaultdict(Counter)
    for r in validos:
        por_clase[r["esperada"]][1] += 1
        por_clase[r["esperada"]][0] += r["esperada"] == r["obtenida"]
        confusion[r["esperada"]][r["obtenida"]] += 1
    como = Counter(r["como_se_decidio"] for r in validos)
    destinos_error = Counter(r["obtenida"] for r in validos if r["esperada"] != r["obtenida"])
    return {
        "n": len(resultados),
        "errores_de_llamada": len(resultados) - len(validos),
        "accuracy": round(aciertos / len(validos), 4) if validos else None,
        "recall_por_intencion": {k: round(v[0] / v[1], 4) for k, v in sorted(por_clase.items())},
        "aciertos_por_intencion": {k: f"{v[0]}/{v[1]}" for k, v in sorted(por_clase.items())},
        "como_se_decidio": dict(como),
        "destino_de_los_errores": dict(destinos_error),
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "emergencias_perdidas": [
            {"texto": r["texto"], "obtenida": r["obtenida"], "como_se_decidio": r["como_se_decidio"]}
            for r in validos if r["esperada"] == "EMERGENCY" and r["obtenida"] != "EMERGENCY"
        ],
    }


def imprimir(resumen: dict):
    print(f"\nn={resumen['n']} · errores de llamada: {resumen['errores_de_llamada']}")
    print(f"accuracy: {resumen['accuracy']:.1%}")
    for intencion, aciertos in resumen["aciertos_por_intencion"].items():
        print(f"  {intencion:22} {aciertos:>7}  ({resumen['recall_por_intencion'][intencion]:.1%})")
    print(f"cómo se decidió: {resumen['como_se_decidio']}")
    print(f"destino de los errores: {resumen['destino_de_los_errores']}")
    for e in resumen["emergencias_perdidas"]:
        print(f"  EMERGENCIA PERDIDA → {e['obtenida']} ({e['como_se_decidio']}): {e['texto']}")


def main():
    parser = argparse.ArgumentParser(description="Ruteo del Orchestrator sobre dataset.csv")
    parser.add_argument("--por-intencion", type=int, help="solo las primeras N frases de cada intención")
    parser.add_argument("--hilos", type=int, default=4)
    parser.add_argument("--etiqueta", default="sin_etiqueta", help="qué versión del prompt se mide")
    parser.add_argument("--json", type=Path, help="guarda resultados crudos y resumen")
    parser.add_argument("--solo-llm", action="store_true", help="sin la red de seguridad de emergencias")
    args = parser.parse_args()
    global SOLO_LLM
    SOLO_LLM = args.solo_llm

    frases = cargar_frases(args.por_intencion)
    print(f"Ruteo de {len(frases)} frases, {args.hilos} hilos, intenciones: {', '.join(INTENCIONES)}")
    inicio = datetime.now()
    with ThreadPoolExecutor(max_workers=args.hilos) as ejecutor:
        resultados = list(ejecutor.map(_clasificar, frases))
    resumen = resumir(resultados)
    imprimir(resumen)

    if args.json:
        args.json.write_text(json.dumps({
            "etiqueta": args.etiqueta,
            "inicio": inicio.isoformat(timespec="seconds"),
            "fin": datetime.now().isoformat(timespec="seconds"),
            "intenciones": INTENCIONES,
            "resolucion_de_modelos": describir_resolucion(),
            "hilos": args.hilos,
            "solo_llm": args.solo_llm,
            "resumen": resumen,
            "resultados": resultados,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nGuardado en {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

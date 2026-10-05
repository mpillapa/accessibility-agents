# Evaluación ampliada del RAG agéntico, aislado del Orchestrator.
#
# Uso (desde la raíz del repo, con VPN):
#   python -m medicion.evaluar_rag --experimento rag_2026-10-04
#   python -m medicion.evaluar_rag --experimento rag_2026-10-04 --reanudar
#   python -m medicion.evaluar_rag --experimento rag_2026-10-04 --solo-puntuar
#
# Guarda las fuentes crudas de cada ejecución y puntúa aparte: si se corrige la
# verdad de referencia (datos/casos_rag.json), --solo-puntuar recalcula sin
# volver a llamar al modelo.

import argparse
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as TiempoAgotado
from datetime import datetime
from pathlib import Path

from langchain_core.runnables import RunnableLambda

from infraestructura.modelos import describir_resolucion
from medicion.campana import UMBRAL_EMBEDDING_SEGUNDOS, sondear_embeddings
from medicion.criterios import fuentes_de_la_respuesta
from medicion.estadistica import media_de, wilson
from medicion.registro import RegistroEjecucion
from orquestacion_langgraph.rag_agentico.subgrafo import consultar_recetario

RUTA_CASOS = Path(__file__).parent / "datos" / "casos_rag.json"
CARPETA = Path(__file__).parent.parent / "resultados" / "rag"
TIMEOUT_SEGUNDOS = 120
TIPOS = ["exacto", "descripcion", "coloquial", "ambigua", "fuera"]


def cargar_casos() -> list[dict]:
    return json.loads(RUTA_CASOS.read_text(encoding="utf-8"))["casos"]


def ejecutar(caso: dict, repeticion: int, experimento: str) -> dict:
    registro = RegistroEjecucion()
    uid = uuid.uuid4().hex
    config = {"callbacks": [registro], "tags": [experimento], "run_name": f"{caso['id']}-r{repeticion}",
              "metadata": {"experimento": experimento, "id_ejecucion": f"{caso['id']}-r{repeticion}", "uid": uid}}
    ejecutor = ThreadPoolExecutor(max_workers=1)
    inicio = time.perf_counter()
    futuro = ejecutor.submit(RunnableLambda(consultar_recetario).invoke, caso["consulta"], config)
    resultado, error, timeout = {}, None, False
    try:
        resultado = futuro.result(timeout=TIMEOUT_SEGUNDOS)
    except TiempoAgotado:
        timeout = True
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    segundos = round(time.perf_counter() - inicio, 3)
    ejecutor.shutdown(wait=False)
    resumen = registro.resumen()
    traza = resultado.get("traza") or []
    return {
        "id": f"{caso['id']}-r{repeticion}", "uid": uid, "caso": caso["id"], "tipo": caso["tipo"],
        "consulta": caso["consulta"], "repeticion": repeticion,
        "inicio": datetime.now().isoformat(timespec="seconds"), "segundos": segundos,
        "timeout": timeout, "error": error,
        "hubo_resultado": resultado.get("hubo_resultado"), "intentos": resultado.get("intentos"),
        "fuentes_aprobadas": resultado.get("fuentes", []),
        "fuentes_respuesta": fuentes_de_la_respuesta(traza),
        "camino": [p.get("nodo") for p in traza],
        "respuesta": resultado.get("respuesta"),
        "tokens": resumen["tokens_sistema"], "fingerprints": resumen["fingerprints"],
        "segundos_por_nodo": [(n["nodo"], n["segundos"]) for n in registro.nodos],
        "llamadas_sin_respuesta": resumen.get("llamadas_sin_respuesta", []),
    }


def puntuar(fila: dict, caso: dict) -> dict:
    """`acierto`: usó una fuente aceptada (o ninguna, si el plato no está).
    `limpio`: además, no mezcló ninguna fuente que no corresponde."""
    aceptadas, usadas = set(caso["fuentes_aceptadas"]), set(fila["fuentes_respuesta"])
    cortada = fila["timeout"] or bool(fila["error"])
    acierto = not cortada and (bool(usadas & aceptadas) if aceptadas else not usadas)
    return {"acierto": acierto, "limpio": acierto and usadas <= aceptadas, "cortada": cortada}


def _tabla(filas: list[dict], casos: dict) -> list[str]:
    lineas = ["| Tipo | Consultas | Ejecuciones | Acierto | IC 95% | Sin mezclar fuentes | Reformuló | Tiempo (s) | Tokens salida |",
              "|---|---|---|---|---|---|---|---|---|"]
    for tipo in TIPOS + ["Todas"]:
        grupo = [f for f in filas if tipo == "Todas" or f["tipo"] == tipo]
        if not grupo:
            continue
        p = [puntuar(f, casos[f["caso"]]) for f in grupo]
        k, n = sum(x["acierto"] for x in p), len(p)
        bajo, alto = wilson(k, n)
        completas = [f for f, x in zip(grupo, p) if not x["cortada"]]
        m_t, de_t = media_de([f["segundos"] for f in completas])
        m_s, de_s = media_de([f["tokens"]["salida"] for f in completas])
        reform = sum("reformular" in f["camino"] for f in completas)
        lineas.append(
            f"| {tipo} | {len({f['caso'] for f in grupo})} | {n} | {k}/{n} ({100 * k / n:.0f}%) | "
            f"{100 * bajo:.0f}–{100 * alto:.0f}% | {sum(x['limpio'] for x in p)}/{n} | {reform}/{len(completas)} | "
            f"{m_t:.1f} ± {de_t or 0:.1f} | {m_s:.0f} ± {de_s or 0:.0f} |")
    return lineas


def _fallos(filas: list[dict], casos: dict) -> list[str]:
    lineas = ["| Consulta | Tipo | Aceptadas | Usó | Camino |", "|---|---|---|---|---|"]
    for f in filas:
        if not puntuar(f, casos[f["caso"]])["acierto"]:
            motivo = "timeout" if f["timeout"] else (f["error"] or "")[:40]
            lineas.append(f"| {f['id']} {f['consulta']!r} | {f['tipo']} | {casos[f['caso']]['fuentes_aceptadas'] or '—'} | "
                          f"{f['fuentes_respuesta'] or '—'} {motivo} | {' → '.join(f['camino'])} |")
    return lineas if len(lineas) > 2 else ["Ninguno."]


def informe(ruta: Path) -> Path:
    filas = [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]
    casos = {c["id"]: c for c in cargar_casos()}
    texto = [f"# Evaluación ampliada del RAG `{ruta.stem}`", "",
             f"- {len({f['caso'] for f in filas})} consultas, {len(filas)} ejecuciones; "
             f"fingerprints {sorted({x for f in filas for x in f['fingerprints']})}.",
             "- Acierto: la respuesta usó una fuente aceptada, o ninguna si el plato no está en el recetario.",
             "- Sin mezclar: además no usó ninguna fuente ajena. Timeout (120 s) o excepción = fallo.", "",
             "## Por tipo de consulta", "", *_tabla(filas, casos), "",
             "## Fallos (leer antes de contar)", "", *_fallos(filas, casos), ""]
    salida = ruta.with_name(ruta.stem + "_tablas.md")
    salida.write_text("\n".join(texto) + "\n", encoding="utf-8")
    return salida


def main():
    parser = argparse.ArgumentParser(description="Evaluación ampliada del RAG agéntico")
    parser.add_argument("--experimento", required=True)
    parser.add_argument("--repeticiones", type=int, default=3)
    parser.add_argument("--reanudar", action="store_true")
    parser.add_argument("--solo-puntuar", action="store_true", help="recalcular tablas sin llamar al modelo")
    args = parser.parse_args()

    CARPETA.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA / f"{args.experimento}.jsonl"
    if args.solo_puntuar:
        print(f"Tablas en {informe(ruta)}")
        return 0
    if ruta.exists() and not args.reanudar:
        print(f"Ya existe {ruta}. Usá --reanudar o --solo-puntuar.")
        return 1

    sonda = sondear_embeddings()
    print(f"Sonda de embeddings: {sonda} s")
    if min(sonda) > UMBRAL_EMBEDDING_SEGUNDOS:
        print("  Embeddings lentos: no se empieza (ver medicion/README.md, sonda previa).")
        return 2

    hechas = set()
    if ruta.exists():
        hechas = {json.loads(l)["id"] for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()}
    plan = [(c, r) for r in range(1, args.repeticiones + 1) for c in cargar_casos()
            if f"{c['id']}-r{r}" not in hechas]
    (CARPETA / f"{args.experimento}_meta.json").write_text(json.dumps({
        "inicio": datetime.now().isoformat(timespec="seconds"), "sonda_embeddings_segundos": sonda,
        "resolucion_de_modelos": describir_resolucion(), "repeticiones": args.repeticiones,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    with ruta.open("a", encoding="utf-8") as salida:
        for n, (caso, rep) in enumerate(plan, 1):
            f = ejecutar(caso, rep, args.experimento)
            salida.write(json.dumps(f, ensure_ascii=False) + "\n")
            salida.flush()
            ok = "OK   " if puntuar(f, caso)["acierto"] else "FALLA"
            print(f"  [{n:3}/{len(plan)}] {ok} {f['id']:7} {f['segundos']:6.1f}s usó={f['fuentes_respuesta']} "
                  f"esperadas={caso['fuentes_aceptadas'] or '—'}", flush=True)
    print(f"Tablas en {informe(ruta)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

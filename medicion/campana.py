# Campaña de medición: tokens, tiempo y éxito por tarea, usuario y agente.
#
# Uso (desde la raíz del repo, con VPN):
#   python -m medicion.campana --experimento piloto_2026-10-01 --piloto
#   python -m medicion.campana --experimento campana_2026-10-03
#   python -m medicion.campana --experimento campana_2026-10-03 --reanudar
#
# REQUIERE el LLM, los embeddings y el índice del recetario. Diseño en
# medicion/README.md y en la chuleta (sección 7).
#
# Cada ejecución:
#   - entra por TEXTO (la voz se midió aparte: corpus_audio/, bitácora 14-17);
#   - pasa por el mismo procesar_consulta_en_vivo que usa la interfaz;
#   - va etiquetada para LangSmith (tag = experimento, metadata = tarea,
#     usuario, frase, repetición, id) y además se registra en local con el
#     callback de medicion/registro.py;
#   - se corta a los TIMEOUT_SEGUNDOS y cuenta como fallo.
#
# El resultado es un JSONL en resultados/campana/: una línea por ejecución, con
# la respuesta completa. Se escribe tras cada ejecución, así que un corte de VPN
# no pierde lo hecho y --reanudar sigue donde quedó.

import argparse
import json
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as TiempoAgotado
from datetime import datetime
from pathlib import Path

from infraestructura.modelos import describir_resolucion
from infraestructura.trazas import describir_trazas
from medicion.casos import TAREAS, cargar_campana, plan_de_ejecuciones
from medicion.criterios import catalogo_de_medicamentos, evaluar
from medicion.registro import RegistroEjecucion
from orquestacion_langgraph.grafo import procesar_consulta_en_vivo

CARPETA = Path(__file__).parent.parent / "resultados" / "campana"

# Pedido de Cristian (30-09): pasado este tiempo la ejecución es un fallo.
TIMEOUT_SEGUNDOS = 120

# Una ejecución cortada sigue corriendo en su hilo (Python no puede matarlo).
# Se la espera hasta este tope ANTES de lanzar la siguiente, sin contarla, para
# que dos consultas no compitan por el servidor y ensucien la latencia.
ESPERA_TRAS_TIMEOUT_SEGUNDOS = 300

# Verificación previa del servidor de embeddings. Motivo (bitácora 22.6): el
# 2026-10-01 Ollama tenía BGE-M3 en CPU y el servidor estaba saturado por otro
# usuario (carga 391 en 224 núcleos). Cada embedding tardaba ~25 s en vez de
# décimas, y eso infla T2 y T6 con un tiempo que no es del sistema. En
# condiciones normales una consulta tarda menos de 1 s.
UMBRAL_EMBEDDING_SEGUNDOS = 3.0
SONDAS_EMBEDDING = 3


def _commit_actual() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True, cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        return None


def _hay_cambios_sin_commit() -> bool | None:
    try:
        salida = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                check=True, cwd=Path(__file__).parent).stdout
        return bool(salida.strip())
    except Exception:
        return None


def sondear_embeddings(veces: int = SONDAS_EMBEDDING) -> list[float]:
    """Segundos de `veces` embeddings de una consulta corta, como los del RAG."""
    from rag.embeddings import embed_textos

    tiempos = []
    for _ in range(veces):
        inicio = time.perf_counter()
        embed_textos(["¿Qué ingredientes lleva el hornado?"])
        tiempos.append(round(time.perf_counter() - inicio, 2))
    return tiempos


def _recorrer(consulta: str, usuario: str, config: dict) -> tuple[list[str], dict | None]:
    camino, final = [], None
    for paso in procesar_consulta_en_vivo(consulta, id_perfil=usuario, config=config):
        if paso["nodo"]:
            camino.append(paso["nodo"])
        else:
            final = paso["estado_final"]
    return camino, final


def ejecutar(entrada: dict, experimento: str, definiciones: dict, catalogo: list[str],
             timeout: float = TIMEOUT_SEGUNDOS) -> dict:
    tarea, usuario, frase = entrada["tarea"], entrada["usuario"], entrada["frase"]
    registro = RegistroEjecucion()
    # Único por ejecución: si un experimento se reanuda o se repite, el mismo
    # `id` aparece más de una vez en LangSmith y el `uid` permite cruzarlos sin
    # mezclar corridas.
    uid = uuid.uuid4().hex
    config = {
        "callbacks": [registro],
        "tags": [experimento, tarea],
        "metadata": {"experimento": experimento, "id_ejecucion": entrada["id"], "uid": uid, "tarea": tarea,
                     "usuario": usuario, "frase": frase["id"], "repeticion": entrada["repeticion"]},
        "run_name": entrada["id"],
    }

    ejecutor = ThreadPoolExecutor(max_workers=1)
    inicio_iso = datetime.now().isoformat(timespec="seconds")
    inicio = time.perf_counter()
    futuro = ejecutor.submit(_recorrer, frase["consulta"], usuario, config)
    camino, final, error, timeout_ocurrido = [], None, None, False
    try:
        camino, final = futuro.result(timeout=timeout)
    except TiempoAgotado:
        timeout_ocurrido = True
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    segundos = round(time.perf_counter() - inicio, 3)

    if timeout_ocurrido:
        try:
            futuro.result(timeout=ESPERA_TRAS_TIMEOUT_SEGUNDOS)
        except Exception:
            pass
    ejecutor.shutdown(wait=False)

    final = final or {}
    ejecucion = {
        "intencion": final.get("intencion"),
        "camino": camino,
        "respuesta": final.get("respuesta"),
        "traza_rag": final.get("traza_rag"),
        "cruce": final.get("cruce"),
        "error": error,
        "timeout": timeout_ocurrido,
    }
    veredicto = evaluar(tarea, definiciones.get(tarea, {}), frase, usuario, ejecucion, catalogo)
    razonamiento = final.get("razonamiento") or ""

    return {
        "id": entrada["id"], "uid": uid, "experimento": experimento, "tarea": tarea, "usuario": usuario,
        "frase": frase["id"], "consulta": frase["consulta"], "repeticion": entrada["repeticion"],
        "inicio": inicio_iso, "segundos_sistema": segundos,
        "exito": veredicto["exito"], "chequeos": veredicto["chequeos"], "detalle": veredicto["detalle"],
        "decidio_la_red_de_emergencia": razonamiento.startswith("Red de seguridad"),
        **ejecucion,
        "razonamiento": razonamiento,
        **registro.resumen(),
        "llamadas": registro.llamadas,
    }


def _ya_hechas(ruta: Path) -> set[str]:
    if not ruta.exists():
        return set()
    return {json.loads(linea)["id"] for linea in ruta.read_text(encoding="utf-8").splitlines() if linea.strip()}


def main():
    parser = argparse.ArgumentParser(description="Campaña de medición del sistema multiagente")
    parser.add_argument("--experimento", required=True,
                        help="nombre del experimento: es el tag en LangSmith y el nombre del archivo")
    parser.add_argument("--tareas", nargs="+", default=TAREAS, choices=TAREAS)
    parser.add_argument("--usuarios", nargs="+")
    parser.add_argument("--repeticiones", type=int, default=3)
    parser.add_argument("--frases", type=int, help="usar solo las primeras N frases de cada tarea")
    parser.add_argument("--piloto", action="store_true",
                        help="1 usuario × 6 tareas × 1 frase × 1 repetición (chuleta 7.5)")
    parser.add_argument("--reanudar", action="store_true", help="saltar las ejecuciones ya registradas")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SEGUNDOS)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--sin-sonda", action="store_true",
                        help="no verificar la latencia de embeddings antes de empezar")
    args = parser.parse_args()

    campana = cargar_campana()
    usuarios = args.usuarios or campana["usuarios"]
    repeticiones, frases = args.repeticiones, args.frases
    if args.piloto:
        usuarios, repeticiones, frases = usuarios[:1], 1, 1

    CARPETA.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA / f"{args.experimento}.jsonl"
    ruta_meta = CARPETA / f"{args.experimento}_meta.json"
    if ruta.exists() and not args.reanudar:
        print(f"Ya existe {ruta}. Usá --reanudar o elegí otro --experimento.")
        return 1

    plan = plan_de_ejecuciones(args.tareas, usuarios, repeticiones, frases, args.semilla)
    hechas = _ya_hechas(ruta)
    pendientes = [e for e in plan if e["id"] not in hechas]

    meta = {
        "experimento": args.experimento,
        "inicio": datetime.now().isoformat(timespec="seconds"),
        "commit": _commit_actual(),
        "cambios_sin_commit": _hay_cambios_sin_commit(),
        "argumentos": vars(args),
        "usuarios": usuarios, "repeticiones": repeticiones,
        "total_planificado": len(plan),
        "timeout_segundos": args.timeout,
        "resolucion_de_modelos": describir_resolucion(),
        "trazas": describir_trazas(),
        "fingerprints_vistos": [],
    }
    if ruta_meta.exists():
        anterior = json.loads(ruta_meta.read_text(encoding="utf-8"))
        meta["fingerprints_vistos"] = anterior.get("fingerprints_vistos", [])
        meta["reanudaciones"] = anterior.get("reanudaciones", []) + [meta["inicio"]]
        meta["inicio"] = anterior["inicio"]

    if not args.sin_sonda:
        sonda = sondear_embeddings()
        meta["sonda_embeddings_segundos"] = sonda
        print(f"Sonda de embeddings: {sonda} s (umbral {UMBRAL_EMBEDDING_SEGUNDOS} s)")
        if min(sonda) > UMBRAL_EMBEDDING_SEGUNDOS:
            print("  El servidor de embeddings está lento: los tiempos de T2 y T6 no serían los del "
                  "sistema. No se empieza. Revisar `uptime` y `curl .../api/ps` (size_vram), o "
                  "usar --sin-sonda si se quiere medir igual.")
            return 2

    print(f"{args.experimento}: {len(plan)} planificadas, {len(hechas)} ya hechas, {len(pendientes)} por correr")
    print(f"  trazas LangSmith: {meta['trazas']['activo']} ({meta['trazas']['motivo']})\n")

    definiciones = campana["tareas"]
    catalogo = catalogo_de_medicamentos()
    with ruta.open("a", encoding="utf-8") as salida:
        for n, entrada in enumerate(pendientes, 1):
            r = ejecutar(entrada, args.experimento, definiciones, catalogo, args.timeout)
            salida.write(json.dumps(r, ensure_ascii=False) + "\n")
            salida.flush()

            nuevos = [f for f in r["fingerprints"] if f not in meta["fingerprints_vistos"]]
            if nuevos and meta["fingerprints_vistos"]:
                print(f"\n  AVISO: el servidor cambió de fingerprint {meta['fingerprints_vistos']} -> {nuevos}."
                      f" Protocolo: repetir el bloque (repetición {r['repeticion']}).\n")
            meta["fingerprints_vistos"].extend(nuevos)
            ruta_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

            fallidos = [k for k, v in r["chequeos"].items() if not v]
            estado = "OK   " if r["exito"] else f"FALLA {fallidos}"
            tokens = r["tokens_sistema"]
            print(f"  [{n:3}/{len(pendientes)}] {estado} {r['id']:28} {r['segundos_sistema']:6.1f}s "
                  f"tok {tokens['entrada']}/{tokens['salida']} (razon. {tokens['razonamiento']})", flush=True)

    meta["fin"] = datetime.now().isoformat(timespec="seconds")
    ruta_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    filas = [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\nÉxito: {sum(f['exito'] for f in filas)}/{len(filas)} · guardado en {ruta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

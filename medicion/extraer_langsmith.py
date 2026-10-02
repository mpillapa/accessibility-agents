# Extrae de LangSmith los tokens y tiempos de un experimento de la campaña y
# los contrasta con el registro local. Si difieren, se investiga antes de reportar.
#
# Uso (desde la raíz del repo, con LANGSMITH_API_KEY en el .env):
#   python -m medicion.extraer_langsmith --experimento piloto_2026-10-01
#   python -m medicion.extraer_langsmith --experimento piloto_2026-10-01 \
#       --comparar resultados/campana/piloto_2026-10-01.jsonl --json salida.json
#
# Filtra por el tag del experimento, que LangChain propaga a las llamadas hijas.

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from medicion.registro import ubicacion

load_dotenv(Path(__file__).parent.parent / ".env", override=True)


def extraer(experimento: str, proyecto: str | None = None) -> dict[str, dict]:
    """{uid: {id_ejecucion, segundos_raiz, llamadas: [...], tokens_sistema, agentes, fingerprints}}

    Por `uid`: `id_ejecucion` se repite si el experimento se reanudó.
    """
    from langsmith import Client

    proyecto = proyecto or os.getenv("LANGSMITH_PROJECT", "accessibility-agents")
    corridas = list(Client().list_runs(project_name=proyecto, filter=f'has(tags, "{experimento}")'))

    por_ejecucion: dict[str, dict] = defaultdict(lambda: {"segundos_raiz": None, "llamadas": []})
    for run in corridas:
        metadata = (run.extra or {}).get("metadata", {})
        uid = metadata.get("uid")
        if not uid:
            continue
        datos = por_ejecucion[uid]
        datos["id_ejecucion"] = metadata.get("id_ejecucion")
        if run.parent_run_id is None and run.end_time and run.start_time:
            datos["segundos_raiz"] = round((run.end_time - run.start_time).total_seconds(), 3)
        if run.run_type == "llm":
            agente, nodo = ubicacion(metadata)
            llm_output = (run.outputs or {}).get("llm_output") or {}
            datos["llamadas"].append({
                "agente": agente, "nodo": nodo,
                "tokens_entrada": run.prompt_tokens, "tokens_salida": run.completion_tokens,
                "tokens_razonamiento": (run.completion_token_details or {}).get("reasoning"),
                "segundos": round((run.end_time - run.start_time).total_seconds(), 3) if run.end_time else None,
                "fingerprint": llm_output.get("system_fingerprint"),
            })

    for datos in por_ejecucion.values():
        llamadas = datos["llamadas"]
        datos["tokens_sistema"] = {
            "llamadas_llm": len(llamadas),
            "entrada": sum(c["tokens_entrada"] or 0 for c in llamadas),
            "salida": sum(c["tokens_salida"] or 0 for c in llamadas),
            "razonamiento": sum(c["tokens_razonamiento"] or 0 for c in llamadas),
        }
        agentes: dict[str, dict] = {}
        for c in llamadas:
            a = agentes.setdefault(c["agente"], {"llamadas_llm": 0, "tokens_entrada": 0, "tokens_salida": 0})
            a["llamadas_llm"] += 1
            a["tokens_entrada"] += c["tokens_entrada"] or 0
            a["tokens_salida"] += c["tokens_salida"] or 0
        datos["agentes"] = agentes
        datos["fingerprints"] = sorted({c["fingerprint"] for c in llamadas if c["fingerprint"]})
    return dict(por_ejecucion)


def comparar(local: list[dict], remoto: dict[str, dict]) -> dict:
    """Ejecución por ejecución: ¿coinciden los tokens? ¿cuánto difiere el tiempo?"""
    filas, faltan = [], []
    for r in local:
        otro = remoto.get(r["uid"])
        if otro is None:
            faltan.append(r["id"])
            continue
        tl, tr = r["tokens_sistema"], otro["tokens_sistema"]
        filas.append({
            "id": r["id"],
            "tokens_iguales": all(tl[k] == tr[k] for k in ("llamadas_llm", "entrada", "salida", "razonamiento")),
            "local": tl, "langsmith": tr,
            "segundos_local": r["segundos_sistema"], "segundos_langsmith": otro["segundos_raiz"],
        })
    return {
        "ejecuciones_locales": len(local),
        "en_langsmith": len(filas),
        "faltan_en_langsmith": faltan,
        "tokens_iguales": sum(f["tokens_iguales"] for f in filas),
        "filas": filas,
    }


def main():
    parser = argparse.ArgumentParser(description="Extrae un experimento de LangSmith")
    parser.add_argument("--experimento", required=True)
    parser.add_argument("--proyecto")
    parser.add_argument("--comparar", type=Path, help="JSONL local de medicion/campana.py")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    remoto = extraer(args.experimento, args.proyecto)
    print(f"LangSmith: {len(remoto)} ejecuciones con el tag {args.experimento!r}")
    salida = {"experimento": args.experimento, "ejecuciones": remoto}

    if args.comparar:
        local = [json.loads(l) for l in args.comparar.read_text(encoding="utf-8").splitlines() if l.strip()]
        c = comparar(local, remoto)
        print(f"Local: {c['ejecuciones_locales']} · en LangSmith: {c['en_langsmith']} · "
              f"tokens idénticos: {c['tokens_iguales']}/{c['en_langsmith']}")
        if c["faltan_en_langsmith"]:
            print(f"  Faltan en LangSmith: {c['faltan_en_langsmith']}")
        for f in c["filas"]:
            marca = "=" if f["tokens_iguales"] else "≠"
            print(f"  {marca} {f['id']:28} local {f['local']['entrada']}/{f['local']['salida']}/"
                  f"{f['local']['razonamiento']} · LS {f['langsmith']['entrada']}/{f['langsmith']['salida']}/"
                  f"{f['langsmith']['razonamiento']} · {f['segundos_local']}s vs {f['segundos_langsmith']}s")
        salida["comparacion"] = c

    if args.json:
        args.json.write_text(json.dumps(salida, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"Guardado en {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

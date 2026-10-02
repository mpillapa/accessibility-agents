# Evaluación de comportamiento del RAG agéntico contra el LLM y el recetario reales.
# La lógica del grafo con dobles está en pruebas/prueba_ciclo_rag.py.
#
# Uso (desde la raíz del repo):
#   python -m pruebas.evaluar_rag_real
#   python -m pruebas.evaluar_rag_real --json resultados.json
#
# Requiere VPN institucional y el recetario ingerido (python -m rag.ingesta).

import argparse
import json
import sys
import time

from orquestacion_langgraph.rag_agentico.subgrafo import consultar_recetario

# `fuentes_aceptables` es lista porque un plato puede estar en varios archivos;
# None si no debería usar el recetario. `minimo_caracteres` detecta respuestas truncas.
CASOS = [
    {
        "id": "acierto_directo",
        "consulta": "léeme la receta del bolón de verde",
        "espera_resultado": True,
        "fuentes_aceptables": ["bolon de verde.png"],
        "minimo_caracteres": 300,
        "nota": "el usuario nombra el plato tal como está en el recetario",
    },
    {
        "id": "descripcion_sin_nombrar_plato",
        "consulta": "quiero hacer eso de plátano verde majado con chicharrón adentro",
        "espera_resultado": True,
        # El chicharrón lo distingue de tortillas y empanadas de verde.
        "fuentes_aceptables": ["bolon de verde.png"],
        "minimo_caracteres": 300,
        "nota": "describe el plato sin nombrarlo, con un ingrediente que lo distingue",
    },
    {
        "id": "no_esta_en_recetario",
        "consulta": "quiero preparar sushi",
        "espera_resultado": False,
        "fuentes_aceptables": None,
        "minimo_caracteres": 0,
        "nota": "no está en el recetario: debe admitirlo, no apoyarse en una receta de arroz",
    },
    {
        "id": "receta_multi_paso",
        "consulta": "como hago el llapingacho",
        "espera_resultado": True,
        "fuentes_aceptables": ["llapingacho.jpg", "receta1.jpeg"],
        # Una receta completa no entra en menos; la línea base daba 243 (un paso).
        "minimo_caracteres": 500,
        "nota": "la receta está repartida en varios fragmentos: debe responder completa",
    },
    {
        "id": "sin_necesidad_de_buscar",
        "consulta": "gracias mijito, ya me salio rico",
        "espera_resultado": True,
        "fuentes_aceptables": None,
        "minimo_caracteres": 0,
        "nota": "un agradecimiento no necesita el recetario",
    },
]


def _fuentes_usadas(traza: list[dict]) -> list[str]:
    for paso in reversed(traza):
        if paso["nodo"] == "generar":
            return paso.get("fuentes", [])
    return []


def _consulto_recetario(traza: list[dict]) -> bool:
    return any(p["nodo"] == "recuperar" for p in traza)


def evaluar_caso(caso: dict) -> dict:
    inicio = time.time()
    resultado = consultar_recetario(caso["consulta"])
    latencia = round(time.time() - inicio, 2)

    fuentes = _fuentes_usadas(resultado["traza"])
    respuesta = resultado["respuesta"] or ""
    aceptables = caso["fuentes_aceptables"]

    # Solo señales objetivas; la calidad de la redacción se juzga leyendo.
    return {
        "id": caso["id"],
        "consulta": caso["consulta"],
        "nota": caso["nota"],
        "hubo_resultado": resultado["hubo_resultado"],
        "espera_resultado": caso["espera_resultado"],
        "resultado_correcto": resultado["hubo_resultado"] == caso["espera_resultado"],
        "fuentes_usadas": fuentes,
        "fuentes_aceptables": aceptables,
        "fuente_correcta": (
            any(f in aceptables for f in fuentes) if aceptables else not fuentes
        ),
        "consulto_recetario": _consulto_recetario(resultado["traza"]),
        "intentos": resultado["intentos"],
        "caracteres_respuesta": len(respuesta),
        "minimo_caracteres": caso["minimo_caracteres"],
        "respuesta_completa": len(respuesta) >= caso["minimo_caracteres"],
        "camino": [p["nodo"] for p in resultado["traza"]],
        "latencia_s": latencia,
        "respuesta": respuesta,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="guarda los resultados crudos en este archivo")
    args = parser.parse_args()

    print("Evaluación del RAG agéntico con LLM y recetario reales")
    print("(requiere VPN y 'python -m rag.ingesta' ya corrido)\n")

    resultados = []
    for caso in CASOS:
        try:
            r = evaluar_caso(caso)
        except Exception as e:
            print(f"  ERROR en {caso['id']}: {type(e).__name__}: {e}")
            continue
        resultados.append(r)

        def marca(ok):
            return "OK  " if ok else "MAL "

        print(f"[{r['id']}]  {r['consulta']!r}")
        print(f"    {r['nota']}")
        print(f"    camino: {' -> '.join(r['camino'])}")
        print(f"    [{marca(r['resultado_correcto'])}] encontró={r['hubo_resultado']} (esperado {r['espera_resultado']})")
        print(f"    [{marca(r['fuente_correcta'])}] fuentes={r['fuentes_usadas']} (aceptables {r['fuentes_aceptables']})")
        print(f"    [{marca(r['respuesta_completa'])}] respuesta={r['caracteres_respuesta']} car (mínimo {r['minimo_caracteres']})")
        print(f"    intentos={r['intentos']}  {r['latencia_s']}s")
        print(f"    -> {r['respuesta'][:240].replace(chr(10), ' ')}")
        print()

    if not resultados:
        print("No se pudo evaluar ningún caso.")
        return 1

    total = len(resultados)
    print("=" * 74)
    print(f"encontró/no-encontró correcto : {sum(r['resultado_correcto'] for r in resultados)}/{total}")
    print(f"fuente correcta               : {sum(r['fuente_correcta'] for r in resultados)}/{total}")
    print(f"respuesta completa            : {sum(r['respuesta_completa'] for r in resultados)}/{total}")
    print(f"latencia media                : {sum(r['latencia_s'] for r in resultados)/total:.2f}s")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(resultados, f, ensure_ascii=False, indent=2)
        print(f"\nResultados crudos en {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

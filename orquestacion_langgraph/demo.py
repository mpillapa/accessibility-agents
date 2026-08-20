# Demo en vivo de la réplica en LangGraph. Muestra la intención detectada por
# el Orchestrator, el nodo especialista al que se ruteó y la respuesta final.
#
# Uso (desde la raíz del repo):
#   python -m orquestacion_langgraph.demo                 -> corre las frases de ejemplo
#   python -m orquestacion_langgraph.demo "tu frase aquí" -> corre una frase que tú escribes

import sys
import time

import requests
from orquestacion_langgraph.agentes import VLLM_CHAT_BASE_URL, VLLM_CHAT_MODEL, VLLM_API_KEY
from orquestacion_langgraph.grafo import procesar_consulta_verbose

CONSULTAS_DEMO = [
    "¿ya me toca la pastillita del corazón?",       # MEDICATION_HEALTH
    "avísale a mi hija que ya almorcé",             # FAMILY_COMMUNICATION
    "léeme la receta del arroz con leche",          # RECIPE_MULTIMEDIA — camino directo
    "eso dulce del arrocito que hacía mi mamá",     # RECIPE_MULTIMEDIA — obliga a reformular
    "quiero preparar sushi",                        # RECIPE_MULTIMEDIA — no está: debe admitirlo
]


# Comprueba que el servidor vLLM remoto responde antes de arrancar la demo.
def verificar_vllm():
    print(f"Servidor vLLM: {VLLM_CHAT_BASE_URL}")
    print(f"Modelo:        {VLLM_CHAT_MODEL}\n")
    try:
        inicio = time.time()
        r = requests.get(
            f"{VLLM_CHAT_BASE_URL}/models",
            headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
            timeout=10,
        )
        latencia = time.time() - inicio

        modelos = [m["id"] for m in r.json().get("data", [])]
        if VLLM_CHAT_MODEL not in modelos:
            print(f"AVISO: {VLLM_CHAT_MODEL} no aparece en el servidor. Modelos disponibles: {modelos}")
            return False

        print(f"Conexión OK ({latencia*1000:.0f} ms de round-trip a {VLLM_CHAT_BASE_URL})")
        print(f"Modelos en el servidor: {modelos}\n")
        return True
    except requests.exceptions.ConnectionError:
        print(f"ERROR: {VLLM_CHAT_BASE_URL} no responde. Verifica la VPN institucional (GlobalProtect).")
        return False


def correr(consulta):
    print("=" * 70)
    print(f"CONSULTA: {consulta}")
    print("=" * 70)
    print(f"[ENTRADA] consulta -> {consulta!r}\n")

    resultado = procesar_consulta_verbose(consulta)

    # Si pasó por el especialista en recetas, muestra qué hizo el subgrafo de
    # RAG por dentro: si decidió buscar, cuántos fragmentos aceptó o descartó,
    # y si tuvo que reformular la consulta.
    if resultado.get("traza_rag"):
        print("-" * 70)
        print("CICLO DEL RAG AGÉNTICO (dentro del nodo de recetas):")
        for paso in resultado["traza_rag"]:
            detalle = {k: v for k, v in paso.items() if k != "nodo"}
            print(f"    [{paso['nodo']}] {detalle}")
        print()

    print("-" * 70)
    print("RESPUESTA FINAL:")
    print(resultado["respuesta"])
    print(f"\n(latencia: {resultado['latencia_segundos']}s)\n")


def main():
    if not verificar_vllm():
        return

    if len(sys.argv) > 1:
        correr(" ".join(sys.argv[1:]))
    else:
        for consulta in CONSULTAS_DEMO:
            correr(consulta)


if __name__ == "__main__":
    main()

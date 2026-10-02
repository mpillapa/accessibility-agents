# Antecedente fuera de alcance: demo de la delegación del Orchestrator en CrewAI
# (verbose=True muestra a quién delega).
#
# Uso (desde la raíz del repo, con VPN):
#   python -m orquestacion_crewai.demo                 -> corre las frases de ejemplo
#   python -m orquestacion_crewai.demo "tu frase aquí" -> corre una frase que tú escribes

import sys
import time

import requests
from crewai import Task, Crew, Process
from orquestacion_crewai.agentes import (
    VLLM_CHAT_BASE_URL,
    VLLM_CHAT_MODEL,
    VLLM_API_KEY,
    crear_orchestrator,
    crear_agente_medicacion,
    crear_agente_recetas,
    crear_agente_familia_stub,
    crear_agente_emergencia_stub,
)

# Una frase por intención, elegidas por ser claras para la demo.
CONSULTAS_DEMO = [
    "¿ya me toca la pastillita del corazón?",       # MEDICATION_HEALTH
    "avísale a mi hija que ya almorcé",             # FAMILY_COMMUNICATION
    "léeme la receta del arroz con leche",          # RECIPE_MULTIMEDIA
]


# Comprueba que el servidor vLLM responde y muestra la latencia del round-trip.
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


def construir_crew(consulta):
    orchestrator = crear_orchestrator()
    especialistas = [
        crear_agente_medicacion(),
        crear_agente_recetas(),
        crear_agente_familia_stub(),
        crear_agente_emergencia_stub(),
    ]

    tarea = Task(
        description=(
            f"El usuario adulto mayor dijo: '{consulta}'. "
            f"Identifica de qué tipo de consulta se trata y delega al especialista "
            f"correspondiente. Si es conversación trivial (saludo, comentario), "
            f"responde tú directamente de forma amable y breve."
        ),
        expected_output="Respuesta en español del especialista correspondiente.",
        agent=orchestrator,
    )

    return Crew(
        agents=[orchestrator] + especialistas,
        tasks=[tarea],
        process=Process.sequential,
        verbose=True,   # muestra el razonamiento y la delegación
    )


def correr(consulta):
    print("=" * 70)
    print(f"CONSULTA: {consulta}")
    print("=" * 70)

    inicio = time.time()
    resultado = construir_crew(consulta).kickoff()
    latencia = time.time() - inicio

    print("\n" + "-" * 70)
    print("RESPUESTA FINAL:")
    print(resultado)
    print(f"\n(latencia: {latencia:.1f}s)\n")


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

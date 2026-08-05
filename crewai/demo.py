# Demo en vivo: el Orchestrator identifica la intención y delega al
# especialista (allow_delegation=True, sin clasificador). Con verbose=True se
# ve en pantalla cómo el agente razona y a quién delega.
#
# Uso:
#   python crewai/demo.py                 -> corre las frases de ejemplo
#   python crewai/demo.py "tu frase aquí" -> corre una frase que tú escribes
#
# El LLM corre en el servidor vLLM remoto de la universidad (ver agentes.py:
# VLLM_CHAT_BASE_URL / VLLM_CHAT_MODEL), no local. La demo comprueba ese
# endpoint (OpenAI-compatible: GET /models) y mide el round-trip real contra
# el servidor para dejar visible que sí se conecta.

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import requests
from crewai import Task, Crew, Process
from agentes import (
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


# Comprueba que el servidor vLLM remoto responde antes de arrancar la demo.
# Imprime host, modelo y latencia del round-trip para dejar visible ante los
# tutores que la demo sí está hablando con el servidor de la universidad.
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


# Ensambla la crew con verbose=True para que la delegación sea visible.
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
        verbose=True,   # <- clave de la demo: muestra el razonamiento y la delegación
    )


# Corre una sola consulta y muestra la respuesta final con su latencia.
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

    # Si pasas una frase por línea de comandos, corre solo esa.
    if len(sys.argv) > 1:
        correr(" ".join(sys.argv[1:]))
    else:
        for consulta in CONSULTAS_DEMO:
            correr(consulta)


if __name__ == "__main__":
    main()

# Demo en vivo de la estrategia A2: el Orchestrator usa el clasificador como
# herramienta (@tool). Con verbose=True se ve en pantalla cómo el agente llama
# a la herramienta, obtiene la intención y luego delega al especialista.
#
# Uso:
#   python src/demo_a2.py                 -> corre las frases de ejemplo
#   python src/demo_a2.py "tu frase aquí" -> corre una frase que tú escribes
#
# Requiere Ollama corriendo con qwen2.5:7b.

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import requests
from crewai import Task, Crew, Process
from agentes import (
    crear_orchestrator_con_herramienta,
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


# Comprueba que Ollama responde antes de arrancar la demo.
def verificar_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        modelos = [m["name"] for m in r.json().get("models", [])]
        if not any("qwen2.5" in m for m in modelos):
            print("AVISO: qwen2.5:7b no aparece. Ejecuta: ollama pull qwen2.5:7b")
            return False
        print(f"Ollama OK. Modelos: {modelos}\n")
        return True
    except requests.exceptions.ConnectionError:
        print("ERROR: Ollama no responde en localhost:11434. Arráncalo antes de la demo.")
        return False


# Ensambla la crew de A2 con verbose=True para que la llamada al tool sea visible.
def construir_crew(consulta):
    orchestrator = crear_orchestrator_con_herramienta()
    especialistas = [
        crear_agente_medicacion(),
        crear_agente_recetas(),
        crear_agente_familia_stub(),
        crear_agente_emergencia_stub(),
    ]

    tarea = Task(
        description=(
            f"El usuario dijo: '{consulta}'. "
            f"Usa la herramienta de clasificación para identificar la intención, "
            f"luego delega al especialista correcto."
        ),
        expected_output="Respuesta en español del especialista correspondiente.",
        agent=orchestrator,
    )

    return Crew(
        agents=[orchestrator] + especialistas,
        tasks=[tarea],
        process=Process.sequential,
        verbose=True,   # <- clave de la demo: muestra el razonamiento y el tool call
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
    if not verificar_ollama():
        return

    # Si pasas una frase por línea de comandos, corre solo esa.
    if len(sys.argv) > 1:
        correr(" ".join(sys.argv[1:]))
    else:
        for consulta in CONSULTAS_DEMO:
            correr(consulta)


if __name__ == "__main__":
    main()

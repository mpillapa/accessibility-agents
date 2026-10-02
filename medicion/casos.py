# Casos de la campaña: qué se pregunta, a quién y qué se espera.
# T6 se lee de interacciones/datos/casos_prueba.json para no duplicar su verdad de referencia.

import json
import random
from pathlib import Path

from interacciones.datos import cargar_casos as cargar_casos_interacciones

RUTA_CASOS = Path(__file__).parent / "datos" / "casos_campana.json"

TAREAS = ["T1", "T2", "T3", "T4", "T5", "T6"]


def cargar_campana() -> dict:
    return json.loads(RUTA_CASOS.read_text(encoding="utf-8"))


def frases_de(tarea: str, campana: dict | None = None) -> list[dict]:
    """Las frases de una tarea, cada una con su `id` y su `consulta`."""
    campana = campana or cargar_campana()
    if tarea == "T6":
        return cargar_casos_interacciones()["campana"]["frases"]
    return campana["tareas"][tarea]["frases"]


def plan_de_ejecuciones(tareas: list[str], usuarios: list[str], repeticiones: int,
                        frases_por_tarea: int | None = None, semilla: int = 42) -> list[dict]:
    """Ejecuciones de la campaña, intercaladas al azar (semilla fija) dentro de cada repetición.

    La repetición r termina antes de la r+1, para poder repetir un bloque (chuleta 7.5).
    """
    campana = cargar_campana()
    generador = random.Random(semilla)
    plan = []
    for repeticion in range(1, repeticiones + 1):
        bloque = [
            {"tarea": tarea, "usuario": usuario, "frase": frase, "repeticion": repeticion,
             "id": f"{tarea}-{frase['id']}-{usuario}-r{repeticion}"}
            for tarea in tareas
            for frase in frases_de(tarea, campana)[:frases_por_tarea]
            for usuario in usuarios
        ]
        generador.shuffle(bloque)
        plan.extend(bloque)
    return plan

# Carga y búsqueda del vademécum, perfiles y prescripciones (datos ficticios). No decide nada.
# "Receta" en el repo es de cocina; la médica se llama prescripción (README, "Nombres").

import json
from functools import lru_cache
from pathlib import Path

DIRECTORIO_DATOS = Path(__file__).parent / "datos"
ARCHIVO_MEDICAMENTOS = DIRECTORIO_DATOS / "medicamentos.json"
ARCHIVO_PERFILES = DIRECTORIO_DATOS / "perfiles.json"
ARCHIVO_PRESCRIPCIONES = DIRECTORIO_DATOS / "prescripciones.json"


@lru_cache(maxsize=1)
def cargar_medicamentos() -> list[dict]:
    """El vademécum ficticio: material de consulta para verificar y buscar alternativas."""
    return json.loads(ARCHIVO_MEDICAMENTOS.read_text(encoding="utf-8"))["medicamentos"]


@lru_cache(maxsize=1)
def cargar_perfiles() -> list[dict]:
    return json.loads(ARCHIVO_PERFILES.read_text(encoding="utf-8"))["perfiles"]


@lru_cache(maxsize=1)
def cargar_prescripciones() -> list[dict]:
    """Fuente de verdad de qué toma cada persona."""
    return json.loads(ARCHIVO_PRESCRIPCIONES.read_text(encoding="utf-8"))["prescripciones"]


def obtener_perfil(id_perfil: str) -> dict | None:
    """Un perfil por id, o None: quien llama decide qué hacer, nunca adivinando."""
    return next((p for p in cargar_perfiles() if p["id"] == id_perfil), None)


def prescripciones_de(id_perfil: str) -> list[dict]:
    """Todas sus prescripciones, vigentes o no (puede haber crónica y aguda a la vez)."""
    return [r for r in cargar_prescripciones() if r["id_perfil"] == id_perfil]


def buscar_medicamento(nombre: str) -> dict | None:
    """Por nombre, ignorando mayúsculas y tildes (los nombres llegan escritos o por voz)."""
    objetivo = _normalizar(nombre)
    return next((m for m in cargar_medicamentos() if _normalizar(m["nombre"]) == objetivo), None)


def _normalizar(texto: str) -> str:
    import unicodedata

    sin_tildes = unicodedata.normalize("NFKD", texto)
    sin_tildes = "".join(c for c in sin_tildes if not unicodedata.combining(c))
    return sin_tildes.strip().lower()

# Carga del vademécum, los perfiles y las prescripciones. No decide nada: solo
# lee y busca.
#
# Los tres archivos son DATOS FICTICIOS de un prototipo académico. Ver el campo
# "_aviso" de cada JSON.
#
# NOMBRES: en este repositorio "receta" (rag/recetas_data, el intent
# RECIPE_MULTIMEDIA) significa receta DE COCINA. La receta médica se llama
# PRESCRIPCIÓN en todo el código y en la memoria, para que no se confundan.

import json
from functools import lru_cache
from pathlib import Path

DIRECTORIO_DATOS = Path(__file__).parent / "datos"
ARCHIVO_MEDICAMENTOS = DIRECTORIO_DATOS / "medicamentos.json"
ARCHIVO_PERFILES = DIRECTORIO_DATOS / "perfiles.json"
ARCHIVO_PRESCRIPCIONES = DIRECTORIO_DATOS / "prescripciones.json"


@lru_cache(maxsize=1)
def cargar_medicamentos() -> list[dict]:
    """El vademécum ficticio completo, 62 entradas.

    Ya NO es el menú del que el sistema elige: es material de CONSULTA, para
    validar lo que dice una prescripción y para buscar alternativas del mismo
    grupo cuando a la persona se le acabó algo.
    """
    return json.loads(ARCHIVO_MEDICAMENTOS.read_text(encoding="utf-8"))["medicamentos"]


@lru_cache(maxsize=1)
def cargar_perfiles() -> list[dict]:
    """Las personas ficticias sobre las que se resuelven las consultas."""
    return json.loads(ARCHIVO_PERFILES.read_text(encoding="utf-8"))["perfiles"]


@lru_cache(maxsize=1)
def cargar_prescripciones() -> list[dict]:
    """Las recetas médicas ficticias. Fuente de verdad de qué toma cada persona."""
    return json.loads(ARCHIVO_PRESCRIPCIONES.read_text(encoding="utf-8"))["prescripciones"]


def obtener_perfil(id_perfil: str) -> dict | None:
    """Un perfil por id, o None si no existe.

    Devolver None en vez de lanzar es deliberado: quien llama tiene que decidir
    qué hacer sin perfil, y en este dominio "no sé de quién me hablás" nunca
    debe resolverse adivinando.
    """
    return next((p for p in cargar_perfiles() if p["id"] == id_perfil), None)


def prescripciones_de(id_perfil: str) -> list[dict]:
    """Todas las prescripciones de una persona, vigentes o no.

    Puede devolver varias: una persona puede tener a la vez un tratamiento
    crónico y uno agudo. Consolidarlas en un solo plan del día es trabajo de
    medicacion/prescripciones.py, no de acá.
    """
    return [r for r in cargar_prescripciones() if r["id_perfil"] == id_perfil]


def buscar_medicamento(nombre: str) -> dict | None:
    """Una entrada del vademécum por nombre, sin distinguir mayúsculas ni tildes.

    Existe porque los nombres llegan escritos por personas ("se me acabó el
    paracetamol") o transcritos por Whisper, no copiados del JSON.
    """
    objetivo = _normalizar(nombre)
    return next((m for m in cargar_medicamentos() if _normalizar(m["nombre"]) == objetivo), None)


def _normalizar(texto: str) -> str:
    import unicodedata

    sin_tildes = unicodedata.normalize("NFKD", texto)
    sin_tildes = "".join(c for c in sin_tildes if not unicodedata.combining(c))
    return sin_tildes.strip().lower()

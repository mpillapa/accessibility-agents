# Carga del catálogo de interacciones, los ingredientes por receta y los casos
# de prueba. No decide nada: solo lee y busca. El cruce (qué medicamento choca
# con qué plato para qué persona) es regla de negocio y va en reglas.py.

import json
from functools import lru_cache
from pathlib import Path

DIRECTORIO_DATOS = Path(__file__).parent / "datos"
ARCHIVO_INTERACCIONES = DIRECTORIO_DATOS / "interacciones_alimentarias.json"
ARCHIVO_INGREDIENTES = DIRECTORIO_DATOS / "ingredientes_recetas.json"
ARCHIVO_CASOS = DIRECTORIO_DATOS / "casos_prueba.json"

SEVERIDADES = ("evitar", "precaucion")
PARTES = ("principal", "acompanamiento", "opcional")


def _leer(archivo: Path) -> dict:
    return json.loads(archivo.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def cargar_catalogo() -> dict:
    """El catálogo completo: categorías de alimento, interacciones y la lista
    de medicamentos revisados sin interacción registrada."""
    return _leer(ARCHIVO_INTERACCIONES)


def categorias_alimento() -> dict[str, dict]:
    return cargar_catalogo()["categorias_alimento"]


def interacciones() -> list[dict]:
    return cargar_catalogo()["interacciones"]


def interacciones_de(medicamento: str) -> list[dict]:
    """Las interacciones registradas para un medicamento. Lista vacía si no
    tiene ninguna, que NO es lo mismo que no haberlo revisado: para eso está
    `medicamentos_revisados()`."""
    return [i for i in interacciones() if i["medicamento"] == medicamento]


def medicamentos_revisados() -> set[str]:
    """Los medicamentos que el catálogo cubre, con o sin interacción. Un
    medicamento recetado que no esté acá es un hueco del catálogo, no un
    medicamento seguro."""
    catalogo = cargar_catalogo()
    con_interaccion = {i["medicamento"] for i in catalogo["interacciones"]}
    return con_interaccion | set(catalogo["sin_interacciones_registradas"])


@lru_cache(maxsize=1)
def cargar_ingredientes() -> dict[str, dict]:
    """Los ingredientes de cada receta del índice, por nombre de fuente."""
    return _leer(ARCHIVO_INGREDIENTES)["fuentes"]


def alimentos_marcados_de(fuente: str) -> list[dict] | None:
    """Los alimentos con categoría de interacción de una fuente, sumando todos
    sus platos. Cada uno lleva el nombre del plato del que sale.

    Devuelve None si la fuente no está en el archivo. Quien llama tiene que
    tratar eso como "no sé qué lleva", nunca como "no lleva nada".
    """
    entrada = cargar_ingredientes().get(fuente)
    if entrada is None:
        return None
    return [
        {**alimento, "plato": plato["nombre"]}
        for plato in entrada["platos"]
        for alimento in plato["alimentos_marcados"]
    ]


@lru_cache(maxsize=1)
def cargar_casos() -> dict:
    """Los casos con la verdad de referencia: 'desarrollo' (fuente dada) y
    'campana' (frases de la tarea T6 para la medición)."""
    return _leer(ARCHIVO_CASOS)

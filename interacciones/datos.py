# Carga del catálogo de interacciones, ingredientes por receta y casos de prueba. No decide
# nada: el cruce es regla de negocio y va en reglas.py.

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
    """Categorías de alimento, interacciones y medicamentos revisados sin interacción."""
    return _leer(ARCHIVO_INTERACCIONES)


def categorias_alimento() -> dict[str, dict]:
    return cargar_catalogo()["categorias_alimento"]


def interacciones() -> list[dict]:
    return cargar_catalogo()["interacciones"]


def interacciones_de(medicamento: str) -> list[dict]:
    """Lista vacía no implica revisado: para eso está `medicamentos_revisados()`."""
    return [i for i in interacciones() if i["medicamento"] == medicamento]


def medicamentos_revisados() -> set[str]:
    """Los que el catálogo cubre. Uno recetado que no esté es un hueco, no un medicamento seguro."""
    catalogo = cargar_catalogo()
    con_interaccion = {i["medicamento"] for i in catalogo["interacciones"]}
    return con_interaccion | set(catalogo["sin_interacciones_registradas"])


@lru_cache(maxsize=1)
def cargar_ingredientes() -> dict[str, dict]:
    """Ingredientes por nombre de fuente del índice."""
    return _leer(ARCHIVO_INGREDIENTES)["fuentes"]


def alimentos_marcados_de(fuente: str) -> list[dict] | None:
    """Alimentos marcados de todos los platos de la fuente, cada uno con su plato.

    None si la fuente no está: significa "no sé qué lleva", nunca "no lleva nada".
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
    """Verdad de referencia: 'desarrollo' (fuente dada) y 'campana' (frases de T6)."""
    return _leer(ARCHIVO_CASOS)

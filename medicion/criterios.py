# Criterios de éxito de la campaña (chuleta 7.2). Funciones puras: reciben lo
# que devolvió una ejecución y la verdad de referencia, y dicen qué chequeos
# pasaron. Se prueban sin servidor en pruebas/prueba_medicion.py.
#
# Regla común a todas las tareas: un timeout o una excepción es FALLO (pedido de
# Cristian el 30-09), aunque lo que alcanzó a salir fuera correcto.
#
# Los criterios son automáticos y por eso tienen falsos positivos y negativos.
# Todo fallo se lee antes de contarlo (bitácora 16.6).

import re
import unicodedata

from medicacion.datos import cargar_medicamentos

CAMINO_SIMPLE = {
    "T1": "medicacion",
    "T2": "recetas",
    "T3": "emergencia",
    "T4": "familia",
    "T5": "small_talk",
}


def normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in sin_tildes if not unicodedata.combining(c)).lower()


def nombra(texto: str, medicamento: str) -> bool:
    """Si el texto nombra el medicamento como palabra completa, sin importar
    tildes ni mayúsculas ('Losartan' = 'losartán')."""
    return re.search(rf"\b{re.escape(normalizar(medicamento))}\b", normalizar(texto)) is not None


def catalogo_de_medicamentos() -> list[str]:
    datos = cargar_medicamentos()
    lista = datos["medicamentos"] if isinstance(datos, dict) else datos
    return [m["nombre"] for m in lista]


def camino_t6_correcto(camino: list[str]) -> bool:
    return (len(camino) == 4 and camino[0] == "orchestrator"
            and set(camino[1:3]) == {"medicacion_cruce", "recetas_cruce"} and camino[3] == "integrador")


def fuentes_de_la_respuesta(traza_rag: list[dict] | None) -> list[str]:
    """Las fuentes con las que el RAG redactó la respuesta (paso `generar`).
    Lista vacía si no generó: no encontró nada o decidió no buscar."""
    for paso in reversed(traza_rag or []):
        if paso.get("nodo") == "generar":
            return paso.get("fuentes", [])
    return []


def evaluar(tarea: str, definicion: dict, frase: dict, usuario: str, ejecucion: dict,
            catalogo: list[str] | None = None) -> dict:
    """Devuelve {"exito", "chequeos", "detalle"}.

    `definicion` es la entrada de la tarea en casos_campana.json; `ejecucion`,
    lo que registró medicion/campana.py (intencion, camino, respuesta,
    traza_rag, cruce, error, timeout).
    """
    chequeos = {"sin_error": not ejecucion.get("error") and not ejecucion.get("timeout")}
    detalle = {}
    camino = ejecucion.get("camino") or []
    respuesta = ejecucion.get("respuesta") or ""

    if tarea == "T6":
        from interacciones.reglas import pares

        esperado = frase["esperado_por_usuario"][usuario]
        cruce = ejecucion.get("cruce") or {}
        fuentes = cruce.get("fuentes_cruzadas", [])
        obtenidos = pares(cruce) if cruce else set()
        esperados = {tuple(p) for p in esperado["interacciones"]}
        chequeos.update({
            "intencion": ejecucion.get("intencion") == definicion["intencion"],
            "camino": camino_t6_correcto(camino),
            "fuente": any(f in frase["fuentes_aceptadas"] for f in fuentes),
            "interacciones": cruce.get("evaluable", False) and obtenidos == esperados,
        })
        detalle = {"fuentes": fuentes, "interacciones_obtenidas": sorted(obtenidos),
                   "interacciones_esperadas": sorted(esperados)}
    else:
        chequeos["intencion"] = ejecucion.get("intencion") == definicion["intencion"]
        chequeos["camino"] = camino == ["orchestrator", CAMINO_SIMPLE[tarea]]

        if tarea == "T1":
            propios = definicion["medicamentos_por_usuario"][usuario]
            catalogo = catalogo if catalogo is not None else catalogo_de_medicamentos()
            faltan = [m for m in propios if not nombra(respuesta, m)]
            ajenos = [m for m in catalogo if m not in propios and nombra(respuesta, m)]
            chequeos["nombra_todos"] = not faltan
            chequeos["ninguno_ajeno"] = not ajenos
            detalle = {"faltan": faltan, "ajenos": ajenos}

        elif tarea == "T2":
            fuentes = fuentes_de_la_respuesta(ejecucion.get("traza_rag"))
            aceptadas = frase["fuentes_aceptadas"]
            chequeos["fuente"] = any(f in aceptadas for f in fuentes) if aceptadas else not fuentes
            detalle = {"fuentes": fuentes, "fuentes_aceptadas": aceptadas}

    return {"exito": all(chequeos.values()), "chequeos": chequeos, "detalle": detalle}

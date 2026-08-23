# Gestión del corpus de audio: qué frases faltan grabar y dónde se guardan.
#
# El corpus se arma a partir de dataset.csv, que ya tiene 415 frases etiquetadas
# por intención. Grabarlas permite medir algo que no se puede medir con audio
# suelto: cuánta precisión de ruteo pierde el sistema al entrar por voz en vez
# de por texto.
#
# El índice es un CSV para que se pueda abrir en Excel y revisar a mano. Es un
# formato de prototipo, no una base de datos: sirve para esta etapa y no debe
# presentarse como arquitectura definitiva.

import csv
from pathlib import Path

from asr.config import CORPUS_GRABACIONES_DIR, CORPUS_INDICE

DATASET = Path(__file__).parent.parent / "dataset.csv"

CAMPOS_INDICE = [
    "id_frase",       # identificador estable: fila del dataset
    "archivo",        # nombre del wav dentro de grabaciones/
    "intencion",      # etiqueta del dataset (la verdad de referencia del ruteo)
    "texto",          # transcripción correcta (la verdad de referencia del ASR)
    "hablante",       # quién grabó: permite comparar voces distintas
    "duracion_s",
]


def cargar_frases() -> list[dict]:
    """Las frases del dataset, con un id estable por posición."""
    with open(DATASET, encoding="utf-8-sig") as f:
        return [
            {"id_frase": f"f{i:04d}", "intencion": fila["intent"], "texto": fila["text"]}
            for i, fila in enumerate(csv.DictReader(f))
        ]


def cargar_indice() -> list[dict]:
    """Lo que ya está grabado. Lista vacía si no se grabó nada todavía."""
    if not CORPUS_INDICE.exists():
        return []
    with open(CORPUS_INDICE, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def registrar(fila: dict) -> None:
    """Agrega (o reemplaza) una grabación en el índice.

    Reemplaza en vez de duplicar para que se pueda regrabar una frase que salió
    mal sin ensuciar el corpus.
    """
    CORPUS_INDICE.parent.mkdir(parents=True, exist_ok=True)
    filas = [f for f in cargar_indice() if f["id_frase"] != fila["id_frase"]]
    filas.append({campo: fila.get(campo, "") for campo in CAMPOS_INDICE})
    filas.sort(key=lambda f: f["id_frase"])

    with open(CORPUS_INDICE, "w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS_INDICE)
        escritor.writeheader()
        escritor.writerows(filas)


def progreso() -> dict:
    """Cuánto falta, por intención.

    Importa el desglose y no solo el total: para medir accuracy de ruteo hacen
    falta frases de las cinco intenciones. 200 frases de una sola intención no
    sirven para eso.
    """
    frases = cargar_frases()
    grabadas = {f["id_frase"] for f in cargar_indice()}

    por_intencion = {}
    for frase in frases:
        etiqueta = frase["intencion"]
        datos = por_intencion.setdefault(etiqueta, {"total": 0, "grabadas": 0})
        datos["total"] += 1
        if frase["id_frase"] in grabadas:
            datos["grabadas"] += 1

    return {
        "total": len(frases),
        "grabadas": len(grabadas),
        "por_intencion": por_intencion,
    }


def siguiente_pendiente(intencion: str | None = None) -> dict | None:
    """La próxima frase sin grabar.

    Recorre las intenciones de forma intercalada, no en bloque: así el corpus
    queda balanceado desde el principio y con 50 frases grabadas ya se puede
    medir ruteo, en vez de tener 50 frases de una sola intención.
    """
    grabadas = {f["id_frase"] for f in cargar_indice()}
    pendientes = [f for f in cargar_frases() if f["id_frase"] not in grabadas]
    if intencion:
        pendientes = [f for f in pendientes if f["intencion"] == intencion]
    if not pendientes:
        return None

    # Intercalado: se elige la intención con menos grabadas hasta ahora.
    conteo = progreso()["por_intencion"]
    def prioridad(frase):
        datos = conteo.get(frase["intencion"], {"grabadas": 0})
        return (datos["grabadas"], frase["id_frase"])

    return min(pendientes, key=prioridad)


def ruta_audio(id_frase: str) -> Path:
    return CORPUS_GRABACIONES_DIR / f"{id_frase}.wav"

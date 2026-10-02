# Ingesta del recetario: OCR de las imágenes, troceado, control de calidad,
# embeddings y colección de ChromaDB en rag/chroma_db/. Requiere VPN.
#
# Uso (desde la raíz del repo):
#   python -m rag.ingesta                                # llama al OCR
#   python -m rag.ingesta --desde-ocr resultados/X.json  # usa un OCR guardado
#
# --desde-ocr reconstruye el índice aunque el modelo de OCR ya no exista (bitácora 11).

import argparse
import json
from pathlib import Path

import chromadb

from rag.calidad import evaluar_texto
from rag.config import CHROMA_COLLECTION, CHROMA_DIR, RECETAS_DATA_DIR
from rag.embeddings import embed_textos
from rag.ocr import extraer_texto_de_imagen

EXTENSIONES_TEXTO = {".txt", ".md"}
EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".webp"}

# Los fragmentos legítimos llegan a ~900 caracteres; el tope es para bloques de
# OCR sin saltos de línea. Debe caber en la ventana de embeddings o el vector no
# corresponde al texto guardado (bitácora 5.4).
MAXIMO_CARACTERES_FRAGMENTO = 1200

# Más corto que esto se fusiona con el siguiente: los encabezados sueltos
# ("PREPARACIÓN") contaminaban la búsqueda (bitácora 3.4).
MINIMO_CARACTERES_FRAGMENTO = 40


def _partir_por_longitud(texto: str, maximo: int) -> list[str]:
    """Parte un bloque largo respetando límites de palabra.

    Una palabra más larga que el tope se corta igual: el OCR puede devolver
    cadenas sin espacios (bitácora 5.6).
    """
    partes, actual = [], ""
    for palabra in texto.split():
        while len(palabra) > maximo:
            if actual:
                partes.append(actual)
                actual = ""
            partes.append(palabra[:maximo])
            palabra = palabra[maximo:]
        if actual and len(actual) + 1 + len(palabra) > maximo:
            partes.append(actual)
            actual = palabra
        else:
            actual = f"{actual} {palabra}".strip()
    if actual:
        partes.append(actual)
    return partes


def _fusionar_cortos(fragmentos: list[str], minimo: int, maximo: int) -> list[str]:
    """Pega los fragmentos demasiado cortos al siguiente (el último, al anterior)."""
    resultado: list[str] = []
    pendiente = ""

    for fragmento in fragmentos:
        candidato = f"{pendiente}\n{fragmento}".strip() if pendiente else fragmento
        if len(candidato) < minimo:
            pendiente = candidato
            continue
        if len(candidato) <= maximo:
            resultado.append(candidato)
            pendiente = ""
        else:
            if pendiente:
                resultado.append(pendiente)
            resultado.append(fragmento)
            pendiente = ""

    if pendiente:
        if resultado:
            ultimo = f"{resultado[-1]}\n{pendiente}"
            if len(ultimo) <= maximo:
                resultado[-1] = ultimo
            else:
                resultado.append(pendiente)
        else:
            resultado.append(pendiente)

    return resultado


def _trocear(texto: str, maximo: int = MAXIMO_CARACTERES_FRAGMENTO) -> list[str]:
    """Trocea en fragmentos de a lo sumo `maximo` caracteres: por párrafos,
    después por líneas y como último recurso por longitud."""
    fragmentos = []
    for parrafo in texto.split("\n\n"):
        parrafo = parrafo.strip()
        if not parrafo:
            continue
        if len(parrafo) <= maximo:
            fragmentos.append(parrafo)
            continue

        acumulado = ""
        for linea in parrafo.split("\n"):
            linea = linea.strip()
            if not linea:
                continue
            if len(acumulado) + len(linea) + 1 <= maximo:
                acumulado = f"{acumulado}\n{linea}".strip()
            else:
                if acumulado:
                    fragmentos.append(acumulado)
                acumulado = linea if len(linea) <= maximo else ""
                if not acumulado:
                    fragmentos.extend(_partir_por_longitud(linea, maximo))
        if acumulado:
            fragmentos.append(acumulado)

    return _fusionar_cortos(fragmentos, MINIMO_CARACTERES_FRAGMENTO, maximo)


def _leer_archivo(archivo: Path, textos_ocr: dict[str, str] | None = None) -> str | None:
    extension = archivo.suffix.lower()
    if extension in EXTENSIONES_TEXTO:
        return archivo.read_text(encoding="utf-8")
    if extension in EXTENSIONES_IMAGEN:
        if textos_ocr is not None:
            if archivo.name not in textos_ocr:
                # No se llama al OCR a escondidas: el índice sale entero del archivo dado.
                raise ValueError(f"El OCR guardado no tiene texto para {archivo.name}")
            return textos_ocr[archivo.name]
        print(f"  OCR: {archivo.name}...")
        return extraer_texto_de_imagen(archivo)
    return None


def cargar_ocr_guardado(ruta: Path) -> dict[str, str]:
    """Texto por imagen de un archivo de pruebas/evaluar_ocr.py. Falla si
    alguna imagen quedó con error."""
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    con_error = [n for n, r in datos["imagenes"].items() if r["error"]]
    if con_error:
        raise ValueError(f"El OCR guardado tiene imágenes con error: {con_error}. Completalo con --reanudar.")
    print(f"Usando OCR guardado de {datos['modelo_ocr']} ({ruta})")
    return {nombre: r["texto"] for nombre, r in datos["imagenes"].items()}


def cargar_fragmentos(textos_ocr: dict[str, str] | None = None) -> tuple[list[tuple[str, str, str, int]], list[dict]]:
    """Devuelve (fragmentos, rechazos).

    Fragmento: (id, texto, archivo_fuente, orden). `orden` es la posición en su
    archivo; la usa el RAG agéntico para expandir a la receta completa.
    """
    fragmentos: list[tuple[str, str, str, int]] = []
    rechazos: list[dict] = []

    for archivo in sorted(RECETAS_DATA_DIR.iterdir()):
        if not archivo.is_file():
            continue
        texto = _leer_archivo(archivo, textos_ocr)
        if texto is None:
            continue

        # Se juzga primero el archivo entero: un bucle del OCR arruina toda la
        # transcripción, y así se reporta un rechazo en vez de decenas.
        diagnostico = evaluar_texto(texto)
        if diagnostico.es_degenerado:
            print(f"    RECHAZADO: {diagnostico.motivo}")
            rechazos.append({
                "archivo": archivo.name,
                "motivo": diagnostico.motivo,
                "caracteres": len(texto),
                "ratio_palabras_unicas": diagnostico.ratio_palabras_unicas,
            })
            continue

        for i, trozo in enumerate(_trocear(texto)):
            # Un archivo sano en conjunto puede traer un tramo degenerado.
            diagnostico_trozo = evaluar_texto(trozo)
            if diagnostico_trozo.es_degenerado:
                rechazos.append({
                    "archivo": f"{archivo.name} (fragmento {i})",
                    "motivo": diagnostico_trozo.motivo,
                    "caracteres": len(trozo),
                    "ratio_palabras_unicas": diagnostico_trozo.ratio_palabras_unicas,
                })
                continue
            fragmentos.append((f"{archivo.stem}-{i}", trozo, archivo.name, i))

    return fragmentos, rechazos


def ingestar(textos_ocr: dict[str, str] | None = None):
    fragmentos, rechazos = cargar_fragmentos(textos_ocr)

    if not fragmentos:
        print(f"No se pudo indexar nada de {RECETAS_DATA_DIR}")
        if rechazos:
            print("Todo lo leído fue rechazado por el control de calidad.")
        return

    ids = [f[0] for f in fragmentos]
    textos = [f[1] for f in fragmentos]
    fuentes = [f[2] for f in fragmentos]
    ordenes = [f[3] for f in fragmentos]

    print(f"\nGenerando embeddings para {len(textos)} fragmento(s)...")
    embeddings = embed_textos(textos)

    # Se recrea en cada ingesta para no dejar fragmentos huérfanos (bitácora 5.5).
    cliente = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        cliente.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass  # no existía todavía
    coleccion = cliente.create_collection(CHROMA_COLLECTION)
    coleccion.add(
        ids=ids,
        embeddings=embeddings,
        documents=textos,
        metadatas=[{"fuente": f, "orden": o} for f, o in zip(fuentes, ordenes)],
    )

    print(f"\nListo: {len(fragmentos)} fragmento(s) de {len(set(fuentes))} archivo(s) en {CHROMA_DIR}")

    if rechazos:
        print(f"\nRechazados por control de calidad ({len(rechazos)}):")
        for r in rechazos:
            print(f"  - {r['archivo']} ({r['caracteres']} caracteres)")
            print(f"      {r['motivo']}")
        print(
            "\n  Estos archivos NO están en el índice. Suele indicar que el OCR "
            "\n  no pudo leer la imagen y generó texto inventado — revisá la foto: "
            "\n  las páginas dobles muy densas y el texto en varias columnas son "
            "\n  las que más fallan."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta del recetario a ChromaDB.")
    parser.add_argument("--desde-ocr", type=Path, help="archivo de pruebas/evaluar_ocr.py a usar en vez de llamar al OCR")
    args = parser.parse_args()
    ingestar(cargar_ocr_guardado(args.desde_ocr) if args.desde_ocr else None)

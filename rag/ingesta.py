# Ingesta del recetario: lee rag/recetas_data/ (texto e imágenes), pasa las
# imágenes por OCR (glm-ocr), trocea el texto resultante, lo embebe (BGE-M3)
# y lo guarda en una colección local de ChromaDB persistida en rag/chroma_db/.
#
# Uso:
#   python rag/ingesta.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb

from rag.config import CHROMA_COLLECTION, CHROMA_DIR, RECETAS_DATA_DIR
from rag.embeddings import embed_textos
from rag.ocr import extraer_texto_de_imagen

EXTENSIONES_TEXTO = {".txt", ".md"}
EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png"}


def _trocear(texto: str) -> list[str]:
    # Las recetas mock son cortas: un chunk por párrafo es suficiente y
    # mantiene cada fragmento centrado en un paso o sección de la receta.
    return [p.strip() for p in texto.split("\n\n") if p.strip()]


def _leer_archivo(archivo: Path) -> str | None:
    extension = archivo.suffix.lower()
    if extension in EXTENSIONES_TEXTO:
        return archivo.read_text(encoding="utf-8")
    if extension in EXTENSIONES_IMAGEN:
        print(f"  OCR: {archivo.name} (requiere VPN institucional)...")
        return extraer_texto_de_imagen(archivo)
    return None


def cargar_fragmentos() -> list[tuple[str, str, str]]:
    """Devuelve tuplas (id_fragmento, texto, archivo_fuente)."""
    fragmentos = []
    for archivo in sorted(RECETAS_DATA_DIR.iterdir()):
        if not archivo.is_file():
            continue
        texto = _leer_archivo(archivo)
        if texto is None:
            continue
        for i, parrafo in enumerate(_trocear(texto)):
            fragmentos.append((f"{archivo.stem}-{i}", parrafo, archivo.name))
    return fragmentos


def ingestar():
    fragmentos = cargar_fragmentos()
    if not fragmentos:
        print(f"No se encontraron recetas en {RECETAS_DATA_DIR}")
        return

    ids = [f[0] for f in fragmentos]
    textos = [f[1] for f in fragmentos]
    fuentes = [f[2] for f in fragmentos]

    print(f"Generando embeddings para {len(textos)} fragmento(s) (requiere VPN institucional)...")
    embeddings = embed_textos(textos)

    cliente = chromadb.PersistentClient(path=str(CHROMA_DIR))
    coleccion = cliente.get_or_create_collection(CHROMA_COLLECTION)
    coleccion.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=textos,
        metadatas=[{"fuente": f} for f in fuentes],
    )

    print(f"Listo: {len(fragmentos)} fragmento(s) de {len(set(fuentes))} archivo(s) en {CHROMA_DIR}")


if __name__ == "__main__":
    ingestar()

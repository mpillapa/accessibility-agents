# Genera una imagen sintética a partir de texto para poder probar el pipeline
# de OCR (glm-ocr) sin depender de una foto real de una receta manuscrita
# (que no tenemos). NO es una receta manuscrita real: es texto tipeado y
# renderizado como imagen, solo para ejercitar extremo a extremo la ruta
# imagen -> OCR -> chunk -> embedding -> ChromaDB.
#
# Cuando exista una foto real de una receta manuscrita, reemplaza el archivo
# generado aquí por esa foto (mismo nombre o cualquier .jpg/.png dentro de
# esta carpeta) y vuelve a correr rag/ingesta.py.
#
# Uso:
#   python rag/recetas_data/generar_imagen_mock.py

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TEXTO = """Receta: Torta de platano (de la abuela)

3 platanos maduros bien machacados
2 huevos
1 taza de azucar
1/2 taza de aceite
2 tazas de harina
1 cucharadita de polvo de hornear

Mezclar los platanos con los huevos, el azucar y el aceite.
Agregar la harina y el polvo de hornear poco a poco.
Hornear 40 minutos a 180 grados."""


def generar():
    ancho, alto = 800, 600
    imagen = Image.new("RGB", (ancho, alto), color="white")
    dibujo = ImageDraw.Draw(imagen)
    try:
        fuente = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        fuente = ImageFont.load_default()

    dibujo.multiline_text((30, 30), TEXTO, fill="black", font=fuente, spacing=10)

    destino = Path(__file__).parent / "receta_manuscrita_simulada.png"
    imagen.save(destino)
    print(f"Imagen generada en {destino}")


if __name__ == "__main__":
    generar()

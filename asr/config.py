# Configuración del reconocimiento de voz (ASR) con Whisper.
#
# A diferencia del resto del stack (chat, embeddings, OCR), Whisper NO corre en
# un endpoint remoto: corre LOCAL, en la GPU de esta máquina. El servidor de la
# Universidad no expone ningún endpoint de ASR — se escaneó el rango
# 12550-12575 el 2026-08-19 y solo hay chat (12559), embeddings (12556),
# OCR (12560) y un DeepSeek (12555).
#
# Eso rompe la consistencia con el resto del proyecto, que está estandarizado
# sobre vLLM OpenAI-compatible por pedido de los tutores. Es una desviación
# consciente y coincide con lo que pidió Cristian ("te bajas el whisper"), pero
# conviene confirmarla con ellos.

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# Tamaño del modelo. En este equipo (8x H200) 'large-v3' entra sin problema y es
# el más preciso; en una laptop habría que bajar a 'small' o 'base'.
# Opciones: tiny, base, small, medium, large-v2, large-v3
WHISPER_MODELO = os.getenv("WHISPER_MODELO", "large-v3")

# 'cuda' o 'cpu'. Con cuda, compute_type float16; en CPU conviene int8.
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

# Idioma fijo. Dejarlo en None haría que Whisper lo detecte, pero detectar
# idioma sobre audio ruidoso es una fuente extra de error, y acá se sabe que
# el usuario habla español.
WHISPER_IDIOMA = os.getenv("WHISPER_IDIOMA", "es")

# Dónde se guardan los audios grabados y sus etiquetas.
CORPUS_DIR = Path(__file__).parent.parent / "corpus_audio"
CORPUS_GRABACIONES_DIR = CORPUS_DIR / "grabaciones"
CORPUS_INDICE = CORPUS_DIR / "indice.csv"

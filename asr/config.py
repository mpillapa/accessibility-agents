# Configuración de Whisper. Corre local en GPU: no hay endpoint de ASR en el
# servidor (asr/README.md).

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# tiny, base, small, medium, large-v2, large-v3. En una laptop, 'small' o 'base'.
WHISPER_MODELO = os.getenv("WHISPER_MODELO", "large-v3")

# En CPU conviene compute_type int8.
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

# Fijo: detectar idioma sobre audio ruidoso agrega error.
WHISPER_IDIOMA = os.getenv("WHISPER_IDIOMA", "es")

CORPUS_DIR = Path(__file__).parent.parent / "corpus_audio"
CORPUS_GRABACIONES_DIR = CORPUS_DIR / "grabaciones"
CORPUS_INDICE = CORPUS_DIR / "indice.csv"

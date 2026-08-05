# Configuración compartida del RAG de recetas: embeddings (BGE-M3) y OCR
# (glm-ocr), ambos servidos como endpoints OpenAI-compatible por vLLM en la
# Universidad (ver PDF "Acceso a los Endpoints de LLMs"). Requiere VPN
# institucional activa — sin ella estas llamadas fallan por timeout/conexión.

import os
from pathlib import Path

from dotenv import load_dotenv

# override=True: el .env del proyecto manda sobre variables ya presentes en el
# entorno (p. ej. las que VSCode inyecta desde un .env del workspace padre).
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

VLLM_API_KEY = os.getenv("VLLM_API_KEY", "local")

VLLM_EMBEDDINGS_BASE_URL = os.getenv("VLLM_EMBEDDINGS_BASE_URL", "http://172.28.230.10:12556/v1")
VLLM_EMBEDDINGS_MODEL = os.getenv("VLLM_EMBEDDINGS_MODEL", "BAAI/bge-m3")

VLLM_OCR_BASE_URL = os.getenv("VLLM_OCR_BASE_URL", "http://172.28.230.10:12560/v1")
VLLM_OCR_MODEL = os.getenv("VLLM_OCR_MODEL", "zai-org/GLM-OCR")

RECETAS_DATA_DIR = Path(__file__).parent / "recetas_data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
CHROMA_COLLECTION = "recetas"

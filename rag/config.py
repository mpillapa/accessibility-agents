# Configuración del RAG de recetas: endpoints de embeddings y OCR, rutas de
# ChromaDB. Las llamadas a los endpoints requieren VPN.

import os
from pathlib import Path

from dotenv import load_dotenv

from infraestructura.modelos import resolver_modelo

# override=True: el .env del proyecto manda sobre el que inyecta VSCode.
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

VLLM_API_KEY = os.getenv("VLLM_API_KEY", "local")

# Ids resueltos contra el endpoint (infraestructura/modelos.py). Cambiar el modelo
# de embeddings invalida el índice de Chroma ya construido.
VLLM_EMBEDDINGS_BASE_URL = os.getenv("VLLM_EMBEDDINGS_BASE_URL", "http://172.28.230.10:12556/v1")
VLLM_EMBEDDINGS_MODEL = resolver_modelo(
    base_url=VLLM_EMBEDDINGS_BASE_URL,
    modelo_preferido=os.getenv("VLLM_EMBEDDINGS_MODEL", "BAAI/bge-m3"),
    api_key=VLLM_API_KEY,
)

VLLM_OCR_BASE_URL = os.getenv("VLLM_OCR_BASE_URL", "http://172.28.230.10:12560/v1")
VLLM_OCR_MODEL = resolver_modelo(
    base_url=VLLM_OCR_BASE_URL,
    modelo_preferido=os.getenv("VLLM_OCR_MODEL", "zai-org/GLM-OCR"),
    api_key=VLLM_API_KEY,
)

RECETAS_DATA_DIR = Path(__file__).parent / "recetas_data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
CHROMA_COLLECTION = "recetas"

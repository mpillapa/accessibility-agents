# Configuración compartida del RAG de recetas: embeddings (BGE-M3) y OCR
# (glm-ocr), ambos servidos como endpoints OpenAI-compatible por vLLM en la
# Universidad (ver PDF "Acceso a los Endpoints de LLMs"). Requiere VPN
# institucional activa — sin ella estas llamadas fallan por timeout/conexión.

import os
from pathlib import Path

from dotenv import load_dotenv

from infraestructura.modelos import resolver_modelo

# override=True: el .env del proyecto manda sobre variables ya presentes en el
# entorno (p. ej. las que VSCode inyecta desde un .env del workspace padre).
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

VLLM_API_KEY = os.getenv("VLLM_API_KEY", "local")

# Los id de modelo se resuelven contra el endpoint en vez de fijarse a mano:
# los servidores de la Universidad rotan de modelo sin aviso (ver
# infraestructura/modelos.py). El valor del .env queda como preferencia.
#
# OJO con embeddings: cambiar de modelo cambia la dimensión y el espacio
# vectorial, así que un cambio aquí invalida el índice de Chroma ya construido.
# Por eso resolver_modelo() NO sustituye automáticamente cuando el endpoint
# sirve varios modelos (caso Ollama): avisa y deja que falle.
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

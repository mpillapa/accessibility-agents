# Embeddings vía BGE-M3, servido como endpoint OpenAI-compatible por vLLM.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from rag.config import VLLM_API_KEY, VLLM_EMBEDDINGS_BASE_URL, VLLM_EMBEDDINGS_MODEL

_cliente = OpenAI(base_url=VLLM_EMBEDDINGS_BASE_URL, api_key=VLLM_API_KEY)


def embed_textos(textos: list[str]) -> list[list[float]]:
    """Devuelve un embedding por texto de entrada, en el mismo orden. Requiere
    VPN institucional activa (servidor BGE-M3 en 172.28.230.10:12556)."""
    respuesta = _cliente.embeddings.create(model=VLLM_EMBEDDINGS_MODEL, input=textos)
    return [dato.embedding for dato in respuesta.data]

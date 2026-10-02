# Cliente LLM compartido por los nodos. Separado de agentes.py para evitar un
# import circular con el subgrafo de RAG.
# Servidor vLLM de la Universidad (OpenAI-compatible); requiere VPN.

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from infraestructura.modelos import resolver_modelo
from infraestructura.trazas import configurar_trazas

# override=True: el .env del proyecto manda sobre el que inyecta VSCode del workspace padre.
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

VLLM_CHAT_BASE_URL = os.getenv("VLLM_CHAT_BASE_URL", "http://172.28.230.10:12559/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "local")

# El .env es una preferencia: el servidor rota de modelo sin aviso
# (infraestructura/modelos.py).
VLLM_CHAT_MODEL = resolver_modelo(
    base_url=VLLM_CHAT_BASE_URL,
    modelo_preferido=os.getenv("VLLM_CHAT_MODEL", "google/gemma-4-12B-it"),
    api_key=VLLM_API_KEY,
)

# LangSmith opcional: sin API key el sistema funciona igual, sin trazas.
configurar_trazas()

llm = ChatOpenAI(
    model=VLLM_CHAT_MODEL,
    base_url=VLLM_CHAT_BASE_URL,
    api_key=VLLM_API_KEY,
    temperature=0.3,  # bajo para que el ruteo y los veredictos sean consistentes
)

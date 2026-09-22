# Cliente LLM compartido por todos los nodos de LangGraph.
#
# Vive en su propio módulo (y no dentro de agentes.py) para que el subgrafo de
# RAG agéntico pueda usarlo sin importar agentes.py, que a su vez importa el
# subgrafo — sin esto habría un import circular.
#
# El modelo corre en el servidor vLLM de la Universidad con API
# OpenAI-compatible. Requiere VPN institucional activa.

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from infraestructura.modelos import resolver_modelo

# override=True: el .env del proyecto manda sobre variables ya presentes en el
# entorno (p. ej. las que VSCode inyecta desde un .env del workspace padre).
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

VLLM_CHAT_BASE_URL = os.getenv("VLLM_CHAT_BASE_URL", "http://172.28.230.10:12559/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "local")

# El .env expresa una preferencia, no un dato fijo: el servidor de la
# Universidad rota de modelo sin aviso y el id escrito a mano caduca. Ver
# infraestructura/modelos.py para el registro de rotaciones observadas.
VLLM_CHAT_MODEL = resolver_modelo(
    base_url=VLLM_CHAT_BASE_URL,
    modelo_preferido=os.getenv("VLLM_CHAT_MODEL", "google/gemma-4-12B-it"),
    api_key=VLLM_API_KEY,
)

llm = ChatOpenAI(
    model=VLLM_CHAT_MODEL,
    base_url=VLLM_CHAT_BASE_URL,
    api_key=VLLM_API_KEY,
    temperature=0.3,  # bajo para que el ruteo y los veredictos sean consistentes
)

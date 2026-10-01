# OCR/visión vía un modelo multimodal servido como endpoint OpenAI-compatible.
# Se usa chat completions con contenido de imagen (formato "vision" estilo
# OpenAI), no un endpoint de OCR dedicado.
#
# Modelos: zai-org/GLM-OCR en vLLM hasta el 2026-09-09 (generó el índice de
# agosto); desde entonces qwen2.5vl:7b en Ollama, acordado con el tutor el
# 2026-09-30. El modelo sale del .env (VLLM_OCR_MODEL).

import base64
from pathlib import Path

from openai import OpenAI

from rag.config import VLLM_API_KEY, VLLM_OCR_BASE_URL, VLLM_OCR_MODEL

# Tope de salida por imagen. Sin tope, una página que hace entrar al modelo en
# bucle lo deja generando hasta llenar su contexto: el 2026-09-30, qwen2.5vl:7b
# pasó más de 25 minutos con "2 recetas mas.jpg", la misma página que llevó a
# GLM-OCR a repetir una frase 888 veces (bitácora 2.3). La receta más larga del
# recetario tiene ~4.000 caracteres (~1.500 tokens): 4.096 no corta texto
# legítimo, y un bucle se corta, sale con finish_reason="length" y lo rechaza
# rag/calidad.py en vez de colgar la ingesta.
MAXIMO_TOKENS_SALIDA = 4096

# Segundos por imagen. Con el tope de arriba una imagen normal tarda ~20 s.
TIMEOUT_SEGUNDOS = 300

_cliente = OpenAI(
    base_url=VLLM_OCR_BASE_URL,
    api_key=VLLM_API_KEY,
    timeout=TIMEOUT_SEGUNDOS,
    max_retries=0,  # reintentar una imagen que se colgó solo duplica la espera
)

_PROMPT_TRANSCRIPCION = (
    "Transcribe TODO el texto visible en esta imagen tal como está escrito, "
    "sin resumir, sin traducir y sin agregar comentarios. Si es una receta de "
    "cocina manuscrita, conserva ingredientes y cantidades exactamente como "
    "aparecen."
)


def extraer_texto_de_imagen(ruta_imagen: Path) -> str:
    """Texto reconocido en la imagen. Ver extraer_texto_detallado()."""
    return extraer_texto_detallado(ruta_imagen)["texto"]


def extraer_texto_detallado(ruta_imagen: Path) -> dict:
    """Envía una imagen (ej. foto de una receta manuscrita) al endpoint
    OCR/visión y devuelve el texto reconocido. Requiere acceso a la red de la
    universidad. Para medirlo sin texto de referencia: pruebas/evaluar_ocr.py.

    Devuelve {"texto", "fin", "tokens_salida"}. `fin` == "length" significa que
    el modelo agotó MAXIMO_TOKENS_SALIDA: casi siempre un bucle."""
    ruta_imagen = Path(ruta_imagen)
    datos = ruta_imagen.read_bytes()
    b64 = base64.b64encode(datos).decode("ascii")
    extension = ruta_imagen.suffix.lstrip(".").lower() or "png"

    respuesta = _cliente.chat.completions.create(
        model=VLLM_OCR_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT_TRANSCRIPCION},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{extension};base64,{b64}"},
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=MAXIMO_TOKENS_SALIDA,
    )
    eleccion = respuesta.choices[0]
    return {
        "texto": eleccion.message.content or "",
        "fin": eleccion.finish_reason,
        "tokens_salida": respuesta.usage.completion_tokens if respuesta.usage else None,
    }

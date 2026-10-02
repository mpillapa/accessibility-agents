# OCR de imágenes con un modelo multimodal vía chat completions (OpenAI-compatible).
# El modelo sale del .env; hoy qwen2.5vl:7b, antes GLM-OCR (README.md raíz).

import base64
from pathlib import Path

from openai import OpenAI

from rag.config import VLLM_API_KEY, VLLM_OCR_BASE_URL, VLLM_OCR_MODEL

# Corta los bucles del modelo (bitácora 2.3) sin cortar la receta más larga
# (~1.500 tokens); un bucle sale con finish_reason="length" y lo rechaza calidad.py.
MAXIMO_TOKENS_SALIDA = 4096

# Segundos por imagen; una normal tarda ~20 s.
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
    """Devuelve {"texto", "fin", "tokens_salida"} de una imagen. Requiere VPN.

    `fin` == "length" significa que agotó MAXIMO_TOKENS_SALIDA: casi siempre un bucle."""
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

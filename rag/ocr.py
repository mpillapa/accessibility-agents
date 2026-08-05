# OCR/visión vía glm-ocr, servido como endpoint OpenAI-compatible por vLLM.
# Se usa chat completions con contenido de imagen (formato "vision" estilo
# OpenAI), no un endpoint de OCR dedicado — es lo que expone vLLM para este
# contenedor según el PDF de la Universidad.

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from rag.config import VLLM_API_KEY, VLLM_OCR_BASE_URL, VLLM_OCR_MODEL

_cliente = OpenAI(base_url=VLLM_OCR_BASE_URL, api_key=VLLM_API_KEY)

_PROMPT_TRANSCRIPCION = (
    "Transcribe TODO el texto visible en esta imagen tal como está escrito, "
    "sin resumir, sin traducir y sin agregar comentarios. Si es una receta de "
    "cocina manuscrita, conserva ingredientes y cantidades exactamente como "
    "aparecen."
)


def extraer_texto_de_imagen(ruta_imagen: Path) -> str:
    """Envía una imagen (ej. foto de una receta manuscrita) al endpoint
    OCR/visión y devuelve el texto reconocido. Requiere VPN institucional
    activa (servidor GLM-OCR en 172.28.230.10:12560). Verificado el 2026-07-23:
    transcribe correctamente la imagen mock del recetario."""
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
    )
    return respuesta.choices[0].message.content

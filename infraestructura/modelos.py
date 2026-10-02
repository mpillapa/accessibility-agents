# Resuelve el id de modelo que sirve un endpoint vLLM/Ollama (GET /v1/models) en
# vez de fijarlo en el .env, porque el servidor rota de modelo sin aviso.
# Registro de rotaciones: README.md raíz, "Rotación de modelos en el servidor".

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# override=True: el .env del proyecto manda sobre el que inyecta VSCode.
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# Corto: sin VPN no responde y no debe colgar el import; se cae al valor del .env.
TIMEOUT_CONSULTA_SEGUNDOS = 5.0

# Caché por base_url; también alimenta describir_resolucion().
_resoluciones: dict[str, "ResolucionModelo"] = {}


class ResolucionModelo:
    """Qué se pidió, qué se obtuvo y por qué."""

    def __init__(self, base_url: str, preferido: str, resuelto: str, motivo: str):
        self.base_url = base_url
        self.preferido = preferido
        self.resuelto = resuelto
        self.motivo = motivo

    @property
    def hubo_cambio(self) -> bool:
        return self.preferido != self.resuelto

    def __repr__(self) -> str:
        return f"ResolucionModelo({self.resuelto!r}, motivo={self.motivo!r})"


def _consultar_modelos_servidos(base_url: str, api_key: str) -> list[str]:
    """Ids de GET /v1/models, o lista vacía si no se pudo consultar."""
    cliente = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=TIMEOUT_CONSULTA_SEGUNDOS,
        max_retries=0,  # sin VPN reintentar solo multiplica la espera
    )
    try:
        return [modelo.id for modelo in cliente.models.list().data]
    except Exception:
        # Amplio a propósito: ningún fallo debe impedir importar el módulo.
        return []


def resolver_modelo(base_url: str, modelo_preferido: str, api_key: str) -> str:
    """Id de modelo que este endpoint acepta hoy.

    Respeta `modelo_preferido` (el .env) si el servidor lo sirve; si no, usa el
    único que haya y avisa. Cacheado por `base_url`.
    """
    if base_url in _resoluciones:
        return _resoluciones[base_url].resuelto

    servidos = _consultar_modelos_servidos(base_url, api_key)

    if not servidos:
        motivo = "no se pudo consultar el endpoint (¿VPN inactiva?); se usa el valor del .env"
        resuelto = modelo_preferido
    elif modelo_preferido in servidos:
        motivo = "el endpoint sigue sirviendo el modelo configurado en el .env"
        resuelto = modelo_preferido
    elif len(servidos) == 1:
        # Endpoint dedicado: el único modelo servido es el reemplazo.
        resuelto = servidos[0]
        motivo = f"el .env pide {modelo_preferido!r}; el endpoint ahora sirve solo {resuelto!r}"
        print(
            f"  AVISO [{base_url}]: el modelo del .env ({modelo_preferido}) ya no "
            f"está disponible. Se usará {resuelto}.\n"
            f"  Las mediciones tomadas ahora NO son comparables con las hechas "
            f"con {modelo_preferido}."
        )
    else:
        # Varios modelos (Ollama): no se adivina, un modelo equivocado de
        # embeddings corrompería el índice en silencio. Que falle con 404.
        resuelto = modelo_preferido
        motivo = (
            f"el endpoint sirve {len(servidos)} modelos y ninguno es {modelo_preferido!r}; "
            "no se sustituye porque la elección sería arbitraria"
        )
        print(
            f"  AVISO [{base_url}]: el modelo del .env ({modelo_preferido}) no está "
            f"entre los {len(servidos)} que sirve este endpoint.\n"
            f"  Disponibles: {', '.join(servidos)}\n"
            f"  No se sustituye automáticamente: elegí a mano cuál corresponde y "
            f"actualizá el .env."
        )

    _resoluciones[base_url] = ResolucionModelo(base_url, modelo_preferido, resuelto, motivo)
    return resuelto


def describir_resolucion() -> dict[str, dict[str, str]]:
    """Qué modelo se resolvió para cada endpoint; anotarlo junto a toda medición."""
    return {
        base_url: {
            "preferido": r.preferido,
            "resuelto": r.resuelto,
            "motivo": r.motivo,
            "hubo_cambio": str(r.hubo_cambio),
        }
        for base_url, r in _resoluciones.items()
    }

# Resolución del identificador de modelo servido por un endpoint vLLM/Ollama.
#
# POR QUÉ EXISTE ESTE MÓDULO
# --------------------------
# Los endpoints de la Universidad rotan de modelo sin aviso. Verificado tres
# veces en tres semanas sobre el MISMO puerto (12559):
#
#   hasta 2026-09-08   google/gemma-4-12B-it            (contenedor eliminado)
#   2026-09-09..20     zai-org/GLM-5.3-Flash            (contenedor eliminado)
#   desde 2026-09-21   canada-quant/GLM-5.3-Flash-W4A16-MTP
#
# La API OpenAI valida el campo `model` contra el identificador exacto que sirve
# el contenedor: cualquier otro valor devuelve 404 y el sistema entero cae. Con
# el id escrito a mano en el .env, cada rotación del servidor rompe el proyecto
# hasta que alguien lo edita a mano.
#
# La solución es preguntarle al servidor qué está sirviendo (GET /v1/models) en
# vez de asumirlo. El valor del .env pasa de ser un dato obligatorio a ser una
# preferencia: se respeta si el servidor todavía lo sirve, y si no, se usa lo
# que haya y se avisa por consola.
#
# LIMITACIÓN CONOCIDA: esto evita la caída por 404, pero NO hace que los
# resultados sean reproducibles. Un modelo distinto da salidas distintas. Por
# eso `describir_resolucion()` deja registro de qué se resolvió realmente: al
# reportar una medición hay que anotar con qué modelo se obtuvo.

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# override=True: el .env del proyecto manda sobre variables ya presentes en el
# entorno (p. ej. las que VSCode inyecta desde un .env del workspace padre).
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# Segundos de espera al consultar /v1/models. Corto a propósito: sin VPN la
# llamada no va a responder nunca y no queremos que el import quede colgado.
# Si se agota, se usa el modelo preferido del .env y el error aparece después,
# en la llamada real, que es donde el usuario lo entiende.
TIMEOUT_CONSULTA_SEGUNDOS = 5.0

# Qué se resolvió en esta ejecución, por base_url. Sirve para dos cosas:
# evitar una consulta de red por cada import, y poder anotar en las mediciones
# con qué modelo se corrió realmente (ver describir_resolucion()).
_resoluciones: dict[str, "ResolucionModelo"] = {}


class ResolucionModelo:
    """Resultado de resolver el modelo de un endpoint: qué se pidió, qué se
    obtuvo y por qué. Se guarda para poder documentar las mediciones."""

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
    """Devuelve los ids que el endpoint declara en GET /v1/models, o lista
    vacía si no se pudo consultar (sin VPN, servidor caído, timeout)."""
    cliente = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=TIMEOUT_CONSULTA_SEGUNDOS,
        max_retries=0,  # sin VPN reintentar solo multiplica la espera
    )
    try:
        return [modelo.id for modelo in cliente.models.list().data]
    except Exception:
        # Deliberadamente amplio: cualquier fallo de red, DNS, timeout o
        # respuesta mal formada debe degradar al valor del .env, nunca
        # impedir que el módulo se importe.
        return []


def resolver_modelo(base_url: str, modelo_preferido: str, api_key: str) -> str:
    """Devuelve el id de modelo que este endpoint acepta hoy.

    `modelo_preferido` es el valor del .env: se respeta si el servidor lo
    sigue sirviendo. Si el servidor cambió de modelo, se usa el que haya y se
    avisa por consola, porque el cambio afecta la comparabilidad de cualquier
    medición que se tome después.

    El resultado se cachea por `base_url`: una consulta de red por endpoint y
    por ejecución, no una por llamada al modelo.
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
        # Un endpoint vLLM dedicado sirve un solo modelo: si el del .env ya no
        # está, el que hay es necesariamente su reemplazo. Sustituir es seguro.
        resuelto = servidos[0]
        motivo = f"el .env pide {modelo_preferido!r}; el endpoint ahora sirve solo {resuelto!r}"
        print(
            f"  AVISO [{base_url}]: el modelo del .env ({modelo_preferido}) ya no "
            f"está disponible. Se usará {resuelto}.\n"
            f"  Las mediciones tomadas ahora NO son comparables con las hechas "
            f"con {modelo_preferido}."
        )
    else:
        # El endpoint sirve varios modelos (caso Ollama: 14 modelos de chat,
        # visión y embeddings en el mismo puerto) y ninguno es el pedido. NO se
        # puede adivinar cuál corresponde: sustituir un modelo de embeddings por
        # uno de chat produciría vectores de otra dimensión y corrompería el
        # índice en silencio, que es peor que fallar. Se mantiene el preferido
        # para que la llamada real falle con un 404 explícito.
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
    """Qué modelo se resolvió para cada endpoint en esta ejecución.

    Pensado para anotarlo junto a cualquier número que se reporte (latencias,
    WER, accuracy de ruteo). Con el servidor rotando modelos, una medición sin
    esta información no se puede interpretar después.
    """
    return {
        base_url: {
            "preferido": r.preferido,
            "resuelto": r.resuelto,
            "motivo": r.motivo,
            "hubo_cambio": str(r.hubo_cambio),
        }
        for base_url, r in _resoluciones.items()
    }

# Trazabilidad de la ejecución del grafo con LangSmith.
#
# QUÉ RESUELVE
# ------------
# Hasta ahora, ver por dónde pasaba una consulta exigía leer la salida de
# `stream()` en la terminal o desplegar el recorrido en la interfaz. Eso sirve
# para depurar una consulta puntual, pero no para responder preguntas sobre el
# comportamiento acumulado: cuántas veces el RAG tuvo que reformular, qué
# consultas terminaron en `sin_resultado`, cuánto tardó cada nodo en promedio.
#
# LangSmith registra cada ejecución del grafo —nodo por nodo, con entradas,
# salidas, latencia y tokens— en una interfaz donde eso se puede consultar
# después. Sirve para depurar ahora y para auditar el sistema más adelante, que
# es la razón por la que lo pidió el tutor.
#
# CÓMO SE ACTIVA
# --------------
# LangChain y LangGraph ya están instrumentados: basta con que existan las
# variables de entorno correctas. Este módulo no engancha nada a mano, solo
# valida la configuración y deja constancia de si quedó activa o no.
#
# Si no hay API key, el sistema funciona igual, sin trazas. Es deliberado: las
# pruebas y el uso normal no deben depender de un servicio externo.
#
# QUÉ SALE DE LA RED AL ACTIVARLO
# -------------------------------
# LangSmith es un servicio en la nube de LangChain: las trazas incluyen los
# prompts y las respuestas COMPLETAS, y se envían a sus servidores. Para este
# prototipo eso son consultas sobre recetas y no hay problema.
#
# Vale tenerlo presente antes de construir el agente de medicación: si por ahí
# llegara a pasar información de salud real de una persona, enviarla a un
# tercero deja de ser irrelevante y hay que decidirlo a propósito, no por
# omisión. Con datos ficticios, como está planteado, no aplica.

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# Nombre del proyecto en LangSmith. Agrupa las ejecuciones; si no existe, se
# crea solo la primera vez que llega una traza.
PROYECTO_POR_DEFECTO = "accessibility-agents"

_estado = {"activo": False, "motivo": "todavía no se configuró"}


def configurar_trazas() -> bool:
    """Deja el entorno listo para que LangChain envíe trazas a LangSmith.

    Devuelve True si quedaron activas. No lanza excepción si falta la clave:
    sin trazas el sistema funciona igual, y hacer que la ausencia de un servicio
    externo rompa las pruebas sería peor que no tenerlo.
    """
    clave = os.getenv("LANGSMITH_API_KEY", "").strip()
    quiere_trazar = os.getenv("LANGSMITH_TRACING", "").strip().lower() in ("true", "1", "yes")

    if not quiere_trazar:
        _estado.update(activo=False, motivo="LANGSMITH_TRACING no está en 'true'")
        os.environ["LANGSMITH_TRACING"] = "false"
        return False

    if not clave:
        _estado.update(
            activo=False,
            motivo="LANGSMITH_TRACING=true pero falta LANGSMITH_API_KEY en el .env",
        )
        os.environ["LANGSMITH_TRACING"] = "false"
        print(
            "  AVISO: se pidió trazar con LangSmith pero no hay LANGSMITH_API_KEY.\n"
            "  El sistema sigue funcionando, sin trazas."
        )
        return False

    # LangChain lee estas variables en cada llamada; alcanza con dejarlas puestas.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", PROYECTO_POR_DEFECTO)

    _estado.update(
        activo=True,
        motivo=f"trazando al proyecto {os.environ['LANGSMITH_PROJECT']!r}",
    )
    return True


def describir_trazas() -> dict[str, str]:
    """Si las trazas quedaron activas y por qué. Para mostrarlo en la interfaz
    y para dejar constancia junto a las mediciones."""
    return {
        "activo": str(_estado["activo"]),
        "motivo": _estado["motivo"],
        "proyecto": os.getenv("LANGSMITH_PROJECT", PROYECTO_POR_DEFECTO),
    }

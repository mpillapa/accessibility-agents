# Trazas del grafo en LangSmith. LangGraph ya está instrumentado: este módulo
# solo valida las variables de entorno. Sin API key el sistema corre sin trazas.
#
# Las trazas llevan prompts y respuestas completos a la nube de LangChain: no
# usar con datos de salud reales (bitácora 12.4).

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# LangSmith crea el proyecto con la primera traza.
PROYECTO_POR_DEFECTO = "accessibility-agents"

_estado = {"activo": False, "motivo": "todavía no se configuró"}


def configurar_trazas() -> bool:
    """Activa las trazas si el .env lo pide. True si quedaron activas; no falla
    si falta la clave."""
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
    """Si las trazas quedaron activas y por qué."""
    return {
        "activo": str(_estado["activo"]),
        "motivo": _estado["motivo"],
        "proyecto": os.getenv("LANGSMITH_PROJECT", PROYECTO_POR_DEFECTO),
    }

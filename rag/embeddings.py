# Embeddings BGE-M3 vía endpoint OpenAI-compatible, en lotes. Requiere VPN.

from openai import OpenAI

from rag.config import VLLM_API_KEY, VLLM_EMBEDDINGS_BASE_URL, VLLM_EMBEDDINGS_MODEL

_cliente = OpenAI(base_url=VLLM_EMBEDDINGS_BASE_URL, api_key=VLLM_API_KEY)

# El servidor suma los tokens de todos los textos de una petición (bitácora 5.3).
LIMITE_TOKENS_MODELO = 8192

# Margen bajo el límite porque los tokens se estiman por caracteres.
PRESUPUESTO_TOKENS_POR_LOTE = 6000

# Conservador: subestimar tokens provoca el error 400.
CARACTERES_POR_TOKEN = 3.0


def _estimar_tokens(texto: str) -> int:
    return max(1, int(len(texto) / CARACTERES_POR_TOKEN))


def _truncar_si_excede(texto: str) -> str:
    """Recorta un texto que no cabe en la ventana del modelo, en vez de hacer
    fallar la ingesta. Red de seguridad: el troceado de ingesta.py ya lo evita."""
    maximo_caracteres = int(LIMITE_TOKENS_MODELO * CARACTERES_POR_TOKEN * 0.9)
    if len(texto) <= maximo_caracteres:
        return texto
    print(
        f"  AVISO: un fragmento de {len(texto)} caracteres excede la ventana del "
        f"modelo; se truncó a {maximo_caracteres}. Revisa el troceado en rag/ingesta.py."
    )
    return texto[:maximo_caracteres]


def _agrupar_en_lotes(textos: list[str]) -> list[list[str]]:
    lotes: list[list[str]] = []
    lote_actual: list[str] = []
    tokens_actuales = 0

    for texto in textos:
        tokens = _estimar_tokens(texto)
        if lote_actual and tokens_actuales + tokens > PRESUPUESTO_TOKENS_POR_LOTE:
            lotes.append(lote_actual)
            lote_actual = []
            tokens_actuales = 0
        lote_actual.append(texto)
        tokens_actuales += tokens

    if lote_actual:
        lotes.append(lote_actual)
    return lotes


def embed_textos(textos: list[str]) -> list[list[float]]:
    """Un embedding por texto, en el mismo orden."""
    if not textos:
        return []

    preparados = [_truncar_si_excede(t) for t in textos]
    lotes = _agrupar_en_lotes(preparados)

    if len(lotes) > 1:
        print(f"  Embeddings en {len(lotes)} lote(s) para no exceder la ventana del modelo.")

    embeddings: list[list[float]] = []
    for i, lote in enumerate(lotes, 1):
        if len(lotes) > 1:
            print(f"    lote {i}/{len(lotes)} ({len(lote)} fragmento(s))...")
        respuesta = _cliente.embeddings.create(model=VLLM_EMBEDDINGS_MODEL, input=lote)
        # Reordenar por índice: desordenados, cada texto quedaría con el vector de otro.
        for dato in sorted(respuesta.data, key=lambda d: d.index):
            embeddings.append(dato.embedding)

    return embeddings

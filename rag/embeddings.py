# Embeddings vía BGE-M3, servido como endpoint OpenAI-compatible por vLLM.

from openai import OpenAI

from rag.config import VLLM_API_KEY, VLLM_EMBEDDINGS_BASE_URL, VLLM_EMBEDDINGS_MODEL

_cliente = OpenAI(base_url=VLLM_EMBEDDINGS_BASE_URL, api_key=VLLM_API_KEY)

# Límite de contexto de bge-m3 tal como está desplegado en el servidor. El
# servidor cuenta los tokens de TODOS los textos de una misma petición juntos,
# no por texto: mandar 223 fragmentos en una sola llamada devuelve
# "maximum context length is 8192 tokens" aunque cada fragmento sea corto.
LIMITE_TOKENS_MODELO = 8192

# Presupuesto por petición, con margen bajo el límite real porque los tokens se
# estiman a partir de caracteres (ver _estimar_tokens) y la estimación puede
# quedar corta con texto que trae muchos símbolos o acentos.
PRESUPUESTO_TOKENS_POR_LOTE = 6000

# Caracteres por token, aproximado para español. Conservador a propósito:
# subestimar los tokens es lo que provoca el error 400.
CARACTERES_POR_TOKEN = 3.0


def _estimar_tokens(texto: str) -> int:
    return max(1, int(len(texto) / CARACTERES_POR_TOKEN))


def _truncar_si_excede(texto: str) -> str:
    """Un solo fragmento más largo que la ventana del modelo no cabe en ninguna
    petición. Se recorta en vez de hacer fallar toda la ingesta.

    Pasa con fotos de páginas densas: el OCR devuelve la página entera como un
    bloque sin dobles saltos de línea, así que rag/ingesta.py no la puede
    partir en párrafos.
    """
    maximo_caracteres = int(LIMITE_TOKENS_MODELO * CARACTERES_POR_TOKEN * 0.9)
    if len(texto) <= maximo_caracteres:
        return texto
    print(
        f"  AVISO: un fragmento de {len(texto)} caracteres excede la ventana del "
        f"modelo; se truncó a {maximo_caracteres}. Revisa el troceado en rag/ingesta.py."
    )
    return texto[:maximo_caracteres]


def _agrupar_en_lotes(textos: list[str]) -> list[list[str]]:
    """Agrupa los textos en lotes que quepan en el presupuesto de tokens."""
    lotes: list[list[str]] = []
    lote_actual: list[str] = []
    tokens_actuales = 0

    for texto in textos:
        tokens = _estimar_tokens(texto)
        # Un texto que por sí solo llena el presupuesto va en su propia petición.
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
    """Devuelve un embedding por texto de entrada, en el mismo orden.

    Trocea la entrada en varias peticiones para no exceder la ventana de
    contexto del modelo (ver LIMITE_TOKENS_MODELO). Requiere VPN institucional
    activa (servidor BGE-M3 en 172.28.230.10:12556).
    """
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
        # El endpoint puede devolver los datos desordenados; se reordena por
        # el índice que trae cada uno para no desalinear textos y embeddings.
        for dato in sorted(respuesta.data, key=lambda d: d.index):
            embeddings.append(dato.embedding)

    return embeddings

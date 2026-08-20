# Control de calidad del texto que produce el OCR, antes de indexarlo.
#
# Por qué existe este módulo: el 2026-08-19, al procesar "2 recetas mas.jpg"
# (una doble página de un libro de cocina, foto nítida y bien iluminada),
# GLM-OCR leyó correctamente las primeras dos líneas y después entró en un
# bucle: repitió "Sive el mantequilla que se dore por ambio." cientos de veces
# hasta degenerar en texto sin sentido con caracteres chinos. Resultado: 60.648
# caracteres de basura, que eran el 66% de todo el recetario indexado.
#
# No fue un problema de calidad de imagen — fue la densidad y el layout
# multicolumna de la página. Un modelo generativo produce texto plausible
# aunque no pueda leer la entrada; sin una validación explícita, esa basura
# entra al índice y el RAG responde con ella.
#
# Este módulo NO decide qué hacer con el texto malo (eso es de ingesta.py),
# solo lo detecta y reporta por qué.

from dataclasses import dataclass

# --- Umbrales, calibrados con el recetario real (2026-08-19) ---------------
#
# Medición sobre los 223 fragmentos legítimos indexados en ese momento:
#   ratio de palabras únicas -> mínimo 0.3226, percentil 5 = 0.5816,
#                               mediana 0.9032
#   el fragmento degenerado  -> 0.0403
#
# 0.20 queda con margen amplio a los dos lados: 5x por encima del caso
# degenerado y 1.6x por debajo del peor fragmento legítimo. Si al ampliar el
# recetario aparece un fragmento bueno por debajo de este valor, hay que
# recalibrar con los datos nuevos y no simplemente bajar el número.
UMBRAL_RATIO_PALABRAS_UNICAS = 0.20

# El ratio de palabras únicas baja de forma natural en textos largos (las
# preposiciones se repiten), así que no se aplica a textos cortos, donde no es
# informativo.
MINIMO_PALABRAS_PARA_RATIO = 40

# Segunda señal, independiente de la longitud: cuántas veces aparece la
# secuencia de palabras más repetida del texto. Un texto legítimo puede repetir
# una frase corta dos o tres veces ("en un cuenco grande y"); repetirla decenas
# de veces es un bucle del modelo.
#
# Se cuenta la frecuencia TOTAL del n-grama, no las apariciones consecutivas:
# una primera versión contaba solo repeticiones consecutivas comparando
# n-gramas separados por n posiciones, y eso solo detectaba bucles cuyo período
# coincidía con LONGITUD_NGRAMA. Una frase de 8 palabras repetida 30 veces
# pasaba sin ser vista.
LONGITUD_NGRAMA = 5
MAXIMO_FRECUENCIA_NGRAMA = 8


@dataclass
class DiagnosticoTexto:
    """Resultado de evaluar un texto. `motivo` es None si el texto pasó."""

    es_degenerado: bool
    ratio_palabras_unicas: float
    frecuencia_ngrama_maxima: int
    ngrama_mas_repetido: str | None
    palabras: int
    motivo: str | None = None


def ratio_palabras_unicas(texto: str) -> float:
    palabras = texto.split()
    if not palabras:
        return 1.0
    return len(set(palabras)) / len(palabras)


def ngrama_mas_frecuente(texto: str, n: int = LONGITUD_NGRAMA) -> tuple[int, str | None]:
    """Devuelve (frecuencia, texto) del n-grama de palabras más repetido.

    Cuenta apariciones en todo el texto, no solo consecutivas, para detectar
    bucles con cualquier período de repetición.
    """
    palabras = texto.split()
    if len(palabras) < n * 2:
        return 1, None

    frecuencias: dict[tuple[str, ...], int] = {}
    for i in range(len(palabras) - n + 1):
        ngrama = tuple(palabras[i:i + n])
        frecuencias[ngrama] = frecuencias.get(ngrama, 0) + 1

    ngrama, frecuencia = max(frecuencias.items(), key=lambda par: par[1])
    return frecuencia, " ".join(ngrama)


def evaluar_texto(texto: str) -> DiagnosticoTexto:
    """Decide si un texto parece salida degenerada de un modelo generativo."""
    palabras = texto.split()
    ratio = ratio_palabras_unicas(texto)
    frecuencia, ngrama = ngrama_mas_frecuente(texto)

    motivo = None
    if frecuencia > MAXIMO_FRECUENCIA_NGRAMA:
        motivo = (
            f'la secuencia "{ngrama}" aparece {frecuencia} veces '
            f"(umbral: {MAXIMO_FRECUENCIA_NGRAMA})"
        )
    elif len(palabras) >= MINIMO_PALABRAS_PARA_RATIO and ratio < UMBRAL_RATIO_PALABRAS_UNICAS:
        motivo = (
            f"solo {ratio:.1%} de las palabras son distintas "
            f"(umbral: {UMBRAL_RATIO_PALABRAS_UNICAS:.0%})"
        )

    return DiagnosticoTexto(
        es_degenerado=motivo is not None,
        ratio_palabras_unicas=round(ratio, 4),
        frecuencia_ngrama_maxima=frecuencia,
        ngrama_mas_repetido=ngrama,
        palabras=len(palabras),
        motivo=motivo,
    )

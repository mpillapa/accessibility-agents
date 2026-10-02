# Detecta texto degenerado del OCR (bucles, repeticiones) antes de indexarlo.
# Solo diagnostica; qué hacer con el rechazo lo decide ingesta.py.
# Contexto y calibración: README.md raíz, "Control de calidad del OCR".

from dataclasses import dataclass

# Calibrado sobre 223 fragmentos legítimos (mínimo 0.32) y el caso degenerado
# (0.04). Si aparece un fragmento bueno por debajo, recalibrar, no bajar el número.
UMBRAL_RATIO_PALABRAS_UNICAS = 0.20

# En textos cortos el ratio no es informativo.
MINIMO_PALABRAS_PARA_RATIO = 40

# Frecuencia total del n-grama más repetido, no solo consecutiva: así se
# detectan bucles de cualquier período (bitácora 2.3).
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
    """Devuelve (frecuencia, texto) del n-grama de palabras más repetido."""
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

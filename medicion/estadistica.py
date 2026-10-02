# Estadística mínima para las tablas de la campaña, sin dependencias nuevas.
#
# Por qué no scipy: el proyecto no la usa en ningún otro lado y estas tres
# funciones son cortas. Se prueban contra valores conocidos en
# pruebas/prueba_medicion.py.

import math
import statistics


def media_de(valores: list[float]) -> tuple[float | None, float | None]:
    """Media y desviación estándar MUESTRAL (n-1). DE es None con n < 2."""
    valores = [v for v in valores if v is not None]
    if not valores:
        return None, None
    return statistics.fmean(valores), (statistics.stdev(valores) if len(valores) > 1 else None)


def wilson(exitos: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Intervalo de confianza de Wilson (95% por defecto) para una proporción.

    Se usa Wilson y no el intervalo normal porque con proporciones cerca de 0 o
    de 1 (lo esperable en emergencia o small talk) el normal da límites fuera
    de [0, 1] o de ancho cero con 20/20."""
    if n == 0:
        return 0.0, 1.0
    p = exitos / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    margen = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centro - margen), min(1.0, centro + margen)


def _rangos(valores: list[float]) -> list[float]:
    """Rangos 1..N con el promedio para los empates."""
    orden = sorted(range(len(valores)), key=lambda i: valores[i])
    rangos = [0.0] * len(valores)
    i = 0
    while i < len(orden):
        j = i
        while j + 1 < len(orden) and valores[orden[j + 1]] == valores[orden[i]]:
            j += 1
        for k in range(i, j + 1):
            rangos[orden[k]] = (i + j) / 2 + 1
        i = j + 1
    return rangos


def _chi2_cola(x: float, gl: int) -> float:
    """P(X > x) para chi-cuadrado con grados de libertad PARES (forma cerrada).
    Con 5 usuarios gl = 4; con 3 grupos, gl = 2."""
    if gl % 2:
        raise ValueError("solo grados de libertad pares (forma cerrada)")
    mitad = x / 2
    return math.exp(-mitad) * sum(mitad ** k / math.factorial(k) for k in range(gl // 2))


def kruskal_wallis(grupos: list[list[float]]) -> dict:
    """H de Kruskal-Wallis con corrección por empates y su valor p (chi² con
    k-1 gl). Pensado para comparar usuarios: ¿la distribución de tiempos o
    tokens cambia según la persona?"""
    grupos = [[v for v in g if v is not None] for g in grupos]
    grupos = [g for g in grupos if g]
    todos = [v for g in grupos for v in g]
    n = len(todos)
    if len(grupos) < 2 or n < 3:
        return {"H": None, "gl": None, "p": None}
    rangos = _rangos(todos)
    h, inicio = 0.0, 0
    for g in grupos:
        suma = sum(rangos[inicio:inicio + len(g)])
        h += suma * suma / len(g)
        inicio += len(g)
    h = 12 / (n * (n + 1)) * h - 3 * (n + 1)
    conteos: dict[float, int] = {}
    for v in todos:
        conteos[v] = conteos.get(v, 0) + 1
    correccion = 1 - sum(t ** 3 - t for t in conteos.values()) / (n ** 3 - n)
    if correccion > 0:
        h /= correccion
    gl = len(grupos) - 1
    return {"H": h, "gl": gl, "p": _chi2_cola(h, gl) if gl % 2 == 0 else None}

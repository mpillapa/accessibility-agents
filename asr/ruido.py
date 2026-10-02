# Matriz de ruido para evaluar el ASR: superpone ruido sintético a un audio
# limpio con SNR controlado (asr/README.md, "La matriz de ruido"). Solo stdlib.

import array
import math
import random
import wave
from pathlib import Path

# None = audio original, condición de control.
NIVELES_SNR_DB = [None, 20, 10, 5, 0]

TIPOS_RUIDO_SINTETICO = ["blanco", "rosa"]


def leer_wav(ruta: Path) -> tuple[array.array, int]:
    """Devuelve (muestras, frecuencia). Espera WAV mono de 16 bits."""
    with wave.open(str(ruta), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{ruta.name}: se esperaba 16 bits por muestra")
        if w.getnchannels() != 1:
            raise ValueError(f"{ruta.name}: se esperaba audio mono")
        muestras = array.array("h")
        muestras.frombytes(w.readframes(w.getnframes()))
        return muestras, w.getframerate()


def escribir_wav(ruta: Path, muestras: array.array, frecuencia: int) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(frecuencia)
        w.writeframes(muestras.tobytes())


def potencia_media(muestras) -> float:
    """Potencia media de la señal (media de los cuadrados)."""
    if not len(muestras):
        return 0.0
    return sum(float(m) * float(m) for m in muestras) / len(muestras)


def _ruido_blanco(n: int, semilla: int) -> list[float]:
    generador = random.Random(semilla)
    return [generador.gauss(0, 1) for _ in range(n)]


def _ruido_rosa(n: int, semilla: int) -> list[float]:
    """Ruido rosa por Voss-McCartney con 8 generadores."""
    generador = random.Random(semilla)
    generadores = 8
    valores = [generador.gauss(0, 1) for _ in range(generadores)]
    salida = []
    for i in range(n):
        # El generador k se renueva cada 2^k muestras.
        for k in range(generadores):
            if i % (1 << k) == 0:
                valores[k] = generador.gauss(0, 1)
        salida.append(sum(valores) / generadores)
    return salida


def generar_ruido(n: int, tipo: str = "blanco", semilla: int = 0) -> list[float]:
    """Ruido normalizado a potencia media 1, para escalarlo por SNR."""
    if tipo == "blanco":
        crudo = _ruido_blanco(n, semilla)
    elif tipo == "rosa":
        crudo = _ruido_rosa(n, semilla)
    else:
        raise ValueError(f"Tipo de ruido desconocido: {tipo!r}")

    potencia = potencia_media(crudo)
    if potencia == 0:
        return crudo
    factor = 1.0 / math.sqrt(potencia)
    return [c * factor for c in crudo]


def mezclar_con_ruido(
    muestras: array.array, snr_db: float, tipo: str = "blanco", semilla: int = 0
) -> array.array:
    """Superpone ruido a la señal al SNR pedido, en decibelios."""
    potencia_senal = potencia_media(muestras)
    if potencia_senal == 0:
        return array.array("h", muestras)

    potencia_ruido_objetivo = potencia_senal / (10 ** (snr_db / 10))
    amplitud = math.sqrt(potencia_ruido_objetivo)

    ruido = generar_ruido(len(muestras), tipo=tipo, semilla=semilla)

    # Recortar en vez de reescalar: reescalar cambiaría el SNR efectivo.
    mezcla = array.array("h")
    for muestra, r in zip(muestras, ruido):
        valor = int(muestra + amplitud * r)
        mezcla.append(max(-32768, min(32767, valor)))
    return mezcla


def snr_medido_db(limpio: array.array, con_ruido: array.array) -> float:
    """SNR real de un par (limpio, con ruido); el ruido es la diferencia."""
    potencia_senal = potencia_media(limpio)
    diferencia = [float(b) - float(a) for a, b in zip(limpio, con_ruido)]
    potencia_ruido = potencia_media(diferencia)
    if potencia_ruido == 0:
        return float("inf")
    return 10 * math.log10(potencia_senal / potencia_ruido)


def generar_matriz(
    ruta_limpia: Path,
    directorio_salida: Path,
    niveles=None,
    tipos=None,
) -> list[dict]:
    """Genera las variantes con ruido de un audio limpio; una fila por variante
    con el SNR pedido y el medido."""
    niveles = NIVELES_SNR_DB if niveles is None else niveles
    tipos = TIPOS_RUIDO_SINTETICO if tipos is None else tipos

    ruta_limpia = Path(ruta_limpia)
    muestras, frecuencia = leer_wav(ruta_limpia)
    base = ruta_limpia.stem
    filas = []

    for snr in niveles:
        if snr is None:
            destino = Path(directorio_salida) / f"{base}__limpio.wav"
            escribir_wav(destino, muestras, frecuencia)
            filas.append({
                "archivo": destino.name,
                "origen": ruta_limpia.name,
                "tipo_ruido": "ninguno",
                "snr_db_objetivo": "",
                "snr_db_medido": "",
            })
            continue

        for tipo in tipos:
            # Semilla distinta por variante.
            semilla = abs(hash((base, tipo, snr))) % (2**31)
            mezcla = mezclar_con_ruido(muestras, snr, tipo=tipo, semilla=semilla)
            destino = Path(directorio_salida) / f"{base}__{tipo}_{snr}dB.wav"
            escribir_wav(destino, mezcla, frecuencia)
            filas.append({
                "archivo": destino.name,
                "origen": ruta_limpia.name,
                "tipo_ruido": tipo,
                "snr_db_objetivo": snr,
                "snr_db_medido": round(snr_medido_db(muestras, mezcla), 2),
            })

    return filas

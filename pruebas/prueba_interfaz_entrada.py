# Pruebas de cómo la interfaz interpreta la barra del chat (interfaz/entrada.py).
# Uso: python -m pruebas.prueba_interfaz_entrada
# No requiere VPN, GPU ni Streamlit corriendo: dobles de st.chat_input(accept_audio=True).

import sys
from dataclasses import dataclass

from interfaz.entrada import leer_mensaje


class AudioFalso:
    def __init__(self, datos: bytes):
        self.datos = datos

    def getvalue(self) -> bytes:
        return self.datos


@dataclass
class ValorFalso:
    """Lo mismo que streamlit ChatInputValue: .text y .audio."""
    text: str = ""
    audio: AudioFalso | None = None


def prueba_nada_es_none():
    assert leer_mensaje(None) is None
    assert leer_mensaje(ValorFalso(text="   ")) is None
    assert leer_mensaje(ValorFalso(text="", audio=AudioFalso(b""))) is None
    return "sin texto ni audio no hay mensaje (no se procesa nada)"


def prueba_texto():
    assert leer_mensaje(ValorFalso(text=" ¿qué pastillas me tocan? ")) == {"texto": "¿qué pastillas me tocan?", "audio": None}
    assert leer_mensaje("hola") == {"texto": "hola", "audio": None}
    return "el texto llega limpio, venga como ChatInputValue o como str"


def prueba_audio():
    r = leer_mensaje(ValorFalso(audio=AudioFalso(b"RIFF...")))
    assert r == {"texto": None, "audio": b"RIFF..."}
    return "una grabación llega como bytes para transcribir"


def prueba_si_llegan_ambos_gana_la_voz():
    r = leer_mensaje(ValorFalso(text="hola", audio=AudioFalso(b"RIFF...")))
    assert r["audio"] == b"RIFF..." and r["texto"] is None
    return "si llegan texto y audio a la vez, se responde a lo que dijo por voz"


CASOS = [prueba_nada_es_none, prueba_texto, prueba_audio, prueba_si_llegan_ambos_gana_la_voz]


def main():
    print("Pruebas de la barra del chat de la interfaz (sin Streamlit, sin VPN)\n")
    fallos = 0
    for caso in CASOS:
        try:
            print(f"  OK    {caso()}")
        except AssertionError as e:
            fallos += 1
            print(f"  FALLA {caso.__name__}: {e}")
        except Exception as e:
            fallos += 1
            print(f"  ERROR {caso.__name__}: {type(e).__name__}: {e}")

    print(f"\n{len(CASOS) - fallos}/{len(CASOS)} pruebas pasaron")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())

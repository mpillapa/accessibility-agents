# Qué mandó la persona por la barra de chat: texto, voz o nada.
#
# Separado de app.py para poder probarlo sin levantar Streamlit
# (pruebas/prueba_interfaz_entrada.py).
#
# POR QUÉ LA VOZ VA EN LA BARRA DE CHAT (2026-09-30)
# --------------------------------------------------
# Antes había un micrófono aparte (st.audio_input) y fallaba al grabar:
#   1. Para vaciarlo después de enviar, se le cambiaba la clave MIENTRAS se
#      procesaba la respuesta (10-40 s). En pantalla seguía el micrófono viejo;
#      si la persona grababa otra vez sobre él, la página se redibujaba con el
#      micrófono nuevo, vacío, y esa grabación se perdía.
#   2. El micrófono y el cuadro de texto seguían activos durante el proceso:
#      cualquier toque reejecutaba la página y cortaba la respuesta a la mitad,
#      dejando el mensaje sin contestar.
# st.chat_input(accept_audio=True, submit_mode="disable") resuelve las dos
# cosas: es una sola barra, se vacía sola al enviar y queda bloqueada mientras
# el asistente trabaja.


def leer_mensaje(valor) -> dict | None:
    """Convierte el valor de st.chat_input en {"texto", "audio"}.

    `audio` son los bytes WAV de la grabación, o None. Si llegan las dos cosas,
    gana la voz: quien graba espera que le respondan a lo que dijo. Devuelve
    None si no se mandó nada (ni texto ni audio).
    """
    if valor is None:
        return None
    if isinstance(valor, str):  # chat_input sin accept_audio devuelve texto
        texto = valor.strip()
        return {"texto": texto, "audio": None} if texto else None

    audio = getattr(valor, "audio", None)
    datos = audio.getvalue() if audio is not None else None
    if datos:
        return {"texto": None, "audio": datos}
    texto = (getattr(valor, "text", "") or "").strip()
    return {"texto": texto, "audio": None} if texto else None

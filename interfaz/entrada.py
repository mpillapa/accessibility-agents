# Qué mandó la persona por la barra de chat: texto, voz o nada. Separado de
# app.py para probarlo sin Streamlit.
# La voz va en st.chat_input(accept_audio=True, submit_mode="disable"): se vacía
# sola al enviar y queda bloqueada mientras responde (ver interfaz/README.md).


def leer_mensaje(valor) -> dict | None:
    """Convierte el valor de st.chat_input en {"texto", "audio"} o None si vino vacío.

    `audio` son los bytes WAV. Si llegan texto y audio, gana el audio.
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

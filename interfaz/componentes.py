# Cómo se dibuja cada turno de la conversación. Solo presentación.

import streamlit as st

from interfaz import textos
from interfaz.estilo import CASI_NEGRO, BEIGE, ROJO_TEXTO, avatar
from interfaz.voz_salida import boton_escuchar, quitar_emojis


def avatar_usuario(nombre: str | None):
    return avatar((nombre or "U")[0].upper(), BEIGE, CASI_NEGRO)


def avatar_asistente():
    return avatar("A", ROJO_TEXTO)


def _segundos(valor) -> str:
    """12.34 -> '12,3 segundos'. Coma decimal, como se escribe en Ecuador."""
    if valor is None:
        return "—"
    return f"{float(valor):.1f}".replace(".", ",") + " segundos"


def dibujar_turno_usuario(mensaje: dict) -> None:
    with st.chat_message("user", avatar=avatar_usuario(mensaje.get("nombre"))):
        # Marca invisible: permite dar otro fondo a los mensajes de la persona
        # desde el CSS (ver estilo.py, selector :has(.marca-usuario)).
        st.markdown('<span class="marca-usuario"></span>', unsafe_allow_html=True)
        if mensaje.get("por_voz"):
            st.markdown(f'<span class="etiqueta-voz">{textos.ETIQUETA_POR_VOZ}</span>',
                        unsafe_allow_html=True)
        st.markdown(mensaje.get("consulta") or textos.NO_SE_ENTENDIO_EL_AUDIO)


def dibujar_turno_asistente(mensaje: dict, indice: int, leer_al_cargar: bool = False,
                            leer_habilitado: bool = True) -> None:
    """La respuesta, el botón para escucharla y el detalle plegado.

    `leer_al_cargar` va en True solo para la respuesta recién generada: al
    redibujar el historial no se vuelve a leer nada (ver voz_salida.py).
    """
    with st.chat_message("assistant", avatar=avatar_asistente()):
        respuesta = quitar_emojis(mensaje.get("respuesta") or "")

        if mensaje.get("intencion") == "EMERGENCY":
            # La respuesta de emergencia se enmarca en rojo: es la única que
            # tiene que verse aunque la persona no lea nada más.
            with st.container(key=f"emergencia_{indice}"):
                st.markdown(respuesta)
        else:
            st.markdown(respuesta)

        if leer_habilitado:
            boton_escuchar(respuesta, leer_al_cargar=leer_al_cargar)

        _dibujar_detalles(mensaje)


def _dibujar_detalles(mensaje: dict) -> None:
    tema = textos.TEMAS.get(mensaje.get("intencion"), mensaje.get("intencion") or "—")
    st.markdown(
        f'<p class="pie-respuesta">{tema} · respondió en {_segundos(mensaje.get("latencia_segundos"))}</p>',
        unsafe_allow_html=True,
    )

    with st.expander(textos.TITULO_DETALLES):
        transcripcion = mensaje.get("transcripcion")
        if transcripcion:
            # Se muestra aunque la transcripción se haya descartado —sobre todo
            # en ese caso—: ver qué oyó Whisper y por qué no se le creyó es el
            # punto entero del guardrail (orquestacion_langgraph/voz.py).
            st.caption(textos.DETALLE_QUE_OYO)
            st.code(transcripcion.get("texto") or "(nada)", language=None, wrap_lines=True)
            voz = "no encontró voz" if transcripcion.get("sin_voz") else "encontró voz"
            st.caption(
                f"{textos.DETALLE_VOZ_DETECTADA}: {voz} · idioma "
                f"{transcripcion.get('idioma')} (confianza "
                f"{transcripcion.get('probabilidad_idioma')}) · "
                f"{transcripcion.get('duracion_audio_s')} s de audio"
            )
            if mensaje.get("entrada_descartada"):
                st.warning(f"{textos.DETALLE_DESCARTADO}: {mensaje['entrada_descartada']}.")

        if mensaje.get("razonamiento") and not mensaje.get("entrada_descartada"):
            st.caption(textos.DETALLE_POR_QUE)
            st.write(mensaje["razonamiento"])

        st.caption(textos.DETALLE_RECORRIDO)
        st.markdown(_recorrido(mensaje.get("pasos", []), mensaje.get("traza_rag")))


def _recorrido(pasos: list[str], traza_rag) -> str:
    """El camino por el grafo, con los nombres técnicos entre paréntesis para
    quien quiera cruzarlo con el diagrama o con LangSmith."""
    lineas = []
    for nodo in pasos:
        lineas.append(f"1. {textos.PASOS.get(nodo, nodo)} `({nodo})`")
        if nodo in ("recetas", "recetas_cruce") and traza_rag:
            for paso in traza_rag:
                sub = paso.get("nodo") if isinstance(paso, dict) else paso
                lineas.append(f"    - {textos.PASOS_RECETARIO.get(sub, sub)} `({sub})`")
    return "\n".join(lineas) or "—"

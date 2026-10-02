# Interfaz web del sistema multiagente de accesibilidad.
#
# Uso, desde la raíz del repositorio (ahí está .streamlit/config.toml):
#   .venv/bin/streamlit run interfaz/app.py
#   .venv/bin/streamlit run interfaz/app.py --server.address 0.0.0.0   # por IP
# Requiere VPN institucional.

import hashlib
import os
import sys
import tempfile
import threading
from pathlib import Path

# Streamlit pone interfaz/ en sys.path[0]: la raíz va antes de los imports del proyecto.
RAIZ_DEL_PROYECTO = Path(__file__).parent.parent
if str(RAIZ_DEL_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_DEL_PROYECTO))

import streamlit as st

from infraestructura.modelos import describir_resolucion
from infraestructura.trazas import describir_trazas
from interfaz import textos
from interfaz.componentes import dibujar_turno_asistente, dibujar_turno_usuario
from interfaz.entrada import leer_mensaje
from interfaz.estilo import CSS, encabezado_html, pie_html
from medicacion.datos import cargar_perfiles
from orquestacion_langgraph.grafo import procesar_consulta_en_vivo
from orquestacion_langgraph.llm import VLLM_CHAT_BASE_URL, VLLM_CHAT_MODEL

st.set_page_config(page_title=textos.TITULO_PESTANA, page_icon=":material/home:", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

# Mismo audio de emergencia, limpio y degradado, para mostrar el guardrail del ASR.
CORPUS = RAIZ_DEL_PROYECTO / "corpus_audio" / "variantes"
EJEMPLOS = {
    "Emergencia, sin ruido": "f0006__limpio.wav",
    "Emergencia, con mucho ruido (0 dB)": "f0006__blanco_0dB.wav",
    "Emergencia, con ruido moderado (10 dB)": "f0006__blanco_10dB.wav",
    "Medicinas, sin ruido": "f0000__limpio.wav",
}


def _estado_inicial():
    st.session_state.setdefault("historial", [])
    # Huella del último audio enviado por archivo o ejemplo (ver _es_nuevo).
    st.session_state.setdefault("ultimo_audio", None)


@st.cache_resource(show_spinner=False)
def _precargar_whisper() -> threading.Thread:
    """Carga Whisper en un hilo, una vez por proceso (en frío tarda ~37 s).

    Si se graba antes de que termine, asr/transcribir.py espera esa misma carga.
    """
    def cargar():
        try:
            from asr.transcribir import obtener_modelo
            obtener_modelo()
        except Exception as error:  # sin GPU o sin faster-whisper: la voz fallará al usarla
            print(f"No se pudo precargar Whisper: {error}")

    hilo = threading.Thread(target=cargar, daemon=True, name="precarga-whisper")
    hilo.start()
    return hilo


def _barra_lateral() -> tuple[dict, bool]:
    """Devuelve (perfil elegido, si hay que leer en voz alta)."""
    perfiles = cargar_perfiles()
    ids = [p["id"] for p in perfiles]
    inicial = os.getenv("PERFIL_ACTIVO", "rosa")

    with st.sidebar:
        st.subheader(textos.TITULO_PERSONA)
        id_elegido = st.selectbox(
            textos.TITULO_PERSONA,
            ids,
            index=ids.index(inicial) if inicial in ids else 0,
            format_func=lambda i: next(f"{p['nombre']}, {p['edad']} años" for p in perfiles if p["id"] == i),
            label_visibility="collapsed",
            help=textos.AYUDA_PERSONA,
        )
        leer = st.toggle(textos.LEER_EN_VOZ_ALTA, value=True)

        st.space("small")
        if st.button(textos.BOTON_NUEVA, use_container_width=True):
            st.session_state.historial = []
            st.rerun()

        st.divider()
        with st.expander(textos.TITULO_TECNICO):
            st.caption(textos.TECNICO_MODELO)
            st.code(VLLM_CHAT_MODEL, language=None)
            st.caption(VLLM_CHAT_BASE_URL)

            resoluciones = describir_resolucion()
            if any(r["hubo_cambio"] == "True" for r in resoluciones.values()):
                st.warning(textos.TECNICO_CAMBIO)

            trazas = describir_trazas()
            if trazas["activo"] == "True":
                st.caption(f"{textos.TECNICO_TRAZAS_ACTIVAS} `{trazas['proyecto']}`")
                st.link_button(textos.TECNICO_VER_TRAZAS, "https://smith.langchain.com",
                               use_container_width=True)
            else:
                st.caption(textos.TECNICO_TRAZAS_INACTIVAS)

    perfil = next(p for p in perfiles if p["id"] == id_elegido)
    return perfil, leer


def _guardar_audio_temporal(datos: bytes, sufijo: str = ".wav") -> str:
    """transcribir() espera una ruta, no bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=sufijo) as f:
        f.write(datos)
        return f.name


def _es_nuevo(huella: str) -> bool:
    """Evita reprocesar el mismo audio en cada reejecución de Streamlit."""
    if huella == st.session_state.ultimo_audio:
        return False
    st.session_state.ultimo_audio = huella
    return True


def _otras_formas_de_audio() -> str | None:
    """Audio por archivo o de los ejemplos del corpus. Devuelve la ruta de un audio nuevo, o None."""
    with st.expander(textos.OTRAS_FORMAS_DE_AUDIO):
        archivo, ejemplos = st.tabs([textos.PESTANA_ARCHIVO, textos.PESTANA_EJEMPLOS])

        with archivo:
            subido = st.file_uploader(textos.ETIQUETA_ARCHIVO, type=["wav", "mp3", "m4a", "ogg"])
            if subido is not None:
                datos = subido.getvalue()
                if _es_nuevo(hashlib.sha256(datos).hexdigest()):
                    return _guardar_audio_temporal(datos, Path(subido.name).suffix or ".wav")

        with ejemplos:
            disponibles = {n: a for n, a in EJEMPLOS.items() if (CORPUS / a).exists()}
            if not disponibles:
                st.caption(textos.SIN_EJEMPLOS)
            else:
                st.caption(textos.EXPLICACION_EJEMPLOS)
                elegido = st.selectbox("Audio", list(disponibles), label_visibility="collapsed")
                ruta = CORPUS / disponibles[elegido]
                st.audio(str(ruta))
                if st.button(textos.BOTON_ENVIAR_EJEMPLO, use_container_width=True):
                    # Sin _es_nuevo: en una demo se reenvía el mismo ejemplo.
                    st.session_state.ultimo_audio = f"ejemplo:{elegido}"
                    return str(ruta)
    return None


def _elegir_sugerencia(texto: str) -> None:
    st.session_state.pendiente = texto


def _sugerencias() -> None:
    """Preguntas de ejemplo mientras la conversación está vacía.

    El clic solo deja la pregunta pendiente; se responde en la siguiente ejecución.
    """
    st.caption(textos.TITULO_SUGERENCIAS)
    columnas = st.columns(2)
    for i, sugerencia in enumerate(textos.SUGERENCIAS):
        columnas[i % 2].button(sugerencia, key=f"sugerencia_{i}", use_container_width=True,
                               on_click=_elegir_sugerencia, args=(sugerencia,))


def _responder(consulta: str, ruta_audio: str | None, id_perfil: str) -> dict:
    """Corre el grafo mostrando por dónde va. Devuelve el estado final."""
    pasos, estado_final = [], {}
    inicial = textos.TRABAJANDO_VOZ if ruta_audio else textos.TRABAJANDO_TEXTO

    with st.status(inicial, expanded=False) as estado_visual:
        for evento in procesar_consulta_en_vivo(consulta, ruta_audio, id_perfil=id_perfil):
            if evento.get("estado_final"):
                estado_final = evento["estado_final"]
                break
            nodo = evento["nodo"]
            pasos.append(nodo)
            estado_visual.update(label=textos.PASOS.get(nodo, nodo))
            st.caption(textos.PASOS.get(nodo, nodo))
        estado_visual.update(label=textos.LISTO, state="complete", expanded=False)

    estado_final["pasos"] = pasos
    return estado_final


_estado_inicial()
_precargar_whisper()
perfil, leer_en_voz_alta = _barra_lateral()

st.markdown(encabezado_html(textos.TITULO, textos.BAJADA), unsafe_allow_html=True)

consulta = st.session_state.pop("pendiente", None)
# Se vacía si llega un mensaje en esta ejecución, antes de mostrar el progreso.
zona_sugerencias = st.empty()
if not st.session_state.historial and not consulta:
    with zona_sugerencias.container():
        _sugerencias()

for i, mensaje in enumerate(st.session_state.historial):
    if mensaje["rol"] == "user":
        dibujar_turno_usuario(mensaje)
    else:
        dibujar_turno_asistente(mensaje, i, leer_habilitado=leer_en_voz_alta)

# Reserva el lugar del turno nuevo debajo del historial, aunque se procese después.
zona_turno_nuevo = st.container()

audio_nuevo = _otras_formas_de_audio()
st.markdown(pie_html(textos.PIE), unsafe_allow_html=True)

# Texto y voz en la misma barra, bloqueada mientras responde (ver entrada.py).
mensaje = leer_mensaje(st.chat_input(
    textos.PLACEHOLDER_CHAT,
    accept_audio=True,
    submit_mode="disable",
))
if mensaje and mensaje["audio"]:
    audio_nuevo = _guardar_audio_temporal(mensaje["audio"])
    consulta = None
elif mensaje:
    consulta = mensaje["texto"]

if audio_nuevo or consulta:
    zona_sugerencias.empty()
    turno_usuario = {
        "rol": "user",
        "consulta": consulta if not audio_nuevo else None,
        "por_voz": bool(audio_nuevo),
        "nombre": perfil["nombre"],
    }
    st.session_state.historial.append(turno_usuario)

    with zona_turno_nuevo:
        # Con voz, el mensaje queda arriba y se completa al llegar la transcripción.
        lugar_usuario = st.empty()
        with lugar_usuario.container():
            dibujar_turno_usuario(turno_usuario if not audio_nuevo
                                  else {**turno_usuario, "consulta": f"{textos.TRABAJANDO_VOZ}…"})
        try:
            resultado = _responder(consulta or "", audio_nuevo, perfil["id"])
            resultado["rol"] = "assistant"
            if audio_nuevo:
                # Audio descartado por el ASR: queda vacío y se muestra "no se entendió".
                turno_usuario["consulta"] = None if resultado.get("entrada_descartada") else resultado.get("consulta")
                with lugar_usuario.container():
                    dibujar_turno_usuario(turno_usuario)
            st.session_state.historial.append(resultado)
            dibujar_turno_asistente(
                resultado,
                len(st.session_state.historial) - 1,
                # Si la pregunta llegó por voz, la respuesta se lee sola.
                leer_al_cargar=bool(audio_nuevo) and leer_en_voz_alta,
                leer_habilitado=leer_en_voz_alta,
            )
        except Exception as error:
            # Los endpoints rotan de modelo: la caída es esperable, el detalle queda plegado.
            st.error(textos.ERROR_SIN_SERVIDOR)
            with st.expander(textos.ERROR_DETALLE):
                st.code(str(error), language=None)

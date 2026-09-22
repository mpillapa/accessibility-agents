# Interfaz web del sistema multiagente de accesibilidad.
#
# Ejecutar:
#   streamlit run interfaz/app.py
#   streamlit run interfaz/app.py --server.address 0.0.0.0   # accesible por IP
#
# QUÉ MUESTRA Y POR QUÉ
# ---------------------
# Un chat es lo que entiende cualquier persona, pero un chat a secas esconde
# justo lo que distingue a este sistema de un LLM suelto: que hay un orquestador
# decidiendo a qué agente va cada consulta, y que el RAG puede volver sobre sus
# pasos. Por eso cada respuesta trae, plegado, el recorrido real por el grafo.
#
# La barra lateral muestra con qué modelo se está respondiendo. No es un adorno:
# los endpoints de la Universidad rotaron de modelo cuatro veces en trece días
# (ver infraestructura/modelos.py), así que saber qué modelo contestó es parte
# de poder interpretar lo que se ve en pantalla.
#
# Esta capa no contiene lógica de negocio: importa el grafo y lo dibuja.

import sys
from pathlib import Path

# Streamlit ejecuta este archivo como script suelto, así que sys.path[0] es
# interfaz/ y no la raíz del proyecto: sin esto, `import infraestructura` falla
# con ModuleNotFoundError. El resto del proyecto se ejecuta con
# `python -m paquete.modulo`, que sí deja la raíz en sys.path; por eso este
# ajuste hace falta únicamente aquí, y tiene que ir ANTES de los imports del
# proyecto.
RAIZ_DEL_PROYECTO = Path(__file__).parent.parent
if str(RAIZ_DEL_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_DEL_PROYECTO))

import streamlit as st

from infraestructura.modelos import describir_resolucion
from infraestructura.trazas import describir_trazas
from orquestacion_langgraph.grafo import procesar_consulta_en_vivo
from orquestacion_langgraph.llm import VLLM_CHAT_BASE_URL, VLLM_CHAT_MODEL

# Nombres legibles de los nodos del grafo. El usuario final no tiene por qué
# leer identificadores de código.
NOMBRES_DE_NODO = {
    "orchestrator": "Decidiendo a qué agente corresponde",
    "medicacion": "Agente de medicación",
    "recetas": "Agente de recetas",
    "familia": "Agente de comunicación familiar",
    "emergencia": "Agente de emergencia",
    "small_talk": "Conversación",
}

# Nombres legibles de los nodos del subgrafo de RAG agéntico.
NOMBRES_DE_NODO_RAG = {
    "decidir_busqueda": "Decidiendo si hace falta el recetario",
    "recuperar": "Buscando en el recetario",
    "evaluar_relevancia": "Evaluando si lo encontrado sirve",
    "reformular": "No servía: reformulando la búsqueda",
    "expandir_contexto": "Trayendo la receta completa",
    "generar": "Redactando la respuesta",
    "sin_resultado": "No está en el recetario",
    "responder_sin_recetario": "Respondiendo sin buscar",
}

st.set_page_config(page_title="Asistente de accesibilidad", page_icon="🏠", layout="centered")


def _barra_lateral():
    """Estado de la infraestructura: con qué modelo se está respondiendo."""
    with st.sidebar:
        st.subheader("Sistema")
        st.caption("Modelo de chat en uso")
        st.code(VLLM_CHAT_MODEL, language=None)
        st.caption(VLLM_CHAT_BASE_URL)

        resoluciones = describir_resolucion()
        cambiados = [r for r in resoluciones.values() if r["hubo_cambio"] == "True"]
        if cambiados:
            st.warning(
                "El servidor cambió de modelo respecto a lo configurado. "
                "Las respuestas de ahora no son comparables con mediciones previas."
            )

        with st.expander("Endpoints"):
            for url, datos in resoluciones.items():
                st.caption(url)
                st.text(datos["resuelto"])

        st.divider()
        trazas = describir_trazas()
        if trazas["activo"] == "True":
            st.caption("Trazas en LangSmith")
            st.success(f"Activas · proyecto `{trazas['proyecto']}`")
            st.link_button(
                "Ver trazas", "https://smith.langchain.com", use_container_width=True
            )
        else:
            st.caption("Trazas en LangSmith")
            st.info("Desactivadas")

        st.divider()
        if st.button("Limpiar conversación", use_container_width=True):
            st.session_state.historial = []
            st.rerun()


def _dibujar_recorrido(pasos, traza_rag):
    """Muestra por dónde pasó la consulta dentro del grafo."""
    lineas = [f"- {NOMBRES_DE_NODO.get(n, n)}" for n in pasos]
    if traza_rag:
        for paso in traza_rag:
            nodo = paso.get("nodo") if isinstance(paso, dict) else paso
            lineas.append(f"    - {NOMBRES_DE_NODO_RAG.get(nodo, nodo)}")
    st.markdown("\n".join(lineas))


def _responder(consulta: str) -> dict:
    """Ejecuta el grafo mostrando el avance nodo por nodo.

    Devuelve el estado final para guardarlo en el historial.
    """
    pasos = []
    estado_final = {}

    with st.status("Pensando...", expanded=True) as estado_visual:
        for evento in procesar_consulta_en_vivo(consulta):
            if evento.get("estado_final"):
                estado_final = evento["estado_final"]
                break
            nodo = evento["nodo"]
            pasos.append(nodo)
            estado_visual.update(label=NOMBRES_DE_NODO.get(nodo, nodo))
            st.write(f"✓ {NOMBRES_DE_NODO.get(nodo, nodo)}")
        estado_visual.update(label="Listo", state="complete", expanded=False)

    estado_final["pasos"] = pasos
    return estado_final


def _dibujar_respuesta(mensaje: dict):
    """Respuesta del asistente + el recorrido plegado debajo."""
    st.markdown(mensaje["respuesta"])

    intencion = mensaje.get("intencion") or "—"
    latencia = mensaje.get("latencia_segundos")
    resumen = f"Ruteado a **{intencion}**"
    if latencia is not None:
        resumen += f" · {latencia}s"

    with st.expander(resumen):
        if mensaje.get("razonamiento"):
            st.caption("Por qué el orquestador eligió ese agente")
            st.write(mensaje["razonamiento"])
        st.caption("Recorrido por el grafo")
        _dibujar_recorrido(mensaje.get("pasos", []), mensaje.get("traza_rag"))


st.title("Asistente de accesibilidad")
st.caption(
    "Sistema multiagente para adultos mayores: recetas, medicación y "
    "acompañamiento. Prototipo académico."
)

_barra_lateral()

if "historial" not in st.session_state:
    st.session_state.historial = []

for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["rol"]):
        if mensaje["rol"] == "user":
            st.markdown(mensaje["consulta"])
        else:
            _dibujar_respuesta(mensaje)

if consulta := st.chat_input("Escribí tu consulta..."):
    st.session_state.historial.append({"rol": "user", "consulta": consulta})
    with st.chat_message("user"):
        st.markdown(consulta)

    with st.chat_message("assistant"):
        try:
            resultado = _responder(consulta)
            resultado["rol"] = "assistant"
            _dibujar_respuesta(resultado)
            st.session_state.historial.append(resultado)
        except Exception as error:
            # Con los endpoints rotando de modelo, la caída es un estado
            # esperable: hay que decirlo claro en vez de mostrar un stacktrace.
            st.error(
                "No se pudo responder. El servidor de modelos de la Universidad "
                "puede estar reiniciándose o la VPN caída.\n\n"
                f"Detalle: {error}"
            )

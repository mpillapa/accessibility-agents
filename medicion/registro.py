# Registro de tokens y tiempos por nodo durante una ejecución del grafo.
#
# QUÉ RESUELVE
# ------------
# Cristian pidió (reunión del 30-09) tokens y tiempo POR AGENTE y POR SISTEMA.
# LangSmith ya los registra, pero depender solo de un servicio externo para los
# números del paper es frágil: retención de trazas, cuotas, cambios de su API.
# Este callback escribe lo mismo en local y sirve para contrastar las dos
# fuentes (medicion/extraer_langsmith.py).
#
# CÓMO ATRIBUYE CADA LLAMADA A UN AGENTE SIN TOCAR LOS AGENTES
# ------------------------------------------------------------
# LangGraph agrega a la metadata de cada llamada al LLM el nodo que la hizo
# (`langgraph_node`) y la ruta de nodos padre (`langgraph_checkpoint_ns`, con la
# forma "recetas_cruce:<id>|evaluar_relevancia:<id>"). El primer tramo es el
# agente del grafo principal; el último, el subnodo del RAG. Verificado contra
# el servidor real el 2026-10-01 (sonda en la bitácora 22).
#
# Los nodos se cronometran con on_chain_start/on_chain_end: LangGraph abre una
# "chain" por cada nodo, con el mismo nombre que el nodo. Las funciones de ruteo
# y el subgrafo como tal también abren chains, pero con otro nombre, y se
# ignoran.
#
# LÍMITES (declararlos al reportar)
# ---------------------------------
# - Solo cuenta tokens de llamadas a modelos de CHAT. Las de embeddings (BGE-M3,
#   en la recuperación del RAG) no pasan por estos callbacks: su costo queda
#   dentro del TIEMPO del nodo `recuperar`, no en los tokens.
# - El tiempo de un agente es el de su nodo completo (lógica + LLM + red). En la
#   tarea medicamento × comida las dos ramas corren en paralelo, así que la suma
#   de los tiempos por agente es MAYOR que el tiempo del sistema. Es correcto:
#   el tiempo del sistema es el que vive la persona.

import threading
import time

from langchain_core.callbacks import BaseCallbackHandler


def ubicacion(metadata: dict | None) -> tuple[str | None, str | None]:
    """(agente, nodo) de una llamada según la metadata que pone LangGraph.

    El agente es el nodo del grafo principal; el nodo, el que hizo la llamada
    (igual al agente salvo dentro de un subgrafo)."""
    metadata = metadata or {}
    nodo = metadata.get("langgraph_node")
    ns = metadata.get("langgraph_checkpoint_ns") or metadata.get("checkpoint_ns") or ""
    agente = ns.split("|")[0].split(":")[0] or nodo
    return agente, nodo


def _uso(response) -> dict:
    """Tokens, modelo, fingerprint y motivo de fin de una respuesta del LLM.

    Lee primero el mensaje (usage_metadata es el formato estándar de
    LangChain) y completa con llm_output, que es donde langchain-openai deja
    `system_fingerprint`."""
    generacion = response.generations[0][0] if response.generations and response.generations[0] else None
    mensaje = getattr(generacion, "message", None)
    uso = getattr(mensaje, "usage_metadata", None) or {}
    meta_respuesta = getattr(mensaje, "response_metadata", None) or {}
    llm_output = response.llm_output or {}
    detalle_salida = uso.get("output_token_details") or {}
    return {
        "tokens_entrada": uso.get("input_tokens"),
        "tokens_salida": uso.get("output_tokens"),
        "tokens_razonamiento": detalle_salida.get("reasoning"),
        "modelo": meta_respuesta.get("model_name") or llm_output.get("model_name"),
        "fingerprint": meta_respuesta.get("system_fingerprint") or llm_output.get("system_fingerprint"),
        "fin": meta_respuesta.get("finish_reason")
               or (getattr(generacion, "generation_info", None) or {}).get("finish_reason"),
    }


class RegistroEjecucion(BaseCallbackHandler):
    """Acumula las llamadas al LLM y los tiempos de nodo de UNA ejecución.

    Una instancia por ejecución: así una ejecución que se pasa del tiempo y
    sigue corriendo en segundo plano no ensucia la siguiente. Es seguro entre
    hilos porque las ramas en paralelo del grafo llaman al callback a la vez.
    """

    def __init__(self):
        self._candado = threading.Lock()
        self._llm_abiertas: dict = {}
        self._nodos_abiertos: dict = {}
        self.llamadas: list[dict] = []
        self.nodos: list[dict] = []

    # --- llamadas al LLM -------------------------------------------------
    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        with self._candado:
            self._llm_abiertas[run_id] = (time.perf_counter(), *ubicacion(metadata))

    def on_llm_end(self, response, *, run_id, **kwargs):
        with self._candado:
            abierta = self._llm_abiertas.pop(run_id, None)
        if abierta is None:
            return
        inicio, agente, nodo = abierta
        registro = {"agente": agente, "nodo": nodo,
                    "segundos": round(time.perf_counter() - inicio, 3), "error": None}
        registro.update(_uso(response))
        with self._candado:
            self.llamadas.append(registro)

    def on_llm_error(self, error, *, run_id, **kwargs):
        with self._candado:
            abierta = self._llm_abiertas.pop(run_id, None)
            if abierta:
                inicio, agente, nodo = abierta
                self.llamadas.append({"agente": agente, "nodo": nodo,
                                      "segundos": round(time.perf_counter() - inicio, 3),
                                      "error": f"{type(error).__name__}: {error}"})

    # --- nodos del grafo -------------------------------------------------
    def on_chain_start(self, serialized, inputs, *, run_id, metadata=None, **kwargs):
        agente, nodo = ubicacion(metadata)
        if nodo is None or kwargs.get("name") != nodo or nodo.startswith("__"):
            return  # ruteo, el subgrafo como bloque, __start__ interno de LangGraph
        with self._candado:
            self._nodos_abiertos[run_id] = (time.perf_counter(), agente, nodo)

    def _cerrar_nodo(self, run_id, error=None):
        with self._candado:
            abierto = self._nodos_abiertos.pop(run_id, None)
            if abierto:
                inicio, agente, nodo = abierto
                self.nodos.append({"agente": agente, "nodo": nodo,
                                   "segundos": round(time.perf_counter() - inicio, 3), "error": error})

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._cerrar_nodo(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._cerrar_nodo(run_id, f"{type(error).__name__}: {error}")

    # --- resumen ----------------------------------------------------------
    def resumen(self) -> dict:
        with self._candado:
            return resumir(list(self.llamadas), list(self.nodos))


def _suma(valores) -> int | None:
    valores = [v for v in valores if v is not None]
    return sum(valores) if valores else None


def resumir(llamadas: list[dict], nodos: list[dict]) -> dict:
    """Agrega por agente y por sistema.

    `agentes[a].segundos` es la duración del nodo del grafo principal (incluye
    su subgrafo, si tiene). `subnodos` son los pasos internos de un agente
    (los del RAG), para desglosar dónde se va su tiempo."""
    agentes: dict[str, dict] = {}
    for n in nodos:
        if n["nodo"] == n["agente"]:
            a = agentes.setdefault(n["agente"], {"segundos": 0.0})
            a["segundos"] = round(a["segundos"] + n["segundos"], 3)
    for agente, datos in agentes.items():
        propias = [c for c in llamadas if c["agente"] == agente]
        datos.update({
            "llamadas_llm": len(propias),
            "tokens_entrada": _suma(c.get("tokens_entrada") for c in propias) or 0,
            "tokens_salida": _suma(c.get("tokens_salida") for c in propias) or 0,
            "tokens_razonamiento": _suma(c.get("tokens_razonamiento") for c in propias) or 0,
        })
    return {
        "agentes": agentes,
        "subnodos": [n for n in nodos if n["nodo"] != n["agente"]],
        "tokens_sistema": {
            "llamadas_llm": len(llamadas),
            "entrada": _suma(c.get("tokens_entrada") for c in llamadas) or 0,
            "salida": _suma(c.get("tokens_salida") for c in llamadas) or 0,
            "razonamiento": _suma(c.get("tokens_razonamiento") for c in llamadas) or 0,
        },
        "modelos": sorted({c["modelo"] for c in llamadas if c.get("modelo")}),
        "fingerprints": sorted({c["fingerprint"] for c in llamadas if c.get("fingerprint")}),
        "llamadas_cortadas": sum(1 for c in llamadas if c.get("fin") == "length"),
    }

# Callback que registra en local tokens y tiempos por nodo de una ejecución del
# grafo, para no depender solo de LangSmith (se contrastan en extraer_langsmith.py).
# Atribución por agente y límites: ver medicion/README.md.

import threading
import time

from langchain_core.callbacks import BaseCallbackHandler


def ubicacion(metadata: dict | None) -> tuple[str | None, str | None]:
    """(agente, nodo) según la metadata de LangGraph.

    Agente: primer tramo de langgraph_checkpoint_ns; nodo: el que hizo la llamada.
    """
    metadata = metadata or {}
    nodo = metadata.get("langgraph_node")
    ns = metadata.get("langgraph_checkpoint_ns") or metadata.get("checkpoint_ns") or ""
    agente = ns.split("|")[0].split(":")[0] or nodo
    return agente, nodo


def _uso(response) -> dict:
    """Tokens, modelo, fingerprint y motivo de fin de una respuesta del LLM.

    llm_output completa lo que falta: ahí deja langchain-openai `system_fingerprint`.
    """
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
    """Llamadas al LLM y tiempos de nodo de una ejecución.

    Una instancia por ejecución, para que una que siga corriendo tras el timeout
    no ensucie la siguiente. Con candado: las ramas paralelas llaman a la vez.
    """

    def __init__(self):
        self._candado = threading.Lock()
        self._llm_abiertas: dict = {}
        self._nodos_abiertos: dict = {}
        self.llamadas: list[dict] = []
        self.nodos: list[dict] = []

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

    # LangGraph abre una chain por nodo con el mismo nombre; las demás se ignoran.
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

    def resumen(self) -> dict:
        with self._candado:
            r = resumir(list(self.llamadas), list(self.nodos))
            # Llamadas que nunca respondieron: distinguen un cuelgue de un agente lento.
            r["llamadas_sin_respuesta"] = [{"agente": a, "nodo": n} for _, a, n in self._llm_abiertas.values()]
            return r


def _suma(valores) -> int | None:
    valores = [v for v in valores if v is not None]
    return sum(valores) if valores else None


def resumir(llamadas: list[dict], nodos: list[dict]) -> dict:
    """Agrega por agente y por sistema.

    `agentes[a].segundos` incluye su subgrafo; `subnodos` son los pasos internos (RAG).
    """
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

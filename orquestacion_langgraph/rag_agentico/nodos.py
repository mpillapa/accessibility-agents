# Nodos del subgrafo de RAG agéntico.
# Los juicios del LLM salen en texto libre terminado en "ETIQUETA: <valor>",
# extraído con regex y validado con Pydantic (por qué, en ../README.md).

import re
from typing import Literal

from pydantic import BaseModel

from orquestacion_langgraph.llm import llm
from orquestacion_langgraph.rag_agentico.estado import (
    EVALUAR_FRAGMENTO_POR_FRAGMENTO,
    EXPANDIR_A_RECETA_COMPLETA,
    FRAGMENTOS_POR_BUSQUEDA,
    MAX_INTENTOS_RECUPERACION,
    MAXIMO_CARACTERES_CONTEXTO,
    EstadoRAG,
)
from rag.buscar import buscar_receta_detallado, fragmentos_de_fuente


class DecisionBusqueda(BaseModel):
    necesita_recetario: bool
    razonamiento: str


class VeredictoRelevancia(BaseModel):
    es_util: bool
    razonamiento: str


def _extraer_etiqueta(texto: str, etiqueta: str) -> str | None:
    """Saca el valor de la última línea con formato 'ETIQUETA: <valor>'."""
    coincidencias = re.findall(rf"{etiqueta}:\s*([A-ZÁÉÍÓÚÑ_]+)", texto.upper())
    return coincidencias[-1] if coincidencias else None


def _razonamiento_previo(texto: str, etiqueta: str) -> str:
    return texto.split(f"{etiqueta}:")[0].strip() or texto.strip()


def nodo_decidir_busqueda(estado: EstadoRAG) -> dict:
    """Decide si la consulta necesita el recetario ('gracias, ya me salió' no)."""
    respuesta = llm.invoke(
        "Eres el componente de un asistente de cocina para adultos mayores que "
        "decide si hace falta consultar el recetario.\n\n"
        "Consulta el recetario si el usuario pide una receta concreta, sus "
        "ingredientes, cantidades o pasos.\n"
        "NO lo consultes si es un agradecimiento, un comentario sobre algo que "
        "ya cocinó, o una pregunta general que no requiere una receta puntual.\n\n"
        f"Consulta del usuario: '{estado['consulta']}'\n\n"
        "Razona en una línea y termina con una última línea exactamente así:\n"
        "DECISION: BUSCAR   (o)   DECISION: RESPONDER_DIRECTO"
    ).content

    valor = _extraer_etiqueta(respuesta, "DECISION")
    # Ante duda, buscar.
    decision = DecisionBusqueda(
        necesita_recetario=(valor != "RESPONDER_DIRECTO"),
        razonamiento=_razonamiento_previo(respuesta, "DECISION"),
    )

    return {
        "necesita_recetario": decision.necesita_recetario,
        "consulta_busqueda": estado["consulta"],
        "intentos": 0,
        "traza": [{
            "nodo": "decidir_busqueda",
            "necesita_recetario": decision.necesita_recetario,
            "razonamiento": decision.razonamiento,
        }],
    }


def ruta_tras_decidir(estado: EstadoRAG) -> Literal["recuperar", "responder_sin_recetario"]:
    return "recuperar" if estado["necesita_recetario"] else "responder_sin_recetario"


def nodo_recuperar(estado: EstadoRAG) -> dict:
    """Búsqueda semántica; no juzga relevancia."""
    fragmentos = buscar_receta_detallado(
        estado["consulta_busqueda"], k=FRAGMENTOS_POR_BUSQUEDA
    )

    return {
        "fragmentos": fragmentos,
        "intentos": estado["intentos"] + 1,
        "traza": [{
            "nodo": "recuperar",
            "intento": estado["intentos"] + 1,
            "consulta_usada": estado["consulta_busqueda"],
            "recuperados": len(fragmentos),
            "distancias": [f["distancia"] for f in fragmentos],
            "fuentes": [f["fuente"] for f in fragmentos],
        }],
    }


def _nombre_legible_fuente(fuente: str) -> str:
    """'arroz_con_leche.txt' -> 'arroz con leche'. Con 'receta1.jpeg' no aporta, pero no estorba."""
    sin_extension = fuente.rsplit(".", 1)[0]
    return sin_extension.replace("_", " ").replace("-", " ").strip()


def _juzgar_fragmento(consulta: str, fragmento: dict) -> VeredictoRelevancia:
    # Criterio de pertenencia y no de utilidad, con la fuente del fragmento y un
    # ejemplo ajeno a los casos de prueba (la papa). Bitácora 3.2 y 3.3.
    respuesta = llm.invoke(
        "Eres un evaluador estricto de un buscador de recetas.\n\n"
        f"El usuario pidió: '{consulta}'\n\n"
        f"Fragmento del recetario (receta: \"{_nombre_legible_fuente(fragmento['fuente'])}\"):\n"
        f"{fragmento['texto']}\n\n"
        "Pregunta: ¿este fragmento pertenece a la receta que pidió el usuario?\n\n"
        "SI = el fragmento es parte de ESA receta (su título, ingredientes, "
        "pasos o forma de servirla).\n"
        "NO = el fragmento es de una receta distinta, aunque comparta "
        "ingredientes o técnica de cocción.\n\n"
        "Compartir un ingrediente no basta: dos recetas que usan papa siguen "
        "siendo recetas distintas.\n\n"
        "Razona en una línea y termina con una última línea exactamente así:\n"
        "VEREDICTO: SI   (o)   VEREDICTO: NO"
    ).content

    return VeredictoRelevancia(
        es_util=(_extraer_etiqueta(respuesta, "VEREDICTO") == "SI"),
        razonamiento=_razonamiento_previo(respuesta, "VEREDICTO"),
    )


def _juzgar_conjunto(consulta: str, fragmentos: list[dict]) -> VeredictoRelevancia:
    contexto = "\n---\n".join(f["texto"] for f in fragmentos)
    respuesta = llm.invoke(
        "Eres un evaluador estricto. Decide si los fragmentos de recetario de "
        "abajo, en conjunto, alcanzan para responder la consulta del usuario.\n\n"
        f"Consulta del usuario: '{consulta}'\n\n"
        f"Fragmentos:\n{contexto}\n\n"
        "Razona en una línea y termina con una última línea exactamente así:\n"
        "VEREDICTO: SI   (o)   VEREDICTO: NO"
    ).content

    return VeredictoRelevancia(
        es_util=(_extraer_etiqueta(respuesta, "VEREDICTO") == "SI"),
        razonamiento=_razonamiento_previo(respuesta, "VEREDICTO"),
    )


def nodo_evaluar_relevancia(estado: EstadoRAG) -> dict:
    """Filtra los fragmentos que no pertenecen a la receta pedida."""
    fragmentos = estado["fragmentos"] or []

    if not fragmentos:
        return {
            "fragmentos_utiles": [],
            "traza": [{
                "nodo": "evaluar_relevancia",
                "evaluados": 0,
                "aceptados": 0,
                "nota": "no había nada que evaluar (recetario vacío o sin coincidencias)",
            }],
        }

    if EVALUAR_FRAGMENTO_POR_FRAGMENTO:
        veredictos = [_juzgar_fragmento(estado["consulta"], f) for f in fragmentos]
        utiles = [f for f, v in zip(fragmentos, veredictos) if v.es_util]
        detalle = [
            {"fuente": f["fuente"], "util": v.es_util, "razonamiento": v.razonamiento}
            for f, v in zip(fragmentos, veredictos)
        ]
    else:
        veredicto = _juzgar_conjunto(estado["consulta"], fragmentos)
        utiles = fragmentos if veredicto.es_util else []
        detalle = [{"conjunto": True, "util": veredicto.es_util,
                    "razonamiento": veredicto.razonamiento}]

    return {
        "fragmentos_utiles": utiles,
        "traza": [{
            "nodo": "evaluar_relevancia",
            "evaluados": len(fragmentos),
            "aceptados": len(utiles),
            "modo": "por_fragmento" if EVALUAR_FRAGMENTO_POR_FRAGMENTO else "conjunto",
            "detalle": detalle,
        }],
    }


def ruta_tras_evaluar(estado: EstadoRAG) -> Literal["generar", "reformular", "sin_resultado"]:
    """Con útiles se genera; sin útiles se reformula hasta agotar los intentos."""
    if estado["fragmentos_utiles"]:
        return "generar"
    if estado["intentos"] < MAX_INTENTOS_RECUPERACION:
        return "reformular"
    return "sin_resultado"


def nodo_reformular(estado: EstadoRAG) -> dict:
    """Reescribe la consulta a términos de recetario ('el arrocito dulce' -> arroz con leche)."""
    respuesta = llm.invoke(
        "La búsqueda en un recetario no dio resultados útiles. Reescribe la "
        "consulta del usuario para que funcione mejor en una búsqueda "
        "semántica sobre recetas de cocina.\n\n"
        "Usa el nombre probable del plato y sus ingredientes principales. "
        "Quita muletillas, diminutivos y referencias personales ('el que hacía "
        "mi mamá'). No inventes un plato distinto del que pide el usuario.\n\n"
        f"Consulta original del usuario: '{estado['consulta']}'\n"
        f"Búsqueda que ya se intentó y falló: '{estado['consulta_busqueda']}'\n\n"
        "Responde ÚNICAMENTE con la nueva búsqueda, sin comillas ni "
        "explicación, en una sola línea."
    ).content

    nueva = respuesta.strip().strip('"').split("\n")[-1].strip()
    # Vacía o demasiado larga: se repite la consulta original.
    if not nueva or len(nueva) > 200:
        nueva = estado["consulta"]

    return {
        "consulta_busqueda": nueva,
        "traza": [{
            "nodo": "reformular",
            "consulta_anterior": estado["consulta_busqueda"],
            "consulta_nueva": nueva,
        }],
    }


def nodo_expandir_contexto(estado: EstadoRAG) -> dict:
    """Trae, en orden, el resto de los fragmentos de cada receta que pasó el filtro.

    No vuelve a evaluar relevancia (ver README, "Reglas de negocio").
    """
    utiles = estado["fragmentos_utiles"] or []

    if not EXPANDIR_A_RECETA_COMPLETA:
        return {
            "fragmentos_contexto": utiles,
            "traza": [{"nodo": "expandir_contexto", "expansion": "desactivada"}],
        }

    fuentes = list(dict.fromkeys(f["fuente"] for f in utiles))  # sin duplicar, en orden
    contexto: list[dict] = []
    for fuente in fuentes:
        completos = fragmentos_de_fuente(fuente)
        # Colección vieja sin `orden` o archivo borrado: queda lo que pasó el filtro.
        contexto.extend(completos or [f for f in utiles if f["fuente"] == fuente])

    # Al recortar se priorizan los que pasaron el filtro.
    textos_utiles = {f["texto"] for f in utiles}
    if sum(len(f["texto"]) for f in contexto) > MAXIMO_CARACTERES_CONTEXTO:
        priorizados = sorted(contexto, key=lambda f: f["texto"] not in textos_utiles)
        recortado, acumulado = [], 0
        for fragmento in priorizados:
            if acumulado + len(fragmento["texto"]) > MAXIMO_CARACTERES_CONTEXTO:
                continue
            recortado.append(fragmento)
            acumulado += len(fragmento["texto"])
        contexto = sorted(recortado, key=lambda f: (f["fuente"], f.get("orden", 0)))

    return {
        "fragmentos_contexto": contexto,
        "traza": [{
            "nodo": "expandir_contexto",
            "fuentes": fuentes,
            "fragmentos_aprobados": len(utiles),
            "fragmentos_en_contexto": len(contexto),
            "caracteres": sum(len(f["texto"]) for f in contexto),
        }],
    }


def nodo_generar(estado: EstadoRAG) -> dict:
    """Redacta la respuesta final usando SOLO el contexto recuperado."""
    fragmentos = estado.get("fragmentos_contexto") or estado["fragmentos_utiles"]
    contexto = "\n---\n".join(f["texto"] for f in fragmentos)

    respuesta = llm.invoke(
        "Eres un asistente culinario que ayuda a personas mayores a preparar "
        "comidas. Adaptas medidas técnicas a referencias cotidianas (ej: 300ml "
        "= un vaso grande) para que sean comprensibles sin instrumentos de "
        "medición. Guías paso a paso y respondes en español.\n\n"
        "Usa ÚNICAMENTE la información del recetario de abajo. No agregues "
        "ingredientes ni pasos que no estén ahí. Si el recetario no cubre "
        "alguna parte de lo que se pregunta, dilo abiertamente en vez de "
        "completarlo por tu cuenta.\n\n"
        f"Recetario:\n{contexto}\n\n"
        f"Consulta del usuario: '{estado['consulta']}'"
    ).content

    return {
        "respuesta": respuesta,
        "hubo_resultado": True,
        "traza": [{
            "nodo": "generar",
            "fragmentos_usados": len(fragmentos),
            "fuentes": list(dict.fromkeys(f["fuente"] for f in fragmentos)),
        }],
    }


def nodo_sin_resultado(estado: EstadoRAG) -> dict:
    """Respuesta fija y sin LLM, para no inventar una receta (ver README)."""
    return {
        "respuesta": (
            "No encontré esa receta en tu recetario. Puedo ayudarte con otra, "
            "o si tienes la receta escrita en papel, pídele a alguien de tu "
            "familia que le tome una foto y la agregue."
        ),
        "hubo_resultado": False,
        "traza": [{
            "nodo": "sin_resultado",
            "intentos_realizados": estado["intentos"],
            "nota": "respuesta fija, sin LLM, para no inventar una receta",
        }],
    }


def nodo_responder_sin_recetario(estado: EstadoRAG) -> dict:
    """Consultas que no requieren buscar; el LLM responde sin afirmar recetas concretas."""
    respuesta = llm.invoke(
        "Eres un asistente culinario cálido y breve que conversa con una "
        "persona mayor. Responde en español, en dos o tres líneas como máximo.\n\n"
        "No describas recetas concretas ni des ingredientes o cantidades: no "
        "has consultado el recetario. Si el usuario termina pidiendo una "
        "receta puntual, ofrécele buscarla.\n\n"
        f"Mensaje del usuario: '{estado['consulta']}'"
    ).content

    return {
        "respuesta": respuesta,
        "hubo_resultado": True,
        "traza": [{"nodo": "responder_sin_recetario", "consulto_recetario": False}],
    }

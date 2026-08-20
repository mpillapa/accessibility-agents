# Nodos del subgrafo de RAG agéntico.
#
# La diferencia con el RAG anterior (una llamada a buscar_receta() dentro del
# nodo de recetas) es que aquí la recuperación es una DECISIÓN del agente y no
# un paso fijo: decide si buscar, juzga si lo que encontró sirve, reformula la
# consulta si no sirve, y admite que no encontró nada en vez de improvisar.
#
# Todos los nodos que piden un juicio al LLM usan el mismo patrón de salida:
# texto libre terminado en una línea "ETIQUETA: <valor>", que se extrae con
# regex y se valida con Pydantic. No se usa with_structured_output() porque
# no fue confiable con los modelos disponibles — ver el hallazgo documentado
# en ../README.md.

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


# --- Validación de los juicios del LLM -------------------------------------

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


# --- Nodo 1: ¿hace falta consultar el recetario? ----------------------------

def nodo_decidir_busqueda(estado: EstadoRAG) -> dict:
    """No toda consulta de cocina necesita el recetario. 'gracias, ya me salió'
    o '¿qué me recomiendas para hoy?' se responden sin buscar nada.

    Cuesta una llamada extra al LLM en todas las consultas de recetas. Se
    acepta ese costo porque es justamente lo que hace agéntica la recuperación:
    el agente decide si usa la herramienta en vez de usarla siempre."""
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
    # Ante una respuesta ambigua se busca igual: es preferible una búsqueda de
    # más que responder sobre una receta sin haberla consultado.
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


# --- Nodo 2: recuperar de la base vectorial --------------------------------

def nodo_recuperar(estado: EstadoRAG) -> dict:
    """Búsqueda semántica pura. No juzga nada: eso es del nodo evaluador."""
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


# --- Nodo 3: ¿lo recuperado sirve? -----------------------------------------

def _nombre_legible_fuente(fuente: str) -> str:
    """Convierte el nombre de archivo en algo que el LLM pueda leer como plato:
    'arroz_con_leche.txt' -> 'arroz con leche'.

    Algunos archivos del recetario se llaman 'receta1.jpeg' y no dicen de qué
    plato son. En esos casos el nombre no aporta, pero tampoco estorba: el
    evaluador sigue juzgando por el contenido del fragmento.
    """
    sin_extension = fuente.rsplit(".", 1)[0]
    return sin_extension.replace("_", " ").replace("-", " ").strip()


def _juzgar_fragmento(consulta: str, fragmento: dict) -> VeredictoRelevancia:
    # Tres decisiones de este prompt, cada una por un fallo medido:
    #
    # 1. El criterio es "pertenece a la receta pedida", no "es útil". Con
    #    "¿ayuda a responder?" el modelo aceptaba fragmentos de otro plato que
    #    compartía un ingrediente.
    #
    # 2. Se le dice de qué receta viene el fragmento. Sin eso, ante "quiero
    #    preparar sushi" el fragmento "Paso 1: cocinar el arroz en 500ml de
    #    agua" recibía SI en las tres corridas — y con razón, porque el
    #    fragmento no dice de qué receta es y cocinar arroz sí es parte de
    #    hacer sushi. Agregando la fuente ("receta: arroz con leche") pasó a NO
    #    en las tres, sin volverse más estricto con las consultas legítimas.
    #    Ningún prompt lo resolvió sin este dato: el problema era falta de
    #    información, no redacción.
    #
    # 3. El ejemplo del final usa la papa a propósito, y no el caso que se
    #    estaba tratando de corregir (sushi / arroz con leche): con ese ejemplo
    #    dentro del prompt, evaluar una consulta real de sushi daba veredictos
    #    peores — el modelo mezclaba el ejemplo con el caso a juzgar.
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
    """Filtra los fragmentos que no responden la consulta. Es el nodo que evita
    que el generador reciba contexto de una receta equivocada y termine
    respondiendo con seguridad sobre algo que el usuario no preguntó."""
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
    """El corazón del ciclo: con material útil se genera; sin material se
    reformula, salvo que ya se hayan agotado los intentos."""
    if estado["fragmentos_utiles"]:
        return "generar"
    if estado["intentos"] < MAX_INTENTOS_RECUPERACION:
        return "reformular"
    return "sin_resultado"


# --- Nodo 4: reformular y volver a intentar --------------------------------

def nodo_reformular(estado: EstadoRAG) -> dict:
    """Reescribe la consulta a términos de recetario.

    Este nodo es el que más aporta con esta población concreta: un adulto
    mayor rara vez usa el nombre técnico de un plato. Dice 'eso dulce del
    arrocito que hacía mi mamá', y el recetario está indexado como 'arroz con
    leche'. La primera búsqueda falla por vocabulario, no porque la receta
    falte."""
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
    # Si el modelo devuelve algo vacío o absurdamente largo, se conserva la
    # consulta original: es preferible repetir la búsqueda a buscar basura.
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


# --- Nodo 5: recuperar la receta completa ----------------------------------

def nodo_expandir_contexto(estado: EstadoRAG) -> dict:
    """Trae el resto de los fragmentos de cada receta que pasó el filtro.

    La búsqueda semántica devuelve fragmentos sueltos y el filtro de relevancia
    es estricto, así que de una receta troceada en cinco párrafos puede quedar
    aprobado uno solo. Redactar con ese único fragmento produce respuestas como
    "para hacer llapingachos, fríelos en la manteca" — la receta correcta, pero
    un paso aislado de ella.

    Este nodo agrupa por archivo de origen y recupera la receta entera, en
    orden. No vuelve a evaluar relevancia: si un fragmento de la receta pasó el
    filtro, la receta es la que el usuario pidió.
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
        # Si la fuente no se puede reconstruir (colección vieja sin `orden`, o
        # el archivo ya no está), se conserva lo que sí pasó el filtro.
        contexto.extend(completos or [f for f in utiles if f["fuente"] == fuente])

    # Recorte por presupuesto: primero los fragmentos que pasaron el filtro,
    # porque son los que con seguridad responden la consulta.
    textos_utiles = {f["texto"] for f in utiles}
    if sum(len(f["texto"]) for f in contexto) > MAXIMO_CARACTERES_CONTEXTO:
        priorizados = sorted(contexto, key=lambda f: f["texto"] not in textos_utiles)
        recortado, acumulado = [], 0
        for fragmento in priorizados:
            if acumulado + len(fragmento["texto"]) > MAXIMO_CARACTERES_CONTEXTO:
                continue
            recortado.append(fragmento)
            acumulado += len(fragmento["texto"])
        # Se reordena para que la receta llegue al generador en su orden real.
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


# --- Nodos terminales ------------------------------------------------------

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
    """Salida honesta cuando el recetario no tiene la receta.

    Respuesta fija y sin LLM a propósito: es el punto donde el sistema tiene
    más incentivo a inventar, y la forma más segura de no alucinar una receta
    es no darle al modelo la oportunidad de redactarla. Una receta inventada
    no es un error cosmético — puede terminar en alguien cocinando mal algo
    que después se come."""
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
    """Para consultas de cocina que no requieren buscar (agradecimientos,
    comentarios). Responde el LLM, pero sin afirmar nada sobre recetas
    concretas."""
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

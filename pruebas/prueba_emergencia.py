# Pruebas del especialista en emergencias.
#
# Uso (desde la raíz del repo):
#   python -m pruebas.prueba_emergencia
#
# NO requieren VPN: el nodo no llama al LLM, y esa es justamente una de las
# propiedades que se verifican acá.
#
# Por qué este nodo tiene pruebas propias siendo tan corto: es el único camino
# del sistema donde equivocarse tiene consecuencias físicas. Sobre el corpus se
# midió que 31 de 180 emergencias (17%) no llegaban a este nodo por fallos del
# ASR (secciones 10 y 13 de la bitácora); lo mínimo es garantizar que, cuando
# una llega, la respuesta sea correcta y no dependa de nada externo.
#
# Escrito sin pytest, igual que el resto de pruebas/ (ver prueba_ciclo_rag.py).

import sys

from orquestacion_langgraph import agentes
from orquestacion_langgraph.agentes import (
    MENSAJE_DE_EMERGENCIA,
    NUMERO_DE_EMERGENCIAS,
    nodo_emergencia,
)


def prueba_da_el_numero_de_emergencias():
    respuesta = nodo_emergencia({"consulta": "me caí y no me puedo levantar"})["respuesta"]
    assert NUMERO_DE_EMERGENCIAS in respuesta, respuesta
    return f"la respuesta contiene el número de emergencias ({NUMERO_DE_EMERGENCIAS})"


def prueba_no_llama_al_llm():
    """El nodo no debe depender del servidor de modelos.

    Si llamara al LLM, una VPN caída o una rotación de modelo —que pasaron
    cuatro veces en trece días— dejarían sin respuesta justo el camino crítico.
    Se verifica reemplazando el LLM por un doble que falla si alguien lo usa.
    """
    class LLMQueExplota:
        def invoke(self, *a, **k):
            raise AssertionError("el nodo de emergencia NO debe llamar al LLM")

    original = agentes.llm
    agentes.llm = LLMQueExplota()
    try:
        respuesta = nodo_emergencia({"consulta": "me duele el pecho"})["respuesta"]
        assert respuesta, "debe responder algo"
    finally:
        agentes.llm = original
    return "responde sin llamar al LLM (funciona con el servidor caído)"


def prueba_avisa_que_no_llamo_por_el_usuario():
    """Lo más peligroso sería dejar creer que ya viene ayuda cuando no viene."""
    respuesta = nodo_emergencia({"consulta": "ayuda"})["respuesta"].lower()
    assert "no puede llamar por usted" in respuesta, respuesta
    return "avisa explícitamente que el sistema NO hizo la llamada"


def prueba_no_afirma_haber_avisado_a_nadie():
    respuesta = nodo_emergencia({"consulta": "ayuda"})["respuesta"].lower()
    for frase in ["se registró", "ya avisamos", "viene ayuda", "en camino", "notificamos"]:
        assert frase not in respuesta, f"no debe afirmar {frase!r}: {respuesta}"
    return "no afirma haber registrado el aviso ni haber avisado a nadie"


def prueba_no_da_instrucciones_fisicas():
    """El sistema clasifica en una sola categoría EMERGENCY: no distingue una
    caída de un incendio. Un consejo físico único sería contraproducente en
    alguno de los casos ("no se mueva" ante humo en la cocina)."""
    respuesta = nodo_emergencia({"consulta": "hay humo en la cocina"})["respuesta"].lower()
    for frase in ["no se mueva", "no se levante", "acuéstese", "salga corriendo", "trate de no moverse"]:
        assert frase not in respuesta, f"no debe instruir {frase!r}: {respuesta}"
    return "no da instrucciones físicas que dependan del tipo de emergencia"


def prueba_la_respuesta_es_identica_para_cualquier_emergencia():
    consultas = [
        "me caí en el baño y no me puedo levantar",
        "siento que me falta el aire y me duele el pecho",
        "hay humo raro saliendo de la cocina",
    ]
    respuestas = {nodo_emergencia({"consulta": c})["respuesta"] for c in consultas}
    assert len(respuestas) == 1, "debe ser determinista, no variar según la consulta"
    return f"la respuesta es determinista para las {len(consultas)} emergencias probadas"


def prueba_el_numero_es_configurable():
    assert "{numero}" in MENSAJE_DE_EMERGENCIA, "el número no debe estar fijado en el texto"
    ejemplo = MENSAJE_DE_EMERGENCIA.format(numero="112")
    assert "112" in ejemplo and "911" not in ejemplo
    return "el número de emergencias es configurable, no está fijado por código"


CASOS = [
    prueba_da_el_numero_de_emergencias,
    prueba_no_llama_al_llm,
    prueba_avisa_que_no_llamo_por_el_usuario,
    prueba_no_afirma_haber_avisado_a_nadie,
    prueba_no_da_instrucciones_fisicas,
    prueba_la_respuesta_es_identica_para_cualquier_emergencia,
    prueba_el_numero_es_configurable,
]


def main():
    print("Pruebas del especialista en emergencias (sin VPN)\n")
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

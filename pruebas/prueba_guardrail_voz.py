# Pruebas del guardrail de voz: una transcripción no confiable no llega al Orchestrator.
# Uso: python -m pruebas.prueba_guardrail_voz
# No requiere VPN ni GPU (sin Whisper: se prueba la regla). Ver orquestacion_langgraph/voz.py.

import sys

from orquestacion_langgraph.voz import (
    MENSAJE_NO_SE_ENTENDIO,
    RAMA_CONTINUAR,
    RAMA_DESCARTAR,
    _motivo_para_descartar,
    nodo_no_se_entendio,
    ruta_tras_transcribir,
)


def prueba_descarta_cuando_el_vad_no_detecta_voz():
    # Caso real: Whisper dio este texto sobre ruido gaussiano puro; solo el VAD lo sabe.
    motivo = _motivo_para_descartar("Gracias por ver el video.", sin_voz=True)
    assert motivo is not None, "una transcripción sin voz detectada debe descartarse"
    assert "no encontró habla" in motivo, motivo
    return "descarta cuando el VAD no detecta voz, aunque haya texto"


def prueba_descarta_muletilla_conocida_aunque_el_vad_pase():
    # Segunda capa: el VAD deja pasar ruido estructurado.
    motivo = _motivo_para_descartar("Gracias por ver el video.", sin_voz=False)
    assert motivo is not None, "una muletilla conocida debe descartarse"
    assert "muletilla" in motivo, motivo
    return "descarta muletillas conocidas aunque el VAD haya detectado voz"


def prueba_descarta_transcripcion_vacia():
    motivo = _motivo_para_descartar("   ", sin_voz=False)
    assert motivo is not None, "una transcripción vacía debe descartarse"
    return "descarta la transcripción vacía"


def prueba_acepta_una_transcripcion_legitima():
    # La frase de la emergencia perdida, bien transcrita.
    motivo = _motivo_para_descartar(
        "Me caí en el baño y no me puedo levantar, ayúdame por favor", sin_voz=False
    )
    assert motivo is None, f"no debía descartarse, motivo={motivo!r}"
    return "acepta una transcripción legítima (la emergencia real)"


def prueba_acepta_frases_cotidianas():
    # Una frase por cada una de las 5 intenciones.
    frases = [
        "quiero saber cómo hago el llapingacho",
        "ya me tomé la pastilla de la presión",
        "llamá a mi hija por favor",
        "gracias mijito, ya me salió rico",
        "me caí y no me puedo levantar",
    ]
    for frase in frases:
        motivo = _motivo_para_descartar(frase, sin_voz=False)
        assert motivo is None, f"{frase!r} no debía descartarse: {motivo}"
    return f"acepta las {len(frases)} frases cotidianas probadas, sin falsos positivos"


# Se comprueba la rama, no el nodo destino: las claves etiquetan las flechas del diagrama.
def prueba_el_edge_desvia_cuando_hay_motivo():
    destino = ruta_tras_transcribir({"entrada_descartada": "el detector de voz no encontró habla"})
    assert destino == RAMA_DESCARTAR, destino
    return "el edge condicional desvía fuera del Orchestrator cuando hay motivo"


def prueba_el_edge_deja_pasar_cuando_no_hay_motivo():
    destino = ruta_tras_transcribir({"entrada_descartada": None})
    assert destino == RAMA_CONTINUAR, destino
    return "el edge condicional deja pasar al Orchestrator cuando la entrada es confiable"


def prueba_las_ramas_coinciden_con_el_mapa_del_grafo():
    """Un renombre a medias haría fallar a LangGraph recién en ejecución."""
    from orquestacion_langgraph.grafo import construir_grafo

    destinos = {
        e.data
        for e in construir_grafo().get_graph().edges
        if e.source == "transcribir_voz"
    }
    assert destinos == {RAMA_CONTINUAR, RAMA_DESCARTAR}, destinos
    return "las ramas del edge coinciden con el mapa declarado en el grafo"


def prueba_el_nodo_de_descarte_pide_repetir_y_no_clasifica():
    salida = nodo_no_se_entendio({"entrada_descartada": "la transcripción quedó vacía"})
    assert salida["respuesta"] == MENSAJE_NO_SE_ENTENDIO, salida["respuesta"]
    # No debe asumir SMALL_TALK ni otra intención real ante una entrada dudosa.
    assert salida["intencion"] == "NO_SE_ENTENDIO", salida["intencion"]
    assert salida["razonamiento"], "debe quedar registrado por qué se descartó"
    return "el nodo de descarte pide repetir y no clasifica la entrada dudosa"


CASOS = [
    prueba_descarta_cuando_el_vad_no_detecta_voz,
    prueba_descarta_muletilla_conocida_aunque_el_vad_pase,
    prueba_descarta_transcripcion_vacia,
    prueba_acepta_una_transcripcion_legitima,
    prueba_acepta_frases_cotidianas,
    prueba_el_edge_desvia_cuando_hay_motivo,
    prueba_el_edge_deja_pasar_cuando_no_hay_motivo,
    prueba_las_ramas_coinciden_con_el_mapa_del_grafo,
    prueba_el_nodo_de_descarte_pide_repetir_y_no_clasifica,
]


def main():
    print("Pruebas del guardrail de entrada por voz (sin GPU, sin VPN)\n")
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

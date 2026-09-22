# Pruebas del guardrail de entrada por voz.
#
# Uso (desde la raíz del repo):
#   python -m pruebas.prueba_guardrail_voz
#
# NO requieren GPU, ni Whisper cargado, ni VPN: la transcripción se reemplaza
# por un doble. Lo que se verifica es LA REGLA — que una transcripción no
# confiable no llegue al Orchestrator, y que una buena sí.
#
# Por qué importa esta prueba en particular: sin el guardrail, el sistema
# reproduce el fallo medido sobre el corpus, donde 31 de 180 emergencias (17%)
# no se atendieron porque Whisper inventó texto plausible sobre audio pobre.
# Ver orquestacion_langgraph/voz.py.
#
# Escrito sin pytest, igual que el resto de pruebas/ (ver prueba_ciclo_rag.py).

import sys

from orquestacion_langgraph.voz import (
    MENSAJE_NO_SE_ENTENDIO,
    RAMA_CONTINUAR,
    RAMA_DESCARTAR,
    _motivo_para_descartar,
    nodo_no_se_entendio,
    ruta_tras_transcribir,
)


# --- La regla de descarte --------------------------------------------------

def prueba_descarta_cuando_el_vad_no_detecta_voz():
    # El caso real: Whisper devolvió texto con probabilidad de idioma 1.00
    # sobre ruido gaussiano puro. El texto parece válido; el VAD es lo único
    # que sabe que no hubo habla.
    motivo = _motivo_para_descartar("Gracias por ver el video.", sin_voz=True)
    assert motivo is not None, "una transcripción sin voz detectada debe descartarse"
    assert "no encontró habla" in motivo, motivo
    return "descarta cuando el VAD no detecta voz, aunque haya texto"


def prueba_descarta_muletilla_conocida_aunque_el_vad_pase():
    # Segunda capa: el VAD deja pasar audio con ruido estructurado, pero el
    # texto es una de las muletillas que Whisper inventa.
    motivo = _motivo_para_descartar("Gracias por ver el video.", sin_voz=False)
    assert motivo is not None, "una muletilla conocida debe descartarse"
    assert "muletilla" in motivo, motivo
    return "descarta muletillas conocidas aunque el VAD haya detectado voz"


def prueba_descarta_transcripcion_vacia():
    motivo = _motivo_para_descartar("   ", sin_voz=False)
    assert motivo is not None, "una transcripción vacía debe descartarse"
    return "descarta la transcripción vacía"


def prueba_acepta_una_transcripcion_legitima():
    # Contrapeso obligatorio: un guardrail que descarta todo no sirve. Esta es
    # justamente la frase del caso de la emergencia perdida, bien transcrita.
    motivo = _motivo_para_descartar(
        "Me caí en el baño y no me puedo levantar, ayúdame por favor", sin_voz=False
    )
    assert motivo is None, f"no debía descartarse, motivo={motivo!r}"
    return "acepta una transcripción legítima (la emergencia real)"


def prueba_acepta_frases_cotidianas():
    # Falsos positivos: frases normales de las 5 intenciones no deben caer.
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


# --- El desvío en el grafo -------------------------------------------------

# Estos dos casos comprueban la RAMA que elige el edge, no el nombre del nodo
# destino: las claves son descriptivas a propósito, para que el diagrama del
# grafo etiquete cada flecha con el motivo de la decisión (ver construir_grafo).
def prueba_el_edge_desvia_cuando_hay_motivo():
    destino = ruta_tras_transcribir({"entrada_descartada": "el detector de voz no encontró habla"})
    assert destino == RAMA_DESCARTAR, destino
    return "el edge condicional desvía fuera del Orchestrator cuando hay motivo"


def prueba_el_edge_deja_pasar_cuando_no_hay_motivo():
    destino = ruta_tras_transcribir({"entrada_descartada": None})
    assert destino == RAMA_CONTINUAR, destino
    return "el edge condicional deja pasar al Orchestrator cuando la entrada es confiable"


def prueba_las_ramas_coinciden_con_el_mapa_del_grafo():
    """Las ramas que devuelve el edge tienen que existir en el mapa que declara
    el grafo. Si alguien renombra una y olvida la otra, LangGraph fallaría recién
    en tiempo de ejecución; esto lo detecta antes."""
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
    # Lo importante: NO inventa una intención de las 5 reales. Asumir
    # SMALL_TALK ante una entrada dudosa es exactamente el fallo que se
    # quiere evitar.
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

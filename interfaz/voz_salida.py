# Lectura en voz alta de las respuestas.
#
# POR QUÉ EN EL NAVEGADOR Y NO EN EL SERVIDOR
# -------------------------------------------
# La síntesis de voz la hace el navegador (Web Speech API, `speechSynthesis`).
# Tres razones:
#   1. No agrega un modelo más que dependa del servidor de la Universidad, que
#      ya rotó cuatro veces en trece días (bitácora, sección 11).
#   2. A diferencia del micrófono, `speechSynthesis` NO exige HTTPS ni
#      localhost: funciona también cuando la app se abre por IP.
#   3. El texto de la respuesta ya está en la página; no hay que mandarlo a
#      ningún lado.
#
# El costo, declarado: la voz depende del sistema operativo de quien mira. En
# Windows y Android suele haber voces en español; en algunos Linux no hay
# ninguna y el botón no suena.
#
# QUÉ SE LEE
# ----------
# `texto_para_leer()` convierte el markdown de la respuesta en texto hablable:
# sin asteriscos, sin numerales de títulos, sin emojis (un lector de voz dice
# "sol naciente" en vez de callarse) y sin el aviso de datos ficticios, que
# sigue visible en pantalla pero leído en cada respuesta se vuelve ruido.

import json
import re

import streamlit.components.v1 as componentes

from interfaz.estilo import ROJO_TEXTO, BEIGE, CASI_NEGRO

# Velocidad de lectura. 1.0 es la normal del navegador; algo más lenta se
# entiende mejor, sobre todo con números y horarios.
VELOCIDAD_DE_LECTURA = 0.92

# El aviso que medicacion/agente.py agrega al final de cada respuesta.
_AVISO_DATOS_FICTICIOS = re.compile(r"_?Prototipo académico con datos ficticios[^\n]*", re.IGNORECASE)

_EMOJIS = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF"
    "\U0000FE0F\U0000200D\U00002300-\U000023FF\U00002B00-\U00002BFF]+"
)


def quitar_emojis(texto: str) -> str:
    """Saca los emojis del texto. Se usa también para MOSTRAR las respuestas:
    los agentes a veces encabezan con 🌅 o ⚠️, y es lo primero que delata que un
    texto lo escribió un modelo."""
    return re.sub(r"[ \t]{2,}", " ", _EMOJIS.sub("", texto))


def texto_para_leer(markdown: str) -> str:
    """El texto de una respuesta, listo para un lector de voz.

    Función pura: se prueba sin navegador (pruebas/prueba_interfaz_voz.py).
    """
    texto = _AVISO_DATOS_FICTICIOS.sub("", markdown)
    texto = quitar_emojis(texto)
    texto = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", texto)           # [texto](url)
    texto = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", texto)              # código
    texto = re.sub(r"^\s{0,3}#{1,6}\s*", "", texto, flags=re.M)       # títulos
    texto = re.sub(r"^\s*>\s?", "", texto, flags=re.M)                 # citas
    texto = re.sub(r"^\s*[-*+]\s+", "", texto, flags=re.M)             # viñetas
    texto = re.sub(r"^\s*\|?[-:\s|]{3,}\|?\s*$", "", texto, flags=re.M)  # separadores de tabla
    texto = texto.replace("|", ", ")
    texto = re.sub(r"[ \t]*,[ \t]*", ", ", texto)
    texto = re.sub(r"(\*\*|__|\*|_)(.+?)\1", r"\2", texto)            # negritas y cursivas
    texto = re.sub(r"^\s*-{3,}\s*$", "", texto, flags=re.M)            # líneas horizontales
    # Cada renglón termina en pausa: sin esto, una lista se lee de corrido.
    renglones = [r.strip(" ,") for r in texto.splitlines() if r.strip(" ,")]
    renglones = [r if r[-1] in ".:;!?" else r + "." for r in renglones]
    return " ".join(renglones)


def boton_escuchar(texto: str, leer_al_cargar: bool = False) -> None:
    """Botón "Escuchar la respuesta" / "Detener".

    `leer_al_cargar` intenta leer apenas aparece. Solo se usa en la respuesta
    recién generada, nunca al redibujar el historial (Streamlit vuelve a
    ejecutar todo el script en cada interacción: sin esa regla, cada clic
    volvería a leer todas las respuestas anteriores). Si el navegador bloquea la
    lectura automática, el botón sigue ahí.
    """
    hablado = texto_para_leer(texto)
    if not hablado:
        return

    pagina = f"""
<!doctype html><html><head><meta charset="utf-8"><style>
  body {{ margin:0; font-family:"Helvetica Neue",Helvetica,Arial,sans-serif; }}
  button {{
    font-size:16px; padding:8px 16px; min-height:42px; cursor:pointer;
    border:1px solid {BEIGE}; border-radius:8px; background:#fff; color:{CASI_NEGRO};
  }}
  button:hover {{ border-color:{ROJO_TEXTO}; color:{ROJO_TEXTO}; }}
  button[data-hablando="1"] {{ border-color:{ROJO_TEXTO}; color:{ROJO_TEXTO}; }}
  .sin-voz {{ color:#5E5B57; font-size:14px; }}
</style></head><body>
<button id="b" type="button" aria-label="Escuchar la respuesta en voz alta">Escuchar la respuesta</button>
<script>
  const texto = {json.dumps(hablado)};
  const boton = document.getElementById("b");
  const voz = window.speechSynthesis;

  function vozEnEspanol() {{
    const voces = voz ? voz.getVoices() : [];
    const preferidas = ["es-EC", "es-419", "es-US", "es-MX", "es-CO", "es-ES"];
    for (const idioma of preferidas) {{
      const v = voces.find(x => x.lang === idioma);
      if (v) return v;
    }}
    return voces.find(x => x.lang && x.lang.toLowerCase().startsWith("es")) || null;
  }}

  function marcar(hablando) {{
    boton.dataset.hablando = hablando ? "1" : "0";
    boton.textContent = hablando ? "Detener" : "Escuchar la respuesta";
  }}

  function leer() {{
    if (!voz) {{ boton.outerHTML = '<span class="sin-voz">Este navegador no puede leer en voz alta.</span>'; return; }}
    voz.cancel();
    const u = new SpeechSynthesisUtterance(texto);
    u.lang = "es-EC";
    const v = vozEnEspanol();
    if (v) u.voice = v;
    u.rate = {VELOCIDAD_DE_LECTURA};
    u.onend = () => marcar(false);
    u.onerror = () => marcar(false);
    marcar(true);
    voz.speak(u);
  }}

  boton.addEventListener("click", () => {{
    if (voz && voz.speaking) {{ voz.cancel(); marcar(false); }} else {{ leer(); }}
  }});

  // onvoiceschanged puede dispararse varias veces: sin la bandera, la
  // respuesta se leería repetida.
  let yaLeida = false;
  function leerUnaVez() {{ if (!yaLeida) {{ yaLeida = true; leer(); }} }}
  {"if (voz) { if (voz.getVoices().length) leerUnaVez(); else voz.onvoiceschanged = leerUnaVez; }" if leer_al_cargar else ""}
</script></body></html>
"""
    componentes.html(pagina, height=52)


# Lectura en voz alta de las respuestas con speechSynthesis del navegador, que no
# depende del servidor ni exige HTTPS (ver interfaz/README.md).

import json
import re

import streamlit.components.v1 as componentes

from interfaz.estilo import ROJO_TEXTO, BEIGE, CASI_NEGRO

# 1.0 es la normal; algo más lenta se entiende mejor con números y horarios.
VELOCIDAD_DE_LECTURA = 0.92

# El aviso que medicacion/agente.py agrega al final de cada respuesta.
_AVISO_DATOS_FICTICIOS = re.compile(r"_?Prototipo académico con datos ficticios[^\n]*", re.IGNORECASE)

_EMOJIS = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF"
    "\U0000FE0F\U0000200D\U00002300-\U000023FF\U00002B00-\U00002BFF]+"
)


def quitar_emojis(texto: str) -> str:
    """Saca los emojis del texto; también se usa al mostrar las respuestas."""
    return re.sub(r"[ \t]{2,}", " ", _EMOJIS.sub("", texto))


def texto_para_leer(markdown: str) -> str:
    """Markdown de una respuesta convertido en texto hablable, sin el aviso de datos ficticios."""
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

    `leer_al_cargar` solo para la respuesta recién generada: Streamlit redibuja
    el historial en cada interacción y releería todo.
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


# Identidad visual de la interfaz. Solo presentación: ninguna regla de negocio.
#
# De dónde sale cada decisión:
#
# - PALETA: de la hoja de estilos pública de usfq.edu.ec (2026-09-23). El rojo
#   institucional #ED1C24 NO alcanza contraste AA como texto (4.38:1 sobre
#   blanco), así que se usa solo como acento; botones y texto en rojo usan
#   ROJO_TEXTO (6.85:1). Ver también .streamlit/config.toml.
# - TIPOGRAFÍA: el sitio de la USFQ usa Baskerville para títulos y Helvetica
#   para texto. Baskerville solo viene instalada en macOS, así que se carga
#   Libre Baskerville (licencia libre, misma familia) y se cae a Georgia si no
#   hay conexión.
# - TAMAÑOS: el público son adultos mayores. Letra base de 18 px, botones y
#   zona del micrófono más grandes de lo habitual.
#
# No se usa el logotipo de la Universidad: esto es un prototipo de tesis, no un
# producto institucional.

from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

ROJO_USFQ = "#ED1C24"      # solo acentos decorativos (líneas, bordes)
ROJO_TEXTO = "#B5121B"     # texto y botones: contraste 6.85:1 sobre blanco
CASI_NEGRO = "#231F20"
CREMA = "#FAF3E9"
BEIGE = "#D8D4CB"
GRIS_TEXTO = "#5E5B57"     # texto secundario: 6.13:1 sobre crema

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&display=swap');

/* Las familias tipográficas se declaran en .streamlit/config.toml (font y
   headingFont). NO se fuerzan acá con un selector general: la primera versión
   lo hizo y pisó la fuente de íconos de Streamlit, que pasaron a mostrarse
   como texto ("keyboard_double_arrow_right"). */
h1, h2, h3 {{ color: {CASI_NEGRO}; letter-spacing: -0.01em; }}

/* Barra superior de Streamlit: sin fondo, para que el encabezado propio mande. */
[data-testid="stHeader"] {{ background: transparent; }}

/* Contenido un poco más ancho que el "centered" por defecto, sin llegar a wide. */
.block-container {{ max-width: 860px; padding-top: 2.2rem; }}

/* --- Encabezado --------------------------------------------------------- */
.encabezado {{
    border-top: 6px solid {ROJO_USFQ};
    padding: 1.1rem 0 0.9rem 0;
    margin-bottom: 1.2rem;
    border-bottom: 1px solid {BEIGE};
}}
.encabezado h1.encabezado-titulo {{ font-size: 2.3rem !important; font-weight: 700; margin: 0; padding: 0 !important; line-height: 1.15; }}
.encabezado-bajada {{ color: {GRIS_TEXTO}; font-size: 1.05rem; margin: 0.45rem 0 0 0; }}

/* --- Micrófono de la barra del chat: más grande, para dedos y vista cansados --- */
[data-testid="stChatInput"] button {{ transform: scale(1.2); }}

/* --- Conversación ------------------------------------------------------- */
[data-testid="stChatMessage"] {{
    padding: 0.9rem 1rem;
    border-radius: 0.6rem;
    line-height: 1.6;
}}
[data-testid="stChatMessage"]:has(.marca-usuario) {{
    background: {CREMA};
}}
/* Los títulos que escriben los agentes dentro de la respuesta ("## A las 8:00")
   no pueden medir lo mismo que el título de la página. */
[data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2 {{
    font-size: 1.35rem !important; margin: 1rem 0 0.4rem 0; padding: 0;
}}
[data-testid="stChatMessage"] h3, [data-testid="stChatMessage"] h4 {{
    font-size: 1.15rem !important; margin: 0.8rem 0 0.3rem 0; padding: 0;
}}
[data-testid="stChatMessage"] hr {{ margin: 0.8rem 0; }}
.etiqueta-voz {{
    display: inline-block;
    font-size: 0.8rem;
    color: {GRIS_TEXTO};
    border: 1px solid {BEIGE};
    border-radius: 999px;
    padding: 0.05rem 0.6rem;
    margin-bottom: 0.35rem;
    background: #FFFFFF;
}}
.pie-respuesta {{ color: {GRIS_TEXTO}; font-size: 0.85rem; margin-top: 0.3rem; }}

/* Respuesta de emergencia: que no pase desapercibida. */
[class*="st-key-emergencia_"] {{
    border: 2px solid {ROJO_TEXTO} !important;
    background: #FFF5F5;
    border-radius: 0.6rem;
    padding: 0.4rem 0.9rem;
}}

/* --- Botones ------------------------------------------------------------ */
.stButton > button {{
    min-height: 3rem;
    font-size: 1rem;
    border-radius: 0.5rem;
}}
.stButton > button[kind="secondary"] {{ background: #FFFFFF; }}

/* Cuadro de texto del chat: más alto y con el borde rojo al escribir. */
[data-testid="stChatInput"] textarea {{ font-size: 1.05rem; min-height: 3rem; }}
[data-testid="stChatInput"]:focus-within {{ border-color: {ROJO_TEXTO}; }}

/* --- Barra lateral ------------------------------------------------------ */
[data-testid="stSidebar"] {{ border-right: 1px solid {BEIGE}; }}
[data-testid="stSidebar"] h3 {{ font-size: 1.1rem; }}

/* --- Pie ---------------------------------------------------------------- */
.pie-pagina {{
    margin-top: 2.5rem;
    padding-top: 0.8rem;
    border-top: 1px solid {BEIGE};
    color: {GRIS_TEXTO};
    font-size: 0.8rem;
    line-height: 1.5;
}}
</style>
"""


def encabezado_html(titulo: str, bajada: str) -> str:
    return (
        '<div class="encabezado">'
        f'<h1 class="encabezado-titulo">{titulo}</h1>'
        f'<p class="encabezado-bajada">{bajada}</p>'
        "</div>"
    )


def pie_html(texto: str) -> str:
    return f'<div class="pie-pagina">{texto}</div>'


@lru_cache(maxsize=32)
def avatar(inicial: str, fondo: str, texto: str = "#FFFFFF") -> Image.Image:
    """Un círculo con una inicial, en vez del robot y la silueta por defecto.

    Los íconos por defecto de Streamlit son los mismos que usan todos los
    chatbots; una inicial se lee como una persona o un servicio, no como "la
    IA". Se dibuja en memoria para no agregar archivos de imagen al repo.
    """
    lado = 96
    imagen = Image.new("RGBA", (lado, lado), (0, 0, 0, 0))
    dibujo = ImageDraw.Draw(imagen)
    dibujo.ellipse((0, 0, lado - 1, lado - 1), fill=fondo)
    try:
        fuente = ImageFont.truetype("DejaVuSans-Bold.ttf", 46)
    except OSError:
        fuente = ImageFont.load_default()
    caja = dibujo.textbbox((0, 0), inicial, font=fuente)
    ancho, alto = caja[2] - caja[0], caja[3] - caja[1]
    dibujo.text(((lado - ancho) / 2 - caja[0], (lado - alto) / 2 - caja[1]),
                inicial, fill=texto, font=fuente)
    return imagen

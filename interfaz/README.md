# Interfaz web

Interfaz de chat del sistema multiagente, en Streamlit, pensada para usarse
hablando.

## Ejecutar

Desde la **raíz del repositorio**, porque ahí está `.streamlit/config.toml`,
que trae el tema. Si se lanza desde otra carpeta, la app funciona pero sin los
colores ni la tipografía.

```bash
.venv/bin/streamlit run interfaz/app.py                          # solo local
.venv/bin/streamlit run interfaz/app.py --server.address 0.0.0.0 # accesible por IP
```

Queda en `http://localhost:8501` y, con `--server.address 0.0.0.0`, en la IP de
la máquina dentro de la red (por ejemplo `http://172.28.230.10:8501`).

Requiere VPN institucional activa: los modelos corren en los servidores de la
Universidad.

## La voz, de ida y de vuelta

**Hablarle.** El micrófono está en la misma barra del chat, al lado del cuadro
de texto, como en cualquier aplicación de mensajes. Se graba, se envía y la
barra se vacía sola. **Mientras el asistente responde, la barra queda
bloqueada**, así que un toque no corta la respuesta.

> **Por qué cambió (2026-09-30).** Antes había un micrófono aparte
> (`st.audio_input`) y fallaba al grabar. La segunda grabación se perdía,
> porque el micrófono se vaciaba cambiándole la clave a mitad del proceso. Y
> cualquier toque mientras el asistente trabajaba cortaba la respuesta.
> `st.chat_input(accept_audio=True, submit_mode="disable")` resuelve las dos
> cosas. Detalle en `entrada.py`.
>
> **Whisper se precarga en segundo plano al abrir la página.** Antes, la
> primera grabación esperaba ~37 s a que cargara el modelo.

Debajo, en "Enviar un audio ya grabado", hay dos vías más:

| Vía | Para qué |
|---|---|
| **Desde un archivo** | Un `.wav`, `.mp3`, `.m4a` u `.ogg` cualquiera |
| **Audios de prueba** | El audio de emergencia del corpus, limpio y con ruido: muestra el guardrail del ASR en dos clics |

**Escuchar la respuesta.** Cada respuesta trae un botón "Escuchar la
respuesta". Si la pregunta llegó por voz, la respuesta **se lee sola**: quien le
habla al asistente espera que le contesten hablando. Se apaga con "Leer las
respuestas en voz alta", en la barra lateral.

La lectura la hace el navegador (`speechSynthesis`), no el servidor: no depende
de los endpoints de la Universidad y funciona también por IP. Antes de leer, el
texto pasa por `texto_para_leer()`, que saca asteriscos, títulos, emojis y el
aviso de datos ficticios (que sigue visible). Ver `voz_salida.py`.

### Dos limitaciones que hay que saber

- **El micrófono solo funciona en `localhost` o HTTPS.** Es una regla del
  navegador (`getUserMedia` exige contexto seguro), no de la app. Por
  `http://<IP>:8501` el navegador bloquea la grabación. Para grabar desde otra
  máquina:
  ```bash
  ssh -L 8501:localhost:8501 usuario@172.28.230.10
  # después, en el navegador: http://localhost:8501
  ```
  Las otras dos vías de audio no tienen esa restricción. La página ya no lo
  avisa en pantalla (se quitó el 2026-09-23 para no llenar la vista
  principal de instrucciones técnicas); queda documentado acá.
- **La voz que lee depende del sistema de quien mira.** Windows, macOS y
  Android suelen traer voces en español; algunos Linux no traen ninguna, y ahí
  el botón no suena.

## Qué muestra

- **Quién usa el asistente**, en la barra lateral. Cambia la receta médica que
  lee el agente de medicación (`medicacion/datos/perfiles.json`). El valor
  inicial sale de la variable `PERFIL_ACTIVO`, si está definida.
- **Sugerencias** para empezar, mientras la conversación está vacía.
- **La respuesta de emergencia enmarcada en rojo**, para que se vea aunque no se
  lea nada más.
- **Bajo cada respuesta**, el tema y el tiempo que tardó, y plegado en "Cómo se
  llegó a esta respuesta": qué entendió Whisper del audio (incluso cuando se
  descartó, que es el punto entero del guardrail), por qué el orquestador eligió
  ese tema, y el recorrido por el grafo con los nombres de los nodos, para
  cruzarlo con el diagrama o con LangSmith.
- **Mientras trabaja**, el paso en que va ("Revisando su receta médica",
  "Buscando en el recetario"...). Una consulta con RAG tarda entre 6 y 30
  segundos; sin eso la pantalla parece colgada.
- **Información técnica**, plegada en la barra lateral: con qué modelo se está
  respondiendo, si el servidor cambió de modelo, y el enlace a LangSmith.

## Identidad visual

Paleta y tipografía tomadas de la hoja de estilos pública de usfq.edu.ec
(2026-09-23): rojo `#ED1C24`, casi negro `#231F20`, crema `#FAF3E9`, beige
`#D8D4CB`; Baskerville para títulos y Helvetica para texto.

**El rojo institucional no se usa para texto ni botones**: sobre blanco da un
contraste de 4.38:1, por debajo del 4.5:1 de WCAG AA. Para eso se usa `#B5121B`
(6.85:1). En una interfaz para adultos mayores, el contraste no es un detalle.

No se usa el logotipo de la Universidad: es un prototipo de tesis, no un
producto institucional.

Textos en **usted**, igual que los agentes, sin emojis y sin fórmulas de
chatbot. Los avatares son una inicial en un círculo (la de la persona elegida y
una "A" para el asistente), en vez del robot y la silueta por defecto. Los
emojis que a veces escriben los agentes se quitan al mostrar la respuesta.

## Organización

| Archivo | Qué hace |
|---|---|
| `app.py` | El flujo de la página |
| `componentes.py` | Cómo se dibuja cada turno y su detalle |
| `voz_salida.py` | Lectura en voz alta |
| `estilo.py` | CSS, paleta y avatares |
| `textos.py` | Todo el texto visible, en un solo lugar |
| `../.streamlit/config.toml` | Tema: colores, tipografía, tamaño de letra |

Este paquete solo presenta. No clasifica, no decide y no habla con los modelos:
todo eso vive en `orquestacion_langgraph/` y `medicacion/`. El único punto de
contacto con el sistema es `procesar_consulta_en_vivo()`, en
`orquestacion_langgraph/grafo.py`.

## Pruebas

```bash
python -m pruebas.prueba_interfaz_voz   # 8, la limpieza del texto que se lee
```

El resto se verificó a mano y con capturas de un navegador sin pantalla: texto,
voz limpia (EMERGENCY en 5 s), voz a 0 dB (se pide repetir) y vista de celular.
La lectura en voz alta en sí **no** se pudo escuchar desde el servidor, que no
tiene parlantes ni voces instaladas: hay que probarla desde un navegador real.

## Estado

Prototipo académico. No tiene autenticación, control de acceso ni persistencia:
el historial vive en la sesión del navegador y se pierde al recargar.

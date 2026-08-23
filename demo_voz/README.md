# App de grabación de corpus y demo por voz

Página web local con dos modos: grabar el corpus de audio y hacer la demo en
vivo de transcripción.

```bash
python -m demo_voz.servidor --precargar
```

Después abrir **http://localhost:8000**.

## Por qué una app web y no un script

El micrófono está en la máquina de quien graba, no en el servidor. Este equipo
(DGX-H200) **no tiene tarjeta de captura de audio**: `/dev/snd` solo trae `seq`
y `timer`. El navegador captura el audio del lado del cliente y lo sube por
HTTP; Whisper corre acá, en la GPU.

```
[laptop: navegador]              [DGX-H200]
  MediaRecorder                    Whisper large-v3 (GPU)
       |                                  ^
       +------ POST audio WebM -----------+
```

**El micrófono solo funciona por `localhost` o HTTPS.** Trabajando por VSCode
Remote, el port forwarding expone el servidor como localhost en la máquina
cliente, así que funciona. Abrirlo por la IP directa
(`172.28.230.10:8000`) haría que el navegador bloquee el micrófono.

## Modo corpus

Muestra una frase del `dataset.csv`, se lee en voz alta, se graba y se guarda ya
etiquetada con su intención y su transcripción correcta.

- **Recorre las intenciones de forma intercalada**, no en bloque: elige siempre
  la intención con menos grabaciones. Así el corpus queda balanceado desde el
  principio y con 50 frases ya se puede medir accuracy de ruteo, en vez de tener
  50 frases de una sola intención.
- **El progreso se guarda solo.** Se puede cerrar y seguir después.
- **Regrabar una frase la reemplaza**, no la duplica.
- Muestra qué entendió Whisper después de cada grabación — sirve para detectar
  al toque un micrófono mal configurado, no como verdad de referencia (la
  referencia es el texto del dataset).
- Avisa en rojo si el VAD **no detectó voz**: casi siempre significa micrófono
  equivocado o silenciado.

Se desactivan a propósito la cancelación de eco, la supresión de ruido y el
control automático de ganancia del navegador: alterarían la señal y el corpus
dejaría de servir como línea base limpia para la matriz de ruido.

## Modo demo

Se habla y muestra la transcripción. Es lo que se le enseña a los tutores.

**Todavía no está conectado al grafo de agentes**, a propósito: Cristian pidió
no integrar Whisper al sistema multiagente en esta etapa. Conectarlo al
Orchestrator es un cambio chico cuando llegue esa fase.

## Archivos

- `servidor.py` — `http.server` de la librería estándar. Sin FastAPI: es un
  prototipo local de un solo usuario y no vale sumar una dependencia. **No es un
  servidor apto para producción**: sin autenticación, sin HTTPS propio, sin
  límites de concurrencia.
- `corpus.py` — qué falta grabar, dónde se guarda, cómo se lleva el índice. El
  índice es un CSV para poder abrirlo en Excel y revisarlo a mano; es formato de
  prototipo, no una base de datos.
- `static/index.html` — la interfaz, en un solo archivo sin dependencias
  externas.

El audio del navegador llega en WebM/Opus y se convierte a WAV mono 16 kHz con
PyAV (que vino con `faster-whisper`; el sistema no tiene ffmpeg instalado). Se
guarda en WAV porque la matriz de ruido trabaja sobre muestras PCM de 16 bits.

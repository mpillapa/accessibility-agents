# Reconocimiento de voz (ASR) con Whisper

Entrada por voz del asistente. **Módulo aislado**: no importa nada de
`orquestacion_langgraph/` ni de `rag/`. Cristian pidió explícitamente trabajar
Whisper por separado antes de integrarlo al sistema multiagente, y mantener esa
frontera permite medir el ASR sin que se mezcle con los errores del resto.

## Por qué corre local y no como endpoint

Todo el resto del stack (chat, embeddings, OCR) usa endpoints vLLM
OpenAI-compatible en el servidor de la Universidad. Whisper no, porque **no hay
endpoint de ASR**: se escaneó el rango de puertos 12550–12575 el 2026-08-19 y
solo responden chat (12559), embeddings (12556), OCR (12560) y un DeepSeek
(12555).

Es una desviación consciente de la estandarización que pidieron los tutores.
Coincide con lo que dijo Cristian ("te bajas el whisper"), pero conviene
confirmarla.

## Módulos

- `config.py` — modelo, device, idioma y rutas del corpus. Todo por `.env`.
- `transcribir.py` — el wrapper. Carga el modelo una sola vez (~37 s con
  `large-v3` en GPU) y lo reutiliza.
- `ruido.py` — genera la matriz de ruido con SNR controlado.

```bash
python -m asr.transcribir ruta/al/audio.wav
```

## El hallazgo que condiciona el diseño

**Whisper produjo texto en el 100% de 18 audios sin voz** (silencio digital
absoluto incluido), reportando siempre `probabilidad_idioma = 1.00`. Lo que
inventa es residuo de su entrenamiento: *"Gracias por ver el video"*, típico de
subtítulos de YouTube.

El filtro VAD lo elimina por completo (0/18). Por eso:

- `transcribir()` trae `usar_vad=True` por defecto.
- El resultado incluye un campo **`sin_voz`** explícito. Quien consuma una
  transcripción tiene que mirar ese campo, no solo el texto.

Reproducir: `python -m pruebas.evaluar_alucinacion_asr --json salida.json`

**Pendiente de medir:** el VAD también puede descartar voz real débil — una
persona mayor hablando bajo, o desde otra habitación. Esa tasa de falsos
negativos solo se puede medir con el corpus grabado, y es la medición más
importante que queda: si el VAD elimina el 100% de las alucinaciones pero se
come el 15% de las frases reales, la decisión deja de ser obvia.

## La matriz de ruido

En vez de grabar la misma frase en la cocina, en la calle y en silencio —donde
no se controla nada y los resultados no son comparables entre sí— se graba
**una vez en silencio** y se le superpone ruido a niveles medidos. Así la única
variable que cambia entre condiciones es el ruido.

| SNR | Equivale a |
|---|---|
| sin ruido | condición de control |
| 20 dB | ambiente tranquilo |
| 10 dB | televisión de fondo |
| 5 dB | cocina en uso, calle con tráfico |
| 0 dB | ruido tan fuerte como la voz |

Dos tipos de ruido sintético: **blanco** (siseante) y **rosa** (la energía decae
con la frecuencia, se parece más a un ambiente real). Ambos con semilla fija,
así que dos corridas generan exactamente los mismos archivos.

Verificado en `pruebas/prueba_ruido.py`: el SNR medido coincide con el pedido
con error menor a 0.6 dB, de −5 a 20 dB. Es la propiedad que importa — si esa
matemática estuviera mal, toda la evaluación mediría una condición distinta de
la que dice medir.

## Corpus

Se graba con la app de `../demo_voz/`. Ver el README de ese paquete.

Los audios van a `corpus_audio/`, **fuera de git** (415 WAV limpios son ~26 MB,
y con la matriz de ruido pasarían de 250 MB).

**Limitación de método, a declarar en el paper:** el corpus lo graba Manuel, que
no pertenece a la población objetivo. Voz clara, sin temblor, sin prótesis
dental. Los resultados de WER van a ser optimistas respecto de lo que daría un
adulto mayor real. Si más adelante se consiguen 20–30 frases grabadas por una
persona mayor, el campo `hablante` del índice permite comparar los dos grupos
sin rehacer nada — y ese contraste sería un resultado en sí mismo.

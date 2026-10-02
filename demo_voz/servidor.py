# Servidor local de la app de grabación de corpus y demo por voz. Whisper en GPU.
# Prototipo de un solo usuario, no apto para producción (demo_voz/README.md).
#
# Uso (desde la raíz del repo):
#   python -m demo_voz.servidor
#   python -m demo_voz.servidor --puerto 8080
#
# Abrir http://localhost:8000: el micrófono solo funciona por localhost o HTTPS.

import argparse
import json
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from asr.config import CORPUS_GRABACIONES_DIR
from asr.transcribir import transcribir
from demo_voz import corpus

ESTATICOS = Path(__file__).parent / "static"

# Para que un error del navegador no llene el disco.
MAXIMO_BYTES_AUDIO = 25 * 1024 * 1024


def convertir_a_wav_16k(datos: bytes, extension_origen: str = "webm") -> Path:
    """Convierte el audio del navegador (WebM/Opus, MP4/AAC) a WAV mono 16 kHz,
    el formato que necesita la matriz de ruido."""
    with tempfile.NamedTemporaryFile(suffix=f".{extension_origen}", delete=False) as f:
        f.write(datos)
        origen = Path(f.name)

    destino = origen.with_suffix(".wav")
    # Sin ffmpeg en el sistema: se usa PyAV, que trae faster-whisper.
    import av

    contenedor_entrada = av.open(str(origen))
    contenedor_salida = av.open(str(destino), mode="w")
    flujo_salida = contenedor_salida.add_stream("pcm_s16le", rate=16000, layout="mono")

    for cuadro in contenedor_entrada.decode(audio=0):
        cuadro.pts = None
        for paquete in flujo_salida.encode(cuadro):
            contenedor_salida.mux(paquete)
    for paquete in flujo_salida.encode(None):
        contenedor_salida.mux(paquete)

    contenedor_salida.close()
    contenedor_entrada.close()
    origen.unlink(missing_ok=True)
    return destino


def duracion_wav(ruta: Path) -> float:
    import wave

    with wave.open(str(ruta), "rb") as w:
        return round(w.getnframes() / w.getframerate(), 2)


class Manejador(BaseHTTPRequestHandler):
    def log_message(self, formato, *args):
        # Solo se registran los POST.
        if "POST" in (args[0] if args else ""):
            super().log_message(formato, *args)

    def _responder_json(self, datos, codigo=200):
        cuerpo = json.dumps(datos, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def _responder_archivo(self, ruta: Path, tipo: str):
        if not ruta.exists():
            self.send_error(404)
            return
        cuerpo = ruta.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def _leer_cuerpo(self) -> bytes | None:
        longitud = int(self.headers.get("Content-Length", 0))
        if longitud <= 0:
            self._responder_json({"error": "sin audio"}, 400)
            return None
        if longitud > MAXIMO_BYTES_AUDIO:
            self._responder_json(
                {"error": f"audio demasiado grande ({longitud} bytes)"}, 413
            )
            return None
        return self.rfile.read(longitud)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._responder_archivo(ESTATICOS / "index.html", "text/html; charset=utf-8")
        elif self.path == "/api/progreso":
            self._responder_json(corpus.progreso())
        elif self.path == "/api/siguiente":
            frase = corpus.siguiente_pendiente()
            self._responder_json(frase or {"fin": True})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.startswith("/api/grabar"):
            self._grabar()
        elif self.path == "/api/transcribir":
            self._transcribir_suelto()
        else:
            self.send_error(404)

    def _grabar(self):
        """Guarda una grabación del corpus. La transcripción se devuelve solo para
        verificar el audio; la referencia es el texto del dataset."""
        from urllib.parse import parse_qs, urlparse

        parametros = parse_qs(urlparse(self.path).query)
        id_frase = (parametros.get("id") or [""])[0]
        hablante = (parametros.get("hablante") or ["manuel"])[0]
        if not id_frase:
            self._responder_json({"error": "falta el parámetro id"}, 400)
            return

        frases = {f["id_frase"]: f for f in corpus.cargar_frases()}
        frase = frases.get(id_frase)
        if not frase:
            self._responder_json({"error": f"id desconocido: {id_frase}"}, 404)
            return

        datos = self._leer_cuerpo()
        if datos is None:
            return

        try:
            temporal = convertir_a_wav_16k(datos)
        except Exception as e:
            self._responder_json({"error": f"no se pudo convertir el audio: {e}"}, 400)
            return

        CORPUS_GRABACIONES_DIR.mkdir(parents=True, exist_ok=True)
        destino = corpus.ruta_audio(id_frase)
        destino.write_bytes(temporal.read_bytes())
        temporal.unlink(missing_ok=True)

        resultado = transcribir(destino)
        corpus.registrar({
            "id_frase": id_frase,
            "archivo": destino.name,
            "intencion": frase["intencion"],
            "texto": frase["texto"],
            "hablante": hablante,
            "duracion_s": duracion_wav(destino),
        })

        self._responder_json({
            "guardado": destino.name,
            "duracion_s": duracion_wav(destino),
            "texto_esperado": frase["texto"],
            "texto_transcrito": resultado.texto,
            "sin_voz": resultado.sin_voz,
            "progreso": corpus.progreso(),
        })

    def _transcribir_suelto(self):
        """Modo demo: transcribe sin guardar nada en el corpus."""
        datos = self._leer_cuerpo()
        if datos is None:
            return
        try:
            temporal = convertir_a_wav_16k(datos)
        except Exception as e:
            self._responder_json({"error": f"no se pudo convertir el audio: {e}"}, 400)
            return

        resultado = transcribir(temporal)
        duracion = duracion_wav(temporal)
        temporal.unlink(missing_ok=True)

        self._responder_json({
            "texto": resultado.texto,
            "sin_voz": resultado.sin_voz,
            "duracion_s": duracion,
            "tiempo_transcripcion_s": resultado.tiempo_transcripcion_s,
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--puerto", type=int, default=8000)
    parser.add_argument("--precargar", action="store_true",
                        help="carga Whisper al arrancar en vez de en la primera petición")
    args = parser.parse_args()

    if args.precargar:
        from asr.transcribir import obtener_modelo
        obtener_modelo()

    p = corpus.progreso()
    print(f"Corpus: {p['grabadas']}/{p['total']} frases grabadas")
    print()
    print(f"  Abrir en el navegador:  http://localhost:{args.puerto}")
    print()
    print("  El micrófono solo funciona por localhost o HTTPS. Si trabajás por")
    print("  VSCode Remote, el port forwarding ya lo expone como localhost.")
    print("  Ctrl+C para detener.")
    print()

    ThreadingHTTPServer(("0.0.0.0", args.puerto), Manejador).serve_forever()


if __name__ == "__main__":
    main()

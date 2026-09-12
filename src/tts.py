import os
import asyncio
import subprocess
import shutil
import sys
import tempfile
import threading
import wave
from contextlib import contextmanager
from pathlib import Path

from src.logger import get_logger
from src.config import PIPER_BINARY, PIPER_VOICE_MODEL, ALLOW_CLOUD_TTS_FALLBACK

logger = get_logger("TTS")

# Voz opcional para frases públicas, solo después de agotar los motores locales
# y cuando el operador haya habilitado ALLOW_CLOUD_TTS_FALLBACK explícitamente.
EDGE_VOICE = "es-AR-ElenaNeural"
EDGE_TTS_TIMEOUT_SECONDS = 8  # evita colgar el turno de voz si la red está lenta/caída


@contextmanager
def _audio_temporal(suffix: str):
    """Cada respuesta tiene su archivo privado (0600), eliminado aun si falla."""
    with tempfile.NamedTemporaryFile(prefix="niri_response_", suffix=suffix,
                                     delete=False) as audio:
        path = audio.name
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("No pude limpiar el audio temporal (%s).", type(exc).__name__)


def _buscar_voz_piper() -> Path | None:
    """
    Devuelve la voz de Piper a usar, o None si no hay ninguna.

    Si la ruta configurada no existe, se acepta cualquier `.onnx` que haya en
    `models/piper/`: bajar una voz es un paso manual, y obligar además a que el
    archivo se llame exactamente como dice `config.py` sería una trampa tonta
    para el que la baje con su nombre original.
    """
    if PIPER_VOICE_MODEL.exists():
        return PIPER_VOICE_MODEL
    carpeta = PIPER_VOICE_MODEL.parent
    if carpeta.is_dir():
        for candidata in sorted(carpeta.glob("*.onnx")):
            logger.info(f"Voz de Piper encontrada por descubrimiento: {candidata.name}")
            return candidata
    return None


def _buscar_binario_piper() -> str | None:
    """
    Ubica el ejecutable de Piper.

    Además del PATH mira al lado del intérprete que está corriendo: el servicio
    arranca con `.venv/bin/python` sin activar el venv, así que `piper` —que
    `pip install piper-tts` deja en `.venv/bin/`— no aparece en el PATH.
    """
    encontrado = shutil.which(PIPER_BINARY)
    if encontrado:
        return encontrado
    vecino = Path(sys.executable).parent / "piper"
    return str(vecino) if vecino.exists() else None


class TextToSpeech:
    """
    Piper → espeak-ng; edge-tts es un respaldo opcional para texto público.

    Las llamadas existentes son privadas por defecto, incluso si se habilita
    la nube. Nunca marcar como público texto que derive de datos del usuario.
    """

    def __init__(self):
        self._speak_lock = threading.Lock()
        # Verificar que tenemos un reproductor de audio disponible
        self.player = None
        for candidate in ("mpv", "ffplay"):
            if shutil.which(candidate):
                self.player = candidate
                break

        if self.player is None:
            logger.warning("No se encontró mpv ni ffplay. Solo espeak-ng podrá reproducir audio.")

        # Piper: motor local principal. Se prefiere la API en proceso sobre el
        # binario porque el costo de Piper es cargar el modelo, no sintetizar:
        # medido, por subproceso son ~826 ms por frase (el .onnx de 63 MB se
        # relee cada vez) contra 62 ms teniéndolo en memoria, con una carga
        # única de 655 ms al arrancar.
        self.piper_voz = None
        self.piper_bin = None
        voz = _buscar_voz_piper()
        self.piper_voz_path = voz
        if voz is not None:
            self.piper_bin = _buscar_binario_piper()
            try:
                from piper import PiperVoice  # import perezoso: es opcional
                self.piper_voz = PiperVoice.load(str(voz))
                logger.info(f"Piper cargado en memoria ({voz.name}).")
            except Exception as exc:
                logger.warning("No pude cargar Piper en proceso (%s); pruebo con el binario.",
                               type(exc).__name__)
        self.has_piper = self.piper_voz is not None or self.piper_bin is not None
        if not self.has_piper:
            logger.warning(
                f"Piper no disponible todavía (voz encontrada={voz is not None} en "
                f"{PIPER_VOICE_MODEL.parent}). Para TTS 100% local: `pip install piper-tts` "
                "y descargar una voz ES en models/piper/ (ver README.md). "
                "Usando fallback mientras tanto."
            )

        # espeak-ng: último recurso, siempre local
        self.has_espeak = shutil.which("espeak-ng") is not None
        if not self.has_espeak:
            logger.warning("espeak-ng no instalado. El fallback offline final no estará disponible.")

        logger.info(
            f"TTS inicializado (piper={'en proceso' if self.piper_voz is not None else self.has_piper}, "
            f"cloud_fallback={'activado' if ALLOW_CLOUD_TTS_FALLBACK else 'desactivado'}, "
            f"reproductor={self.player}, espeak={self.has_espeak})"
        )

    # --- Piper (local, primario) ---------------------------------------
    def _speak_piper(self, text: str) -> bool:
        if not self.has_piper or not self.player:
            return False
        if self.piper_voz is not None and self._speak_piper_en_proceso(text):
            return True
        return self._speak_piper_subproceso(text) if self.piper_bin else False

    def _speak_piper_en_proceso(self, text: str) -> bool:
        """Sintetiza con el modelo ya cargado en memoria (~62 ms por frase)."""
        try:
            with _audio_temporal(".wav") as audio_path:
                with wave.open(audio_path, "wb") as salida:
                    self.piper_voz.synthesize_wav(text, salida)
                return self._play_file(audio_path)
        except Exception as exc:
            logger.warning("Error sintetizando con Piper en proceso (%s).", type(exc).__name__)
            return False

    def _speak_piper_subproceso(self, text: str) -> bool:
        """Respaldo por binario: recarga el modelo en cada llamada, ~826 ms."""
        try:
            with _audio_temporal(".wav") as audio_path:
                proc = subprocess.run(
                    [self.piper_bin, "--model", str(self.piper_voz_path), "--output_file", audio_path],
                    input=text.encode("utf-8"),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
                if proc.returncode != 0 or os.path.getsize(audio_path) == 0:
                    logger.warning("Piper falló (code=%s).", proc.returncode)
                    return False
                return self._play_file(audio_path)
        except Exception as exc:
            logger.warning("Error ejecutando Piper (%s).", type(exc).__name__)
            return False

    # --- edge-tts (nube, opcional) --------------------------------------
    async def _generate_edge_audio(self, text: str, audio_path: str) -> bool:
        """Genera el audio usando edge-tts y lo guarda en un archivo temporal."""
        try:
            import edge_tts  # import perezoso: solo para texto público con opt-in
            communicate = edge_tts.Communicate(text, EDGE_VOICE)
            await asyncio.wait_for(communicate.save(audio_path), timeout=EDGE_TTS_TIMEOUT_SECONDS)
            return os.path.getsize(audio_path) > 0
        except Exception as exc:
            logger.warning("Error con edge-tts (%s).", type(exc).__name__)
            return False

    # --- espeak-ng (local, último recurso) -------------------------------
    def _speak_offline(self, text: str) -> bool:
        """Usa espeak-ng directamente como fallback offline."""
        try:
            subprocess.run(
                ["espeak-ng", "-v", "es-419", "-s", "160", "--stdin"],
                input=text.encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True, timeout=30
            )
            return True
        except Exception as exc:
            logger.error("Error con espeak-ng (%s).", type(exc).__name__)
            return False

    # --- Reproducción ------------------------------------------------------
    def _play_file(self, filepath: str) -> bool:
        """Reproduce un WAV/MP3 temporal usando mpv o ffplay de forma bloqueante."""
        try:
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0 or not self.player:
                return False
            if self.player == "mpv":
                subprocess.run(
                    ["mpv", "--no-video", "--really-quiet", filepath],
                    check=True, timeout=30
                )
            elif self.player == "ffplay":
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filepath],
                    check=True, timeout=30
                )
            else:
                return False
            return True
        except Exception as exc:
            logger.error("Error reproduciendo audio (%s).", type(exc).__name__)
            return False

    def speak(self, text: str, *, public: bool = False) -> bool:
        """
        Habla en local y devuelve si pudo reproducir la respuesta completa.

        ``public=True`` solo corresponde a frases constantes propias del
        asistente, nunca a nombres, rutas, notas, transcripciones ni contenido
        de archivos. Aun así la nube requiere opt-in y fallos de ambos motores
        locales. El texto privado nunca sale como consecuencia de un fallo.
        """
        if not isinstance(text, str) or not text.strip():
            return False
        if type(public) is not bool:
            raise TypeError("public debe ser un booleano explícito")
        logger.info("Respuesta TTS (%s, %d caracteres).", "pública" if public else "privada", len(text))

        with self._speak_lock:
            if self._speak_piper(text):
                return True

            if self.has_espeak:
                logger.info("Usando espeak-ng como respaldo offline.")
                if self._speak_offline(text):
                    return True

            if ALLOW_CLOUD_TTS_FALLBACK and public and self.player:
                try:
                    with _audio_temporal(".mp3") as audio_path:
                        if asyncio.run(self._generate_edge_audio(text, audio_path)):
                            if self._play_file(audio_path):
                                return True
                except Exception as exc:
                    logger.warning("No pude completar el respaldo de nube (%s).", type(exc).__name__)

            logger.error("No se pudo reproducir la respuesta con los motores permitidos.")
            return False

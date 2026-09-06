import os
import asyncio
import subprocess
import shutil
import sys
import wave
from pathlib import Path

from src.logger import get_logger
from src.config import PIPER_BINARY, PIPER_VOICE_MODEL, ALLOW_CLOUD_TTS_FALLBACK

logger = get_logger("TTS")

# Voz neuronal de Edge: Elena (Argentina), voz de mujer, para sonar más
# amigable. ALLOW_CLOUD_TTS_FALLBACK=true por defecto, así que esta es la
# voz preferida siempre que haya internet; si se apaga el flag, o no hay
# conexión, el asistente cae a Piper -> espeak-ng (100% local).
EDGE_VOICE = "es-AR-ElenaNeural"
EDGE_TTS_TIMEOUT_SECONDS = 8  # evita colgar el turno de voz si la red está lenta/caída
TEMP_WAV_FILE = "/tmp/niri_response.wav"
TEMP_MP3_FILE = "/tmp/niri_response.mp3"


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
    Cadena de motores TTS, de más a menos preferido:
      1. edge-tts (nube, voz de mujer Elena, la más amigable) — SOLO si
         ALLOW_CLOUD_TTS_FALLBACK=true y hay internet; si falla, sigue con:
      2. Piper (100% local, calidad neuronal) — requiere el binario `piper`
         y una voz .onnx en models/piper/. Ver README.md.
      3. espeak-ng (100% local, calidad robótica) — último recurso siempre.
    """

    def __init__(self):
        # Verificar que tenemos un reproductor de audio disponible
        self.player = None
        for candidate in ("mpv", "ffplay"):
            if shutil.which(candidate):
                self.player = candidate
                break

        if self.player is None:
            logger.warning("No se encontró mpv ni ffplay. Ningún motor TTS podrá reproducir audio.")

        # Piper: motor local principal. Se prefiere la API en proceso sobre el
        # binario porque el costo de Piper es cargar el modelo, no sintetizar:
        # medido, por subproceso son ~826 ms por frase (el .onnx de 63 MB se
        # relee cada vez) contra 62 ms teniéndolo en memoria, con una carga
        # única de 655 ms al arrancar.
        self.piper_voz = None
        self.piper_bin = None
        voz = _buscar_voz_piper()
        if voz is not None:
            try:
                from piper import PiperVoice  # import perezoso: es opcional
                self.piper_voz = PiperVoice.load(str(voz))
                logger.info(f"Piper cargado en memoria ({voz.name}).")
            except Exception as e:
                logger.warning(f"No pude cargar Piper en proceso ({e}); pruebo con el binario.")
                self.piper_bin = _buscar_binario_piper()
                self.piper_voz_path = voz
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
        if self.piper_voz is not None:
            return self._speak_piper_en_proceso(text)
        return self._speak_piper_subproceso(text)

    def _speak_piper_en_proceso(self, text: str) -> bool:
        """Sintetiza con el modelo ya cargado en memoria (~62 ms por frase)."""
        try:
            with wave.open(TEMP_WAV_FILE, "wb") as salida:
                self.piper_voz.synthesize_wav(text, salida)
        except Exception as e:
            logger.warning(f"Error sintetizando con Piper en proceso: {e}")
            return False
        self._play_file(TEMP_WAV_FILE)
        return True

    def _speak_piper_subproceso(self, text: str) -> bool:
        """Respaldo por binario: recarga el modelo en cada llamada, ~826 ms."""
        try:
            proc = subprocess.run(
                [self.piper_bin, "--model", str(self.piper_voz_path), "--output_file", TEMP_WAV_FILE],
                input=text.encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            if proc.returncode != 0 or not os.path.exists(TEMP_WAV_FILE):
                stderr = proc.stderr.decode("utf-8", errors="ignore")[:300] if proc.stderr else ""
                logger.warning(f"Piper falló (code={proc.returncode}): {stderr}")
                return False
            self._play_file(TEMP_WAV_FILE)
            return True
        except Exception as e:
            logger.warning(f"Error ejecutando Piper: {e}")
            return False

    # --- edge-tts (nube, opcional) --------------------------------------
    async def _generate_edge_audio(self, text: str) -> bool:
        """Genera el audio usando edge-tts y lo guarda en un archivo temporal."""
        try:
            import edge_tts  # import perezoso: solo si el fallback de nube está habilitado
            communicate = edge_tts.Communicate(text, EDGE_VOICE)
            await asyncio.wait_for(communicate.save(TEMP_MP3_FILE), timeout=EDGE_TTS_TIMEOUT_SECONDS)
            return True
        except Exception as e:
            logger.warning(f"Error con edge-tts (¿sin internet o paquete no instalado?): {e}")
            return False

    # --- espeak-ng (local, último recurso) -------------------------------
    def _speak_offline(self, text: str):
        """Usa espeak-ng directamente como fallback offline."""
        try:
            subprocess.run(
                ["espeak-ng", "-v", "es-419", "-s", "160", text],
                check=True, timeout=30
            )
        except Exception as e:
            logger.error(f"Error con espeak-ng: {e}")

    # --- Reproducción ------------------------------------------------------
    def _play_file(self, filepath: str):
        """Reproduce un WAV/MP3 temporal usando mpv o ffplay de forma bloqueante."""
        if not os.path.exists(filepath) or not self.player:
            return

        try:
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
        except Exception as e:
            logger.error(f"Error reproduciendo audio: {e}")
        finally:
            try:
                os.remove(filepath)
            except OSError:
                pass

    def speak(self, text: str):
        """
        Método principal para hablar. Si ALLOW_CLOUD_TTS_FALLBACK está
        habilitado, intenta primero edge-tts (nube, voz de mujer Elena) —
        solo funciona con internet; si no hay conexión o el flag está
        apagado, usa Piper (local) y, si tampoco está disponible, espeak-ng
        (local, último recurso). Bloquea hasta terminar de hablar.
        """
        logger.info(f"Niri dice: '{text}'")

        if ALLOW_CLOUD_TTS_FALLBACK and self.player:
            success = asyncio.run(self._generate_edge_audio(text))
            if success:
                self._play_file(TEMP_MP3_FILE)
                return

        if self._speak_piper(text):
            return

        if self.has_espeak:
            logger.info("Usando espeak-ng como respaldo offline.")
            self._speak_offline(text)
        else:
            logger.error("No hay motor TTS disponible. Instalá Piper (recomendado) o espeak-ng.")

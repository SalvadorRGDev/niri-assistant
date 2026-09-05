import time
import wave
import datetime
import numpy as np
import sounddevice as sd
from contextlib import nullcontext
from enum import Enum
from pathlib import Path
from typing import Optional

from src.audio import AudioStream, AudioDeviceLost
from src.wake_word import WakeWordDetector
from src.vad import VoiceActivityDetector
from src.stt import SpeechToText
from src.nlu import NLU
from src.executor import Executor
from src.tts import TextToSpeech
from src.confirm import interpret_confirmation
from src.paths import parse_location_speech, speakable_path, DEFAULT_ROOT
from src.actions.dispatch import dispatch as dispatch_non_file_action, NON_FILE_ACTIONS
from src.actions.time_actions import check_due_timers
from src.metrics import TurnoMetricas
from src.logger import get_logger
from src.config import RECORDINGS_DIR, BEEP_ON_WAKE_WORD, SAMPLE_RATE, CHANNELS

logger = get_logger("MainLoop")

VAD_CHUNK_SAMPLES = 512  # 32ms a 16kHz, tamaño fijo que exige Silero VAD

# Acciones sobre un archivo/carpeta EXISTENTE: hay que ubicarlo (en vez de
# preguntar dónde crearlo).
LOCATE_EXISTING_ACTIONS = {"eliminar", "mover", "leer"}
CREATE_ACTIONS = {"crear_carpeta", "crear_archivo"}


class State(Enum):
    IDLE = 1
    LISTENING = 2


def play_beep():
    """Generates a short beep sound."""
    if not BEEP_ON_WAKE_WORD:
        return
    duration = 0.2
    frequency = 880.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    beep = 0.5 * np.sin(2 * np.pi * frequency * t)
    try:
        sd.play(beep, samplerate=SAMPLE_RATE)
        sd.wait()
    except Exception as e:
        logger.warning(f"Could not play beep: {e}")


class MainLoop:
    def __init__(self):
        self.audio = AudioStream()
        self.ww = WakeWordDetector()
        self.vad = VoiceActivityDetector()
        self.stt = SpeechToText()
        self.nlu = NLU()
        self.executor = Executor()
        self.tts = TextToSpeech()
        self.state = State.IDLE
        self.running = False
        # Métricas del turno en curso (None mientras se espera el wake word).
        self.turno: Optional[TurnoMetricas] = None

    def save_utterance(self, audio_data: np.ndarray):
        """Saves the recorded utterance to a WAV file."""
        if len(audio_data) == 0:
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = RECORDINGS_DIR / f"utterance_{timestamp}.wav"

        try:
            # recordings/ no se versiona (contiene voz real), así que en un clon
            # limpio el directorio no existe todavía.
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            with wave.open(str(filepath), 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_data.tobytes())
            logger.info(f"Utterance saved to {filepath} ({(len(audio_data) / SAMPLE_RATE):.2f}s)")
        except Exception as e:
            logger.error(f"Failed to save utterance: {e}")

    def capture_utterance(self, max_wait_seconds: float = 3.0, min_speech_seconds: float = 0.5) -> Optional[np.ndarray]:
        """
        Graba desde el micrófono usando VAD hasta detectar el corte de
        silencio configurado, o hasta `max_wait_seconds` si el usuario nunca
        empieza a hablar. Retorna el audio capturado, o None si no hubo voz
        (silencio total) o la utterance fue demasiado corta para ser útil.

        Se usa para la orden principal (tras el wake word) y para cada
        respuesta hablada de los diálogos de ubicación/confirmación.
        """
        self.vad.reset_states()
        current_utterance = []
        silent_chunks = 0
        silent_chunks_before_speech = 0
        has_spoken = False
        max_prob_seen = 0.0
        max_wait_chunks = int(max_wait_seconds * SAMPLE_RATE / VAD_CHUNK_SAMPLES)

        while True:
            chunk = self.audio.read_chunk(VAD_CHUNK_SAMPLES)
            current_utterance.append(chunk)

            prob = self.vad.process_chunk(chunk)
            if prob > max_prob_seen:
                max_prob_seen = prob

            if prob >= self.vad.threshold:
                has_spoken = True
                silent_chunks = 0
            else:
                if has_spoken:
                    silent_chunks += 1
                else:
                    silent_chunks_before_speech += 1

            if not has_spoken and silent_chunks_before_speech >= max_wait_chunks:
                logger.info(f"No speech detected. Max VAD prob was: {max_prob_seen:.3f}.")
                return None

            if has_spoken and silent_chunks >= self.vad.max_silent_chunks:
                utterance_audio = np.concatenate(current_utterance)
                if len(utterance_audio) > int(SAMPLE_RATE * min_speech_seconds):
                    return utterance_audio
                logger.info("Utterance was too short, discarding.")
                return None

    def _reabrir_audio(self):
        """
        Cierra la entrada de audio y vuelve a elegir dispositivo.

        Se llama cuando el micrófono deja de entregar datos en pleno uso —el caso
        real es desenchufar el USB—. Vuelve a IDLE y descarta el turno en curso:
        cualquier orden a medio capturar quedó incompleta, y actuar sobre media
        orden es peor que perderla.
        """
        self.turno = None
        self.state = State.IDLE
        self.audio.stop()
        self.audio.reset_buffers()
        self.audio.start()

    def _medir(self, nombre: str):
        """
        Contexto que suma el tiempo del bloque a la etapa `nombre` del turno.

        Fuera de un turno no mide nada: el aviso de un temporizador vencido
        también pasa por _say(), pero ocurre en IDLE y no es latencia de una
        orden del usuario, así que ensuciaría la serie.
        """
        if self.turno is None:
            return nullcontext()
        return self.turno.etapa(nombre)

    def _say(self, text: str):
        """Habla `text` y limpia la cola de audio para ignorar el eco de la propia voz."""
        if not text:
            return
        with self._medir("tts"):
            self.tts.speak(text)
        self.audio.q.queue.clear()

    def _listen_reply(self, max_wait_seconds: float = 6.0) -> str:
        """Captura una respuesta corta del usuario y la transcribe. Devuelve
        cadena vacía si no hubo voz (silencio/timeout)."""
        with self._medir("vad"):
            audio = self.capture_utterance(max_wait_seconds=max_wait_seconds)
        if audio is None:
            return ""
        with self._medir("stt"):
            return self.stt.transcribe(audio) or ""

    def _ask_location(self, question: str) -> Optional[Path]:
        """Pregunta una ubicación por voz e intenta resolverla a una ruta
        dentro de la whitelist. Devuelve None si no se entendió."""
        self._say(question)
        answer = self._listen_reply(max_wait_seconds=6.0)
        logger.info(f"[UBICACIÓN USUARIO]: {answer}")
        return parse_location_speech(answer)

    def _confirm_yes_no(self, question: str) -> bool:
        """Hace una pregunta de sí/no y devuelve la decisión (fail-safe: no ante duda)."""
        self._say(question)
        answer = self._listen_reply(max_wait_seconds=5.0)
        logger.info(f"[CONFIRMACIÓN USUARIO]: {answer}")
        return interpret_confirmation(answer) is True

    def _resolve_create_location(self, action) -> Optional[Path]:
        """
        Determina en qué carpeta crear algo. Si el usuario no especificó
        ubicación (ruta_base vacío o no reconocido), pregunta. SIEMPRE
        confirma la ruta final repitiéndola antes de devolverla.
        Devuelve None si el usuario cancela.
        """
        base_dir = parse_location_speech(action.ruta_base)
        if base_dir is None:
            base_dir = self._ask_location(
                "¿En qué carpeta lo creo? Puedo usar Proyectos o Clases."
            )
        if base_dir is None:
            logger.warning(f"No se entendió la ubicación pedida para crear '{action.nombre}'. Usando Proyectos por defecto.")
            base_dir = DEFAULT_ROOT

        pregunta = f"Voy a crear '{action.nombre}' en {speakable_path(base_dir)}. ¿Confirmás?"
        if self._confirm_yes_no(pregunta):
            return base_dir
        self._say("Entendido, operación cancelada.")
        return None

    def _resolve_existing_location(self, nombre: str) -> Optional[Path]:
        """
        Ubica un archivo/carpeta EXISTENTE entre las raíces permitidas
        (eliminar/mover/leer). Si aparece en más de una, o en ninguna,
        pregunta al usuario dónde está. Devuelve None si no se pudo resolver.
        """
        matches = self.executor.find_existing_roots(nombre)
        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            pregunta = f"Encontré '{nombre}' en más de un lugar. ¿Lo tomo de Proyectos o de Clases?"
        else:
            pregunta = f"No encontré '{nombre}' en Proyectos ni en Clases. ¿En qué carpeta está?"

        base_dir = self._ask_location(pregunta)
        if base_dir is None:
            self._say("No entendí la ubicación. Operación cancelada.")
            return None
        return base_dir

    def _resolve_move_destination(self, action, source_base_dir: Path) -> Optional[Path]:
        """
        Resuelve la carpeta de destino para 'mover'. Si `destino` menciona
        una raíz conocida (p.ej. "Clases"), se usa esa ruta absoluta. Si
        menciona algo sin raíz reconocida (p.ej. "trabajo"), se interpreta
        como subcarpeta dentro de la misma raíz de origen. Si no hay
        destino en absoluto, se pregunta.
        """
        dest_path = parse_location_speech(action.destino)
        if dest_path is None and action.destino:
            # No se reconoció ninguna raíz en lo que dijo el NLU: tratarlo
            # como subcarpeta relativa a la raíz donde vive el origen.
            dest_path = source_base_dir / action.destino
        if dest_path is None:
            dest_path = self._ask_location(
                f"¿A qué carpeta muevo '{action.nombre}'? Puedo usar Proyectos o Clases."
            )
        return dest_path

    def _handle_confirmation(self, action, original_text: str, question: str,
                              base_dir: Path, destino_dir: Optional[Path]):
        """
        Pregunta sí/no antes de ejecutar una acción destructiva
        (eliminar/mover) y, si el usuario confirma, la ejecuta con la MISMA
        ubicación ya resuelta (no vuelve a preguntar por la ruta). Ante
        silencio o respuesta ambigua, cancela por seguridad.
        """
        self._say(question)
        answer = self._listen_reply(max_wait_seconds=5.0)

        if not answer:
            logger.warning(f"Sin respuesta de confirmación para: {action}. Cancelado por seguridad.")
            self._say("No escuché una confirmación. Operación cancelada por seguridad.")
            return

        logger.info(f"[CONFIRMACIÓN USUARIO]: {answer}")
        decision = interpret_confirmation(answer)

        if decision is True:
            result = self.executor.execute(action, base_dir=base_dir, raw_text=original_text,
                                             confirmed=True, destino_dir=destino_dir)
            self._say(result.text)
        else:
            logger.info(f"Acción cancelada (respuesta: '{answer}'). NO ejecutada: {action}")
            self._say("Entendido, operación cancelada.")

    def _confirm_non_file_action(self, action, original_text: str, question: str):
        """
        Confirmación por voz para acciones que no son de archivos pero sí son
        irreversibles (hoy solo 'energia': suspender/apagar/reiniciar el equipo).

        Misma política fail-safe que _handle_confirmation: silencio, timeout o
        respuesta ambigua ⇒ cancelar. Es la segunda barrera del hallazgo del
        banco de evaluación, donde "apagá el bluetooth" se clasificaba como
        energia/apagar y llegaba a `systemctl poweroff` sin preguntar.
        """
        self._say(question)
        answer = self._listen_reply(max_wait_seconds=5.0)

        if not answer:
            logger.warning(f"Sin respuesta de confirmación para: {action}. Cancelado por seguridad.")
            self._say("No escuché una confirmación. Operación cancelada por seguridad.")
            return

        logger.info(f"[CONFIRMACIÓN USUARIO]: {answer}")
        if interpret_confirmation(answer) is True:
            result = dispatch_non_file_action(action, raw_text=original_text, confirmed=True)
            self._say(result.text)
        else:
            logger.info(f"Acción cancelada (respuesta: '{answer}'). NO ejecutada: {action}")
            self._say("Entendido, operación cancelada.")

    def _dispatch_action(self, action, text: str):
        """Resuelve ubicación (preguntando/confirmando cuando hace falta) y ejecuta la acción."""
        if action.action == "ninguna":
            result = self.executor.execute(action, raw_text=text)
            self._say(result.text)
            return

        if action.action in NON_FILE_ACTIONS:
            result = dispatch_non_file_action(action, raw_text=text)
            if result.needs_confirmation:
                self._confirm_non_file_action(action, text, result.text)
            else:
                self._say(result.text)
            return

        if action.action != "listar" and not action.nombre:
            self._say("Lo siento, necesito que me digas un nombre para el archivo o carpeta.")
            return

        if action.action in CREATE_ACTIONS:
            base_dir = self._resolve_create_location(action)
            if base_dir is None:
                return  # ya se avisó "operación cancelada"
            result = self.executor.execute(action, base_dir=base_dir, raw_text=text)
            self._say(result.text)
            return

        if action.action in LOCATE_EXISTING_ACTIONS:
            base_dir = self._resolve_existing_location(action.nombre)
            if base_dir is None:
                return

            destino_dir = None
            if action.action == "mover":
                destino_dir = self._resolve_move_destination(action, base_dir)
                if destino_dir is None:
                    self._say("No entendí a dónde moverlo. Operación cancelada.")
                    return

            result = self.executor.execute(action, base_dir=base_dir, raw_text=text, destino_dir=destino_dir)
            if result.needs_confirmation:
                self._handle_confirmation(action, text, result.text, base_dir=base_dir, destino_dir=destino_dir)
            else:
                self._say(result.text)
            return

        # listar (y cualquier acción futura sin ubicación interactiva)
        base_dir = parse_location_speech(action.ruta_base) or DEFAULT_ROOT
        result = self.executor.execute(action, base_dir=base_dir, raw_text=text)
        self._say(result.text)

    def run(self):
        self.running = True
        self.audio.start()
        logger.info("Assistant started. State: IDLE")
        last_timer_check = 0.0

        try:
            while self.running:
                try:
                    if self.state == State.IDLE:
                        # Chequeo de temporizadores/alarmas vencidos, throttleado a
                        # 1 vez por segundo (no en cada chunk de 80ms) para no
                        # pegarle a disco 12 veces por segundo sin necesidad.
                        now = time.monotonic()
                        if now - last_timer_check >= 1.0:
                            last_timer_check = now
                            for mensaje in check_due_timers():
                                self._say(mensaje)

                        # In IDLE, read 80ms chunks (1280 samples) for wake word
                        chunk = self.audio.read_chunk(1280)
                        if self.ww.feed(chunk):
                            logger.info("Transitioning to LISTENING state")
                            play_beep()
                            self.state = State.LISTENING

                    elif self.state == State.LISTENING:
                        self.turno = TurnoMetricas()
                        accion_del_turno = None

                        with self._medir("vad"):
                            utterance_audio = self.capture_utterance(max_wait_seconds=3.0)

                        if utterance_audio is not None:
                            self.save_utterance(utterance_audio)

                            with self._medir("stt"):
                                text = self.stt.transcribe(utterance_audio)
                            # Whisper can hallucinate spaces or symbols, so check if there's actual text
                            if text and any(c.isalpha() for c in text):
                                logger.info(f"[USER SAYS]: {text}")

                                with self._medir("nlu"):
                                    action = self.nlu.parse(text)
                                accion_del_turno = action.action
                                self._dispatch_action(action, text)
                            else:
                                logger.info("Discarding empty or noise-only transcription.")

                        # Una línea por turno en logs/metrics.jsonl. Los tokens solo
                        # se anotan si hubo parseo: si no, last_metrics todavía tiene
                        # los del turno anterior y estaría mintiendo.
                        tokens = self.nlu.last_metrics if accion_del_turno else {}
                        self.turno.registrar(
                            accion=accion_del_turno,
                            audio_s=(round(len(utterance_audio) / SAMPLE_RATE, 2)
                                     if utterance_audio is not None else None),
                            tokens_prompt=tokens.get("prompt_tokens"),
                            tokens_salida=tokens.get("output_tokens"),
                        )
                        self.turno = None

                        logger.info("Transitioning to IDLE state")
                        self.state = State.IDLE
                except AudioDeviceLost as e:
                    # No es un error fatal: el dispositivo puede volver.
                    logger.warning(f"{e} Reabriendo la entrada de audio.")
                    self._reabrir_audio()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Stopping...")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.audio.stop()

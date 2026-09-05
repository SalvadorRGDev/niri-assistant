import numpy as np
from faster_whisper import WhisperModel
from src.config import STT_MODEL_SIZE, STT_COMPUTE_TYPE, STT_MIN_AVG_LOGPROB
from src.logger import get_logger

logger = get_logger("SpeechToText")

# Prompt de contexto para orientar la decodificación hacia las órdenes reales del
# asistente. Se probaron cinco variantes contra grabaciones reales de recordings/:
# una lista de vocabulario suelto ("carpeta, archivo, fotos...") acertó 2 de 5,
# mientras que esta forma —frases completas con la misma estructura que el usuario
# habla— acertó 5 de 5, incluido el caso que antes fallaba siempre ("archivo" se
# decodificaba como "al chivo"). La estructura importa más que la cantidad de
# palabras: Whisper imita el formato del prompt, no solo su vocabulario.
INITIAL_PROMPT = (
    "Crea una carpeta llamada archivo. Crea una carpeta llamada fotos. "
    "Crea una carpeta llamada pruebas. Elimina el archivo. "
    "Mueve el archivo a Proyectos. Abre la aplicación. Sube el volumen. Sí. No."
)

class SpeechToText:
    def __init__(self, model_size=STT_MODEL_SIZE, compute_type=STT_COMPUTE_TYPE):
        logger.info(f"Loading Whisper model '{model_size}' ({compute_type}) on CPU...")
        # device="cpu" is required if no GPU is available
        self.model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
        logger.info("Whisper model loaded successfully.")

    def transcribe(self, audio_array: np.ndarray) -> str:
        """
        Transcribe un array de audio (int16 o float32) a texto.
        faster_whisper procesa internamente float32 normalizado.
        """
        if audio_array.dtype == np.int16:
            audio_array = audio_array.astype(np.float32) / 32768.0

        logger.info("Transcribing audio...")
        segments, info = self.model.transcribe(
            audio_array,
            beam_size=5,
            language="es",
            # Recorta el silencio antes de decodificar: además de acelerar, evita
            # las alucinaciones típicas de Whisper sobre audio vacío/ruido (en los
            # logs aparecían "¡Suscríbete!" y "Subtítulos realizados por la
            # comunidad de Amara.org", frases de sus datos de entrenamiento).
            vad_filter=True,
            # Sesga la decodificación hacia el vocabulario real del asistente. Sin
            # esto, palabras del dominio se decodifican como parecidos fonéticos
            # ("archivo" -> "al chivo", "fotos" -> "Photos").
            initial_prompt=INITIAL_PROMPT,
            # Cada orden es independiente: arrastrar el texto del turno anterior
            # como contexto solo propaga errores entre comandos.
            condition_on_previous_text=False,
        )
        
        # Generator for segments. Se descartan los segmentos de baja confianza:
        # son casi siempre alucinaciones sobre silencio o ruido (ver
        # STT_MIN_AVG_LOGPROB en src/config.py). Actuar sobre una de ellas sería
        # peor que no entender nada, porque el modelo tiende a alucinar frases
        # del initial_prompt, incluidas las destructivas.
        text_parts = []
        for segment in segments:
            if segment.avg_logprob < STT_MIN_AVG_LOGPROB:
                logger.warning(
                    f"Descartando segmento de baja confianza "
                    f"(avg_logprob={segment.avg_logprob:.3f} < {STT_MIN_AVG_LOGPROB}, "
                    f"no_speech_prob={segment.no_speech_prob:.3f}): {segment.text.strip()!r}"
                )
                continue
            text_parts.append(segment.text)

        final_text = " ".join(text_parts).strip()
        logger.info(f"Transcription result: {final_text}")
        return final_text

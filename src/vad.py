import numpy as np
import onnxruntime as ort

from src.config import VAD_MODEL_PATH, VAD_SILENCE_TIMEOUT_MS, VAD_THRESHOLD, SAMPLE_RATE, AUDIO_GAIN
from src.logger import get_logger

logger = get_logger("VoiceActivityDetector")

# Silero VAD (silero_vad.onnx, interfaz combinada v4/v5) espera chunks fijos
# de 512 muestras a 16kHz (32ms). No es configurable: usar otro tamaño rompe
# el estado recurrente del modelo.
SILERO_CHUNK_SAMPLES = 512

# El modelo v5 es causal y espera, además del chunk, un "contexto" de las
# últimas CONTEXT_SAMPLES muestras del chunk anterior pegado adelante (así
# lo hace el wrapper oficial OnnxWrapper.__call__ de snakers4/silero-vad).
# Sin esto, la ventana de convolución queda desalineada y el output se
# satura cerca de 0 sin importar el contenido del audio (bug detectado:
# habla real daba prob<0.004 incluso a volumen máximo).
CONTEXT_SAMPLES = 64


class VoiceActivityDetector:
    def __init__(self, model_path=VAD_MODEL_PATH, threshold=VAD_THRESHOLD, silence_timeout=VAD_SILENCE_TIMEOUT_MS):
        """
        Wrapper de Silero VAD vía onnxruntime puro (silero-vad-lite no compila
        en este entorno, ver README.md). Consume el .onnx nativo
        descargado en models/silero_vad.onnx.
        """
        self.threshold = threshold
        self.silence_timeout_ms = silence_timeout
        self.chunk_ms = int(SILERO_CHUNK_SAMPLES / SAMPLE_RATE * 1000)  # 32ms
        self.max_silent_chunks = max(1, self.silence_timeout_ms // self.chunk_ms)

        logger.info(f"Loading Silero VAD model from {model_path} (threshold={threshold})")
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

        logger.info(f"Silero VAD ready (chunk={SILERO_CHUNK_SAMPLES} samples / {self.chunk_ms}ms, "
                    f"max_silent_chunks={self.max_silent_chunks})")

    def reset_states(self):
        """Reinicia el estado recurrente del modelo (llamar al iniciar cada utterance)."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def process_chunk(self, audio_chunk: np.ndarray) -> float:
        """
        Procesa un chunk de audio int16 de exactamente SILERO_CHUNK_SAMPLES
        muestras y retorna la probabilidad de voz (0.0 - 1.0) según Silero.
        Aplica una ganancia de compensación (AUDIO_GAIN) porque el micrófono
        UAC del usuario tiene salida muy baja (ver README.md).
        """
        if len(audio_chunk) != SILERO_CHUNK_SAMPLES:
            # Silero exige tamaño fijo; si llega un chunk parcial (p.ej. al
            # final del stream), se rellena con ceros para no romper la red.
            padded = np.zeros(SILERO_CHUNK_SAMPLES, dtype=audio_chunk.dtype)
            padded[: len(audio_chunk)] = audio_chunk
            audio_chunk = padded

        audio_f32 = audio_chunk.astype(np.float32) * AUDIO_GAIN / 32768.0
        audio_f32 = np.clip(audio_f32, -1.0, 1.0).reshape(1, -1)

        model_input = np.concatenate([self._context, audio_f32], axis=1)

        try:
            output, self._state = self.session.run(
                ["output", "stateN"],
                {"input": model_input, "state": self._state, "sr": self._sr},
            )
            self._context = model_input[:, -CONTEXT_SAMPLES:]
            return float(output[0][0])
        except Exception as e:
            logger.error(f"Silero VAD inference error: {e}")
            return 0.0

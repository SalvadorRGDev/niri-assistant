import numpy as np
import onnxruntime as ort
from typing import Iterable, Optional
from pyopen_wakeword import Model, OpenWakeWord, OpenWakeWordFeatures

from src.config import WAKE_WORD_MODEL, WAKE_WORD_THRESHOLD, WAKE_WORD_TRIGGER_LEVEL
from src.logger import get_logger

logger = get_logger("WakeWordDetector")

WW_FEATURES = 96


class OnnxClassifier:
    """
    Clasificador de wake word corrido con onnxruntime en vez de TFLite.

    pyopen-wakeword solo sabe cargar modelos .tflite (ver OpenWakeWord.from_model),
    pero TensorFlow no tiene wheel para Python 3.14 en este entorno, así que el
    modelo custom "oye_niri" se entrenó en PyTorch y se exportó a .onnx en su
    lugar. Esta clase replica el comportamiento de ventana deslizante de
    OpenWakeWord.process_streaming (ver pyopen_wakeword/openwakeword.py) para
    que WakeWordDetector.feed() no tenga que distinguir entre ambos backends.
    """

    def __init__(self, model_path):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_shape = self.session.get_inputs()[0].shape
        self.input_windows = int(input_shape[1])
        self.input_name = self.session.get_inputs()[0].name

        self.embeddings = np.zeros((0, WW_FEATURES), dtype=np.float32)

    def process_streaming(self, embeddings: np.ndarray) -> Iterable[float]:
        """Genera probabilidades a partir de embeddings (mismo contrato que OpenWakeWord)."""
        new_frames = embeddings.reshape(-1, WW_FEATURES)
        self.embeddings = np.vstack([self.embeddings, new_frames])

        while self.embeddings.shape[0] >= self.input_windows:
            window = self.embeddings[: self.input_windows]
            self.embeddings = self.embeddings[1:]

            emb_tensor = window.reshape(1, self.input_windows, WW_FEATURES).astype(np.float32)
            prob = self.session.run(None, {self.input_name: emb_tensor})[0]
            yield float(np.asarray(prob).ravel()[0])

    def reset(self) -> None:
        self.embeddings = np.zeros((0, WW_FEATURES), dtype=np.float32)


class WakeWordDetector:
    def __init__(self, model_name=WAKE_WORD_MODEL, threshold=WAKE_WORD_THRESHOLD,
                 trigger_level=WAKE_WORD_TRIGGER_LEVEL):
        self.model_name = model_name
        self.threshold = threshold
        # Frames consecutivos por encima del umbral necesarios para disparar
        # (ver WAKE_WORD_TRIGGER_LEVEL en config.py: evita que un pico
        # aislado de un solo frame de 80ms active el asistente).
        self.trigger_level = trigger_level
        self._consecutive_hits = 0

        logger.info(f"Loading wake word model: {model_name}")

        # Determine built-in model enum if available
        builtin_model = None
        for m in Model:
            if m.value == model_name:
                builtin_model = m
                break

        self.oww_features = OpenWakeWordFeatures.from_builtin()

        if builtin_model:
            self.oww_model = OpenWakeWord.from_builtin(builtin_model)
        elif str(model_name).endswith(".onnx"):
            logger.info(f"Loading custom ONNX classifier from {model_name}")
            self.oww_model = OnnxClassifier(model_name)
        else:
            logger.info(f"Loading custom model from {model_name}")
            self.oww_model = OpenWakeWord.from_model(model_name)
            
    def feed(self, audio_chunk: np.ndarray) -> bool:
        """
        Alimenta un chunk de audio (int16) y retorna True si se detecta la wake word.
        `audio_chunk` idealmente es de tamaño 1280 (80ms).
        """
        # Convert to bytes as pyopen-wakeword expects raw bytes representing int16 samples
        audio_bytes = audio_chunk.tobytes()
        
        for features in self.oww_features.process_streaming(audio_bytes):
            for prob in self.oww_model.process_streaming(features):
                if prob >= self.threshold:
                    self._consecutive_hits += 1
                    logger.debug(
                        f"Wake word frame above threshold ({self._consecutive_hits}/{self.trigger_level}). Score: {prob:.3f}"
                    )
                    if self._consecutive_hits >= self.trigger_level:
                        logger.info(f"Wake word detected! Score: {prob:.3f}")
                        self._consecutive_hits = 0
                        # Reset internal state to avoid consecutive triggers
                        self.oww_features.reset()
                        self.oww_model.reset()
                        return True
                else:
                    self._consecutive_hits = 0
        return False

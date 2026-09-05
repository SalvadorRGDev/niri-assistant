import queue
import sounddevice as sd
import numpy as np
import scipy.signal
from src.config import SAMPLE_RATE, CAPTURE_SAMPLE_RATE, CHANNELS, AUDIO_DEVICE
from src.logger import get_logger

logger = get_logger("AudioStream")

class AudioStream:
    def __init__(self, device=AUDIO_DEVICE):
        self.device = device
        self.q = queue.Queue()
        self.stream = None
        self.buffer = np.array([], dtype=np.int16)
        
        # Determine actual device index/name if possible
        try:
            # We look for a device matching the name or string
            if isinstance(self.device, str):
                devices = sd.query_devices()
                match_id = None
                for i, d in enumerate(devices):
                    if self.device.lower() in d['name'].lower() and d['max_input_channels'] > 0:
                        match_id = i
                        break
                if match_id is not None:
                    self.device = match_id
            logger.info(f"Using audio device: {self.device}")
        except Exception as e:
            logger.warning(f"Failed to query device, using as is: {e}")

    def callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio status: {status}")
        
        # Obtenemos la data original (shape: frames, channels)
        raw_data = indata.copy()[:, 0]
        
        # Remuestreo simple (decimation) para mantener la continuidad de fase entre chunks
        if CAPTURE_SAMPLE_RATE != SAMPLE_RATE:
            factor = CAPTURE_SAMPLE_RATE // SAMPLE_RATE
            resampled_data = raw_data[::factor]
        else:
            resampled_data = raw_data
            
        self.q.put(resampled_data)
        
    def start(self):
        logger.info(f"Starting audio stream... (Captura: {CAPTURE_SAMPLE_RATE}Hz, Salida: {SAMPLE_RATE}Hz)")
        self.stream = sd.InputStream(
            device=self.device,
            samplerate=CAPTURE_SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16',
            callback=self.callback
        )
        self.stream.start()
        
    def stop(self):
        logger.info("Stopping audio stream...")
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
    def read_chunk(self, size: int) -> np.ndarray:
        """Read exactly `size` samples from the audio stream"""
        # If we already have enough in the buffer, take it
        while len(self.buffer) < size:
            new_data = self.q.get()
            self.buffer = np.concatenate((self.buffer, new_data))
            
        chunk = self.buffer[:size]
        self.buffer = self.buffer[size:]
        return chunk

"""
Decide si una grabación contiene voz, mirando cómo se reparte su energía.

Existe porque el nivel en dBFS no alcanza: las dos primeras muestras grabadas
con eval/grabar_wakeword.py marcaban -19 y -13 dBFS —niveles sanos— y sin
embargo Whisper las transcribe vacías y el wake word les da 0.15. Tenían el
70% y el 93% de su energía por debajo de 100 Hz: el ruido de manipular la
laptop, no un "oye niri". Una muestra así envenena el análisis, porque después
no se distingue de un positivo que el modelo falló, que es justo lo que el
análisis intenta medir.

Los umbrales salen de las grabaciones reales de recordings/: las utterances con
habla tienen entre 44% y 89% de su energía en 300-1000 Hz y menos del 4% por
debajo de 100 Hz.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np  # noqa: E402

from src.config import SAMPLE_RATE  # noqa: E402

MIN_PCT_VOZ = 15.0      # energía en 300-1000 Hz
MAX_PCT_RUMBLE = 40.0   # energía por debajo de 100 Hz

# Ventana de la FFT: 1024 muestras son 64 ms a 16 kHz, suficiente para resolver
# los 100 Hz que separan el rumble de la voz.
VENTANA = 1024


def forma_espectral(audio: np.ndarray) -> tuple[float, float]:
    """(% de energía en 300-1000 Hz, % por debajo de 100 Hz) de la grabación."""
    ventana = np.hanning(VENTANA)
    trozos = [audio[i:i + VENTANA]
              for i in range(0, len(audio) - VENTANA, VENTANA // 2)]
    if not trozos:
        return 0.0, 100.0

    espectro = np.mean([np.abs(np.fft.rfft(t.astype(np.float32) * ventana)) ** 2
                        for t in trozos], axis=0)
    freqs = np.fft.rfftfreq(VENTANA, 1 / SAMPLE_RATE)
    total = espectro.sum()
    if total <= 0:
        return 0.0, 100.0

    voz = 100 * espectro[(freqs >= 300) & (freqs < 1000)].sum() / total
    rumble = 100 * espectro[freqs < 100].sum() / total
    return float(voz), float(rumble)


def tiene_voz(audio: np.ndarray) -> tuple[bool, str]:
    """True si la grabación parece contener habla; si no, el motivo del descarte."""
    voz, rumble = forma_espectral(audio)
    if rumble > MAX_PCT_RUMBLE:
        return False, f"{rumble:.0f}% de la energía por debajo de 100 Hz (golpe o roce, no voz)"
    if voz < MIN_PCT_VOZ:
        return False, f"solo {voz:.0f}% de la energía en la banda de voz (300-1000 Hz)"
    return True, ""

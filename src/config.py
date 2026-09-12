import os
from pathlib import Path

# Configuración Base
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
RECORDINGS_DIR = BASE_DIR / "recordings"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

# Log de auditoría append-only (capa 5 del plan): timestamp + transcripción + acción.
AUDIT_LOG_PATH = LOGS_DIR / "audit.log"

# Persistencia de temporizadores/alarmas y notas/recordatorios (fase "tiempo").
TIMERS_STATE_PATH = DATA_DIR / "timers.json"
NOTES_STATE_PATH = DATA_DIR / "notas_recordatorios.json"

# Configuración de Audio
SAMPLE_RATE = 16000
CAPTURE_SAMPLE_RATE = 48000  # UACDemo opera a 48kHz nativo; remuestreamos en software
CHANNELS = 1

# Default audio device
# Set to None to use the system's default microphone (PipeWire/PulseAudio)
AUDIO_DEVICE = None

# Wake Word
# Modelo custom "oye niri" (models/oye_niri.onnx), entrenado con voces
# sintéticas de Piper en español (Argentina/España/México) + negativos
# reales de recordings/ — ver README.md para más detalle y para
# cómo reentrenarlo con la voz real del usuario si hace falta más precisión.
# Se carga con onnxruntime en vez de TFLite (ver src/wake_word.py:OnnxClassifier)
# porque TensorFlow no tiene wheel para Python 3.14 en este entorno.
WAKE_WORD_MODEL = os.getenv("WAKE_WORD_MODEL", str(MODELS_DIR / "oye_niri.onnx"))
# 0.85 no alcanzaba en uso real: el modelo (entrenado solo con voces
# sintéticas) llegaba a ese umbral con un único frame de 80ms disparado por
# sonidos cortos ("i", ruido, etc.), activando el asistente con cualquier
# cosa. Se sube a 0.9 y además se exige sostenerlo varios frames seguidos
# (ver WAKE_WORD_TRIGGER_LEVEL) para que solo "oye niri" dicho completo
# dispare la activación.
WAKE_WORD_THRESHOLD = float(os.getenv("WAKE_WORD_THRESHOLD", "0.9"))
# Cantidad de frames consecutivos (cada uno ~80ms) que deben superar el
# umbral antes de considerar que hubo wake word real, en vez de disparar
# con un solo pico aislado.
WAKE_WORD_TRIGGER_LEVEL = int(os.getenv("WAKE_WORD_TRIGGER_LEVEL", "3"))

# VAD (Silero VAD real, vía onnxruntime — ver src/vad.py)
VAD_MODEL_PATH = MODELS_DIR / "silero_vad.onnx"
VAD_SILENCE_TIMEOUT_MS = int(os.getenv("VAD_SILENCE_TIMEOUT_MS", "800"))
# 0.5 es el umbral recomendado oficialmente por Silero (probabilidad continua,
# no un booleano como en WebRTC-VAD). Ajustar con cautela.
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
# Ganancia de compensación: el micrófono UAC del usuario satura bajo
# (ver README.md). Se aplica antes de pasar el audio a Silero.
AUDIO_GAIN = float(os.getenv("AUDIO_GAIN", "3.0"))

# STT
# 'base' en vez de 'small': medido en este equipo, 'base' transcribe en 0.80s
# lo que 'small' tardaba 2.11s (RTF 0.27x vs 0.70x) y evita los picos de CPU a
# ~90°C que causaba 'small' en el laptop. La precisión con el micrófono actual
# resultó pareja entre ambos (ver historial de logs: 'small' produjo errores
# equivalentes como "y me da fosa" o alucinaciones tipo "¡Suscríbete!"), así que
# el cuello de botella de precisión es el audio de entrada, no el tamaño del
# modelo — de ahí las opciones de decodificación en src/stt.py.
STT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "base")
# Confianza mínima (avg_logprob de Whisper) para aceptar un segmento transcrito.
# Medido sobre grabaciones reales de recordings/: el habla clara cae entre -0.08 y
# -0.30, mientras que las alucinaciones sobre silencio/ruido caen en -0.65 o peor.
# El corte en -0.5 separa ambos grupos con margen. Es una defensa necesaria porque
# el initial_prompt, además de mejorar la precisión, hace que el modelo alucine
# frases DEL PROMPT cuando el audio está vacío — y una de ellas es "Elimina el
# archivo", que es destructiva. Con este filtro, esos casos se descartan y
# main_loop.py los trata como "transcripción vacía" (vuelve a IDLE sin actuar).
STT_MIN_AVG_LOGPROB = float(os.getenv("STT_MIN_AVG_LOGPROB", "-0.5"))
STT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "int8")

# TTS — Piper y espeak-ng se ejecutan siempre en local. La nube está apagada
# por defecto: habilitar el flag solo permite edge-tts como último respaldo
# para frases que el código marque explícitamente como públicas. Las respuestas
# con nombres, rutas o contenido del usuario siguen siendo locales.
PIPER_BINARY = os.getenv("PIPER_BINARY", "piper")
# Voz rioplatense (daniela), elegida por ser la más parecida a la voz de la nube
# disponible como respaldo público opcional. Cuesta más que la mexicana anterior
# —114 MB contra 60 en disco, 241 ms contra 60 por frase, 843 ms contra 533 al
# cargar— y aun así es 4x más rápida que edge-tts, que ronda los 900 ms.
# src/tts.py acepta cualquier .onnx de models/piper/ si este no está.
PIPER_VOICE_MODEL = Path(os.getenv(
    "PIPER_VOICE_MODEL", str(MODELS_DIR / "piper" / "es_AR-daniela-high.onnx")))
ALLOW_CLOUD_TTS_FALLBACK = os.getenv("ALLOW_CLOUD_TTS_FALLBACK", "false").lower() in ("1", "true", "yes")

# UX
BEEP_ON_WAKE_WORD = True

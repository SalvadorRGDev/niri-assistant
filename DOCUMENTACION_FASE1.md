# Documentación Técnica — Fase 1: Asistente de Voz "Salvador"

Este documento detalla la implementación de la **Fase 1**, la cual abarca la inicialización del entorno, el detector de Wake Word continuo y el Detector de Actividad Vocal (VAD) para capturar instrucciones de voz.

## 1. Arquitectura y Módulos

La base del asistente es una máquina de estados sencilla ubicada en `src/main_loop.py` que itera entre dos estados:
- **`IDLE` (En Reposo)**: Escucha pasivamente en fragmentos de 80 milisegundos a la espera de la palabra de activación (wake word).
- **`LISTENING` (Escuchando)**: Activa el VAD de Silero para guardar todo el audio hasta que detecta 800 milisegundos de silencio continuo. Al finalizar, guarda la frase capturada en `.wav` y regresa a `IDLE`.

### Módulos Creados
* **`main.py`**: El punto de entrada principal. Llama a la inicialización del `MainLoop`.
* **`src/config.py`**: Configuración centralizada de variables del sistema como `SAMPLE_RATE`, `WAKE_WORD_MODEL`, y umbrales de detección.
* **`src/audio.py`**: Contiene la clase `AudioStream`. Mantiene un hilo seguro leyendo del micrófono usando `sounddevice`.
* **`src/wake_word.py`**: Envuelve la detección mediante `pyopen-wakeword`.
* **`src/vad.py`**: Maneja el VAD a través de `onnxruntime` directo.
* **`src/logger.py`**: Provee logs estandarizados.
* **`test_phase1.py`**: Herramienta de pruebas para analizar uso de CPU sin ejecutar la parte interactiva pesada.

## 2. Decisiones Técnicas y Solución de Problemas

Durante el desarrollo de esta fase, surgieron limitantes técnicas específicas a este entorno que requirieron adaptaciones estructurales:

### Compatibilidad con Python 3.14
* **Problema:** El paquete original `openwakeword` requiere `tflite-runtime`, el cual no posee binarios (wheels) para Python 3.14.
* **Solución:** Se utilizó **`pyopen-wakeword`**, un reemplazo compatible construido sobre la API original, lo que garantizó compatibilidad y evitó tener que downgradear el sistema a Python 3.12 usando `pyenv`.
* **Modelo actual:** ya no se usa el genérico `hey_jarvis`. El modelo en producción es
  **`models/oye_niri.onnx`**, entrenado con voces sintéticas de Piper en español y
  ejecutado con `onnxruntime` (ver `OnnxClassifier` en `src/wake_word.py` y el detalle
  de entrenamiento en `README_PENDIENTES.md`).

### Motor VAD ligero
* **Problema:** El paquete `silero-vad-lite` oficial fallaba al compilarse en el entorno vía CMake debido a incompatibilidades de la arquitectura de sistema.
* **Solución:** Se implementó una integración manual en `src/vad.py` consumiendo el archivo ONNX de Silero nativo (`silero_vad.onnx`) a través de **`onnxruntime`** para ejecutar inferencias puras. 

### Incompatibilidad de Sample Rate en Hardware
* **Problema:** El micrófono por defecto del usuario (`UACDemoV1.0`) rechazó la captura nativa a `16000 Hz` (error de ALSA `PaErrorCode -9997`). Los modelos neuronales de audio estandarizados colapsan si el sample rate cambia.
* **Solución:** Se integró la librería `scipy` (`scipy.signal.resample_poly`) en `src/audio.py` para aplicar un **remuestreo al vuelo** (resampling). El sistema captura el audio al estándar universal de `48000 Hz` (ampliamente soportado por hardware USB) y lo reduce matemáticamente en memoria a `16000 Hz` antes de enviarlo a las redes neuronales, todo con una penalidad mínima de CPU.

### Nivel de captura del micrófono y alcance real de `AUDIO_GAIN`

* **Problema:** el micrófono UAC del usuario entrega señal baja, y se agregó
  `AUDIO_GAIN=3.0` como compensación.
* **Medición:** analizadas las 51 grabaciones de `recordings/`, el pico medio es
  de **-7.8 dBFS** y el RMS medio de **-27.3 dBFS**. El nivel varía bastante entre
  sesiones (algunas frases llegan a -22 dBFS de RMS y otras a -32 dBFS), y **al
  menos una grabación ya alcanza 0 dBFS**, es decir que clipea. Por lo tanto subir
  un factor de ganancia fijo no sería seguro: distorsionaría las capturas fuertes
  sin arreglar las flojas.
* **Alcance real de la ganancia:** `AUDIO_GAIN` se aplica **únicamente en
  `src/vad.py`**, no en la ruta que alimenta a Whisper. Eso es correcto y no hace
  falta cambiarlo. Se verificó transcribiendo la misma grabación a 1x, 2x, 3x y
  normalizada a -3 dBFS: **la salida es idéntica en los cuatro casos**, porque
  Whisper trabaja sobre características log-mel que son esencialmente invariantes
  a la amplitud. Silero VAD, en cambio, sí es sensible al nivel absoluto, que es
  exactamente donde la ganancia está aplicada.
* **Conclusión:** los errores de transcripción de este proyecto no se arreglan
  subiendo volumen. Se atacaron desde las opciones de decodificación de Whisper
  (ver `DOCUMENTACION_FINAL.md`, sección STT).

## 3. Rendimiento
El benchmark automatizado en estado `IDLE` (estado de operación el 99% del tiempo) muestra un consumo de CPU general promedio de **~17%** (variable según hardware). Este overhead incluye el pooling de la interfaz de audio de ALSA, el proceso continuo del modelo Wake Word en CPU, y la conversión del remuestreo en tiempo real.

Las latencias del pipeline completo (STT + NLU + TTS) y la configuración de GPU
que las hace posibles están documentadas en `DOCUMENTACION_FINAL.md`, sección
"Rendimiento y aceleración por GPU".

## 4. Estructura de Directorios Resultante

```text
Asistente/
├── .venv/                            # Entorno Virtual Python 3.14
├── PLAN_IMPLEMENTACION_SALVADOR.md   # Roadmap general
├── DOCUMENTACION_FASE1.md            # Este documento
├── main.py                           # Entry point
├── requirements.txt                  # Dependencias
├── test_phase1.py                    # Pruebas CPU
├── models/
│   └── silero_vad.onnx               # Modelo nativo de VAD descargado
├── recordings/                       # Outputs temporales en formato .wav
└── src/
    ├── __init__.py
    ├── audio.py                      # Interfaz sounddevice + scipy resample
    ├── config.py                     # Constantes
    ├── logger.py                     # Stdout logs
    ├── main_loop.py                  # Máquina de estados
    ├── vad.py                        # onnxruntime wrapper
    └── wake_word.py                  # pyopen-wakeword wrapper
```

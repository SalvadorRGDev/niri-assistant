# Pendientes manuales — Asistente "Niri"

Este documento lista lo que quedó implementado en código en esta sesión y lo
que **requiere una acción manual tuya** porque no se puede hacer de forma
remota (necesita tu micrófono, tu hardware, o acceso a redes bloqueadas desde
el entorno donde se editó el código).

## 1. TTS local con Piper (antes: edge-tts, dependía de internet)

`src/tts.py` ahora prioriza **Piper** (100% local, calidad neuronal) y solo
cae a `edge-tts` (nube) si activás `ALLOW_CLOUD_TTS_FALLBACK=true` — por
defecto está desactivado, así que hoy el asistente cae directo a
`espeak-ng` si Piper no está instalado.

Para activar Piper:

```bash
cd ~/Proyectos/Asistente
source .venv/bin/activate
pip install piper-tts

mkdir -p models/piper
# Descargar una voz ES (ejemplo es_MX, calidad "medium") desde el repo de
# voces de Piper (rhasspy/piper-voices en Hugging Face) y guardarla como:
#   models/piper/es_MX.onnx
#   models/piper/es_MX.onnx.json
```

Si preferís otra voz o ruta, configurá las variables de entorno
`PIPER_VOICE_MODEL` (ruta al .onnx) y `PIPER_BINARY` (si el ejecutable
`piper` no está en el PATH) — ver `src/config.py`.

Verificá con:
```bash
python test_pipeline.py "Hola, esta es una prueba de voz con Piper"
```
El log de `TTS` te va a decir si detectó Piper (`piper=True`) o si sigue
usando el fallback.

## 2. Wake word custom `oye_niri.onnx` (ya entrenado — falta tu validación real)

`src/wake_word.py` ya no usa el modelo genérico `hey_jarvis`: `models/oye_niri.onnx`
es un clasificador chico (PyTorch → ONNX, corrido con `onnxruntime` porque
TensorFlow no tiene wheel para Python 3.14 en este entorno — ver
`OnnxClassifier` en `src/wake_word.py`) entrenado 100% con voces sintéticas
de Piper en español (8 voces de Argentina/España/México) diciendo "oye niri",
más negativos sintéticos (vocabulario de comandos del asistente + frases
parecidas tipo "oye mari"/"oye kiri") y tus 17 grabaciones reales de
`recordings/` + `test_mic.wav` como negativos de audio real.

Validación en streaming (frame por frame, como corre en producción, no solo
por ventana suelta): 15/15 clips positivos sintéticos disparan la detección,
32/33 clips negativos (sintéticos + tus grabaciones reales) se mantienen en
silencio — ningún falso positivo contra tu voz real grabada.

**Validado en vivo con la voz real del usuario** (sesión del 2026-09-03): el
wake word disparó de forma consistente con scores de 0.902, 0.918, 0.945 y
0.989, sin falsos positivos durante las pruebas. El umbral quedó en `0.9` con
`WAKE_WORD_TRIGGER_LEVEL=3` (tres frames consecutivos por encima del umbral),
que es la combinación que eliminó las activaciones espurias por sonidos cortos.
Este punto ya no está pendiente. Si querés mejorarlo más adelante, el camino es reentrenar
agregando grabaciones reales tuyas diciendo "oye niri" como positivos
adicionales (el pipeline de entrenamiento quedó en
`/tmp/.../scratchpad/wakeword-train/` de esta sesión — no es parte del
repo, así que para reentrenar hace falta rearmar el entorno: ver
`build_features.py`/`train.py` como referencia del proceso).

## 3. Resumen de lo implementado en esta sesión (sin pasos manuales)

- **Log de auditoría append-only** (`src/audit.py`, `logs/audit.log`):
  registra timestamp + transcripción + acción + resultado de *todo* intento,
  incluidos los bloqueados por seguridad.
- **Silero VAD real** (`src/vad.py`): reemplaza el `webrtcvad` (que no
  coincidía con lo documentado en `DOCUMENTACION_FASE1.md`) por el modelo
  `models/silero_vad.onnx` vía `onnxruntime`, validado con audio real
  grabado en `recordings/`.
- **Confirmación por voz obligatoria en `eliminar`/`mover`**
  (`src/executor.py`, `src/confirm.py`, `src/main_loop.py`): Niri
  pregunta y espera "sí"/"no" antes de tocar disco. Ante silencio o
  respuesta ambigua, **cancela por seguridad** (nunca asume "sí").
- **Acciones `mover` y `leer`** implementadas en `src/executor.py` (antes
  caían en "todavía no sé hacer eso").
- **Protección de directorios raíz**: ya no se puede eliminar/mover
  `~/Proyectos` o `~/Clases` completos, aunque técnicamente "pasen" el
  chequeo de whitelist.
- `requirements.txt` limpiado: se quitó `pyttsx3` y `pygame` (no se usaban
  en ningún archivo) y se agregó `piper-tts`.

Todo lo anterior se probó con datos reales o simulaciones controladas antes
de guardarse (inferencia real de Silero sobre tus `.wav` grabados, y un test
de integración mockeado que ejercita el `Executor` real a través de todo el
flujo de confirmación). Lo único que no se pudo probar en este entorno es la
ejecución con micrófono/Ollama/Whisper reales — para eso, correlo en tu
máquina con `python main.py` o `python test_pipeline.py "<texto>"`.

## 4. Sesión de optimización de latencia (2026-09-03)

El tiempo de respuesta bajó de **~7.7s a ~2.5s por turno**. El detalle técnico
completo está en `DOCUMENTACION_FINAL.md`; acá quedan solo los pasos que
requirieron intervención manual con `sudo`, **ya ejecutados**, para que consten
si hay que rehacer el entorno:

```bash
sudo pacman -S ollama-vulkan      # el paquete `ollama` de Arch es solo-CPU
sudo systemctl restart ollama
```

Cambios en código (sin pasos manuales): `STT_MODEL_SIZE` pasó a `base`,
`src/stt.py` incorporó `vad_filter`, un `initial_prompt` de frases y el filtro
de confianza `STT_MIN_AVG_LOGPROB`. Validado en vivo con voz real: 3 de 3
transcripciones correctas en un diálogo completo, incluido el caso "archivo"
que antes se decodificaba siempre como "al chivo".

### Sigue pendiente

- **Piper** (sección 1 de este documento): el TTS local sigue sin instalar, así
  que hoy la voz depende de `edge-tts` (nube) y cae a `espeak-ng` sin internet.
  Es el único punto del pipeline que todavía no es 100% local.

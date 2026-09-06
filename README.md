# Niri — asistente de voz local en español

Asistente de voz que corre **entero en la máquina**, sin enviar audio ni comandos a
ningún servicio externo. Escucha de forma continua esperando la palabra de activación
*"oye niri"*, transcribe la orden, la convierte en una acción validada contra un
esquema cerrado, y la ejecuta sobre un conjunto acotado de directorios y aplicaciones.

Pensado para un escritorio Linux de uso diario (desarrollado sobre CachyOS/Arch con
Hyprland, PipeWire y una GPU de 4 GB), funcionando como servicio de usuario `systemd`
sin privilegios de root.

**Este documento es la única fuente de verdad del proyecto.** El estado del sistema,
sus decisiones técnicas y sus pendientes se registran acá.

---

## Qué puede hacer

| Familia | Acciones | Módulo |
|---|---|---|
| Archivos | `crear_carpeta`, `crear_archivo`, `mover`, `eliminar`, `listar`, `leer` | `src/executor.py` |
| Aplicaciones | `abrir_aplicacion`, `enfocar_ventana`, `ejecutar_rutina` | `src/actions/apps.py` |
| Sistema | `volumen`, `brillo`, `captura_pantalla`, `wifi`, `bluetooth`, `energia` | `src/actions/system.py` |
| Música | `control_musica`, `reproducir_cancion` (MPRIS vía D-Bus) | `src/actions/media.py` |
| Tiempo | `hora`, `temporizador`, `alarma`, `recordatorio`, `nota` | `src/actions/time_actions.py` |
| Utilidades | `calculo`, `conversion`, `traduccion` | `src/actions/utils.py` |
| Charla | `saludo`, `chiste`, `despedida` | `src/actions/smalltalk.py` |

Las acciones de archivos van al `Executor` (tienen modelo de rutas y confirmación); el
resto se rutea por `src/actions/dispatch.py`, que existe justamente para no mezclar
los dos modelos de seguridad.

## Arquitectura

```
Mic → [Wake Word] → [VAD] → [STT] → [NLU → JSON] → [Validación] → [Ejecutor] → [TTS]
      (idle 24/7)   └──────────── activo solo tras la palabra de activación ─────────┘
```

| Capa | Implementación | Archivo |
|---|---|---|
| Captura de audio | `sounddevice` a 48 kHz + remuestreo a 16 kHz con `scipy` | `src/audio.py` |
| Wake word | Clasificador ONNX propio `oye_niri.onnx` vía `onnxruntime` | `src/wake_word.py` |
| VAD | Silero VAD (`silero_vad.onnx`) vía `onnxruntime` | `src/vad.py` |
| STT | `faster-whisper`, modelo `base`, `int8`, CPU | `src/stt.py` |
| NLU | Ollama + `qwen2.5:3b-instruct`, salida JSON forzada | `src/nlu.py` |
| Validación | Pydantic, enum cerrado de acciones | `src/schemas.py` |
| Ejecución | `pathlib`/`os` directo, whitelist de rutas | `src/executor.py`, `src/paths.py` |
| Confirmación | Interpretación de "sí"/"no" por voz | `src/confirm.py` |
| Auditoría | JSON Lines append-only | `src/audit.py` |
| TTS | edge-tts → Piper → espeak-ng, en cadena | `src/tts.py` |
| Orquestación | Máquina de estados `IDLE` / `LISTENING` | `src/main_loop.py` |

## Modelo de seguridad

El asistente borra y mueve archivos a partir de audio transcrito, así que el diseño
parte de asumir que **tanto el STT como el LLM se equivocan**. Ninguno de los dos es
autoridad de seguridad:

1. **El LLM nunca genera shell ni rutas crudas.** Devuelve JSON validado contra
   `FileAction`, con `action` restringida a un enum cerrado. Un valor fuera del enum
   no llega a ejecutarse.
2. **Whitelist de raíces**: las operaciones de archivos solo ocurren bajo `~/Proyectos`
   y `~/Clases`, con `Path.resolve()` neutralizando los escapes `../`.
3. **Las raíces completas no se borran ni se mueven**, aunque técnicamente pasen el
   chequeo de whitelist.
4. **Whitelist de aplicaciones**: el NLU devuelve un *alias*, nunca un comando. Si el
   alias no está en `APP_ALIASES`, la acción se rechaza.
5. **Los parámetros del LLM se normalizan a conjuntos cerrados** antes de usarse: por
   ejemplo `cantidad` se reduce a `subir`/`bajar`/`silenciar`, y nunca viaja literal
   hacia un proceso.
6. **Todo proceso externo se invoca con lista argv**, nunca con `shell=True`, y los
   cálculos aritméticos se evalúan recorriendo el AST (`_safe_eval`), nunca con
   `eval()`.
7. **Confirmación por voz obligatoria en `eliminar` y `mover`.** Ante silencio, timeout
   o respuesta ambigua, la acción se **cancela**: el diseño nunca asume "sí".
8. **Filtro de confianza del STT** (`STT_MIN_AVG_LOGPROB`). No es un extra: el
   `initial_prompt` que mejora la precisión también hace que Whisper alucine frases
   *del propio prompt* sobre audio vacío, y una de ellas es "Elimina el archivo".
9. **Log de auditoría append-only** (`logs/audit.log`): registra timestamp,
   transcripción, acción y resultado de *todo* intento, incluidos los bloqueados.
10. **Hardening del servicio** (`niri.service`): `ProtectSystem=strict`,
    `ProtectHome=read-only`, `NoNewPrivileges=true` y `ReadWritePaths` acotado. Es una
    segunda barrera, independiente del código.

## Requisitos

- Linux con PipeWire/PulseAudio y un micrófono. Las acciones de sistema y ventanas
  asumen Hyprland (`hyprctl`), `wpctl`, `brightnessctl`, `grim`, `nmcli`,
  `bluetoothctl`.
- Python 3.14 (el proyecto se desarrolló sobre esa versión; ver *Decisiones técnicas*).
- [Ollama](https://ollama.com) con un backend de GPU y el modelo `qwen2.5:3b-instruct`.
- `mpv` o `ffplay` para reproducir el audio del TTS.

## Instalación

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Modelo NLU
ollama pull qwen2.5:3b-instruct
```

Los modelos de wake word y VAD (`models/oye_niri.onnx`, `models/silero_vad.onnx`) ya
están incluidos en el repositorio.

**Antes de usarlo, ajustá las rutas permitidas** en `src/paths.py` (`ALLOWED_ROOTS`) y
las rutas del servicio en `niri.service`: hoy están escritas con rutas absolutas del
entorno de desarrollo.

### TTS local con Piper (opcional, recomendado)

Por defecto el TTS usa `edge-tts` (nube) cuando hay internet, porque suena mejor. Para
que el pipeline sea **100% local**:

```bash
.venv/bin/python -m pip install piper-tts
mkdir -p models/piper
V=https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_AR/daniela/high/es_AR-daniela-high
curl -L -o models/piper/es_AR-daniela-high.onnx      "$V.onnx"        # 114 MB
curl -L -o models/piper/es_AR-daniela-high.onnx.json "$V.onnx.json"
```

La voz es `daniela` (es_AR, mujer), elegida por parecerse a la de la nube. Hay
alternativas más baratas en el mismo repositorio: `es_MX/claude/high` pesa 60 MB y
sintetiza en 60 ms contra los 241 ms de esta. Las dos siguen siendo mucho más
rápidas que edge-tts (~900 ms), así que la diferencia no se nota hablando.

`src/tts.py` acepta **cualquier** `.onnx` que haya en `models/piper/`, así que la voz
se puede bajar con su nombre original sin renombrarla. Se carga en memoria al arrancar
(843 ms una vez) y a partir de ahí genera en **241 ms**: por subproceso sería ~4x más,
porque el modelo se relee en cada llamada.

Después, `ALLOW_CLOUD_TTS_FALLBACK=false`. Sin Piper y sin internet, el sistema cae a
`espeak-ng`, que siempre funciona pero suena robótico.

### Búsqueda de archivos (opcional)

Para que Niri responda "¿dónde dejé el resumen de sistemas operativos?" hace falta el
codificador de embeddings. Sin él la búsqueda igual funciona, pero solo por palabras
exactas (BM25):

```bash
mkdir -p models/e5-small
E=https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx
curl -L -o models/e5-small/model.onnx     "$E/model_qint8_avx512_vnni.onnx"  # 113 MB
curl -L -o models/e5-small/tokenizer.json "$E/tokenizer.json"                #  16 MB
curl -L -o models/e5-small/config.json    "$E/config.json"
```

Corre en CPU (935 textos/s, 3 ms por consulta) y **no toca la GPU**: los 4 GB ya están
comprometidos con el NLU y el escritorio. El índice se arma solo en un hilo de fondo y
vive en `data/indice.db`, que no se versiona. Solo indexa lo que hay bajo `~/Proyectos`
y `~/Clases`, y excluye a propósito `recordings/`, `logs/`, `data/` y `models/`: son
datos de runtime del asistente, no documentos, y `logs/audit.log` contiene
transcripciones que no tienen por qué copiarse a otro archivo.

## Configuración

Todo se ajusta por variables de entorno (ver `src/config.py`). Los valores por defecto
no son arbitrarios: salieron de mediciones sobre este hardware.

| Variable | Default | Notas |
|---|---|---|
| `WAKE_WORD_MODEL` | `models/oye_niri.onnx` | Clasificador propio |
| `WAKE_WORD_THRESHOLD` | `0.9` | Con `0.85` se activaba con sonidos cortos |
| `WAKE_WORD_TRIGGER_LEVEL` | `3` | Frames consecutivos (~80 ms c/u) sobre el umbral |
| `VAD_THRESHOLD` | `0.5` | Umbral recomendado por Silero; ajustar con cautela |
| `VAD_SILENCE_TIMEOUT_MS` | `800` | Silencio que cierra la captura |
| `AUDIO_GAIN` | `3.0` | Se aplica **solo al VAD**, no al STT (ver abajo) |
| `STT_MODEL_SIZE` | `base` | `small` no mejoró la precisión y calentaba el equipo |
| `STT_MIN_AVG_LOGPROB` | `-0.5` | Filtro anti-alucinación; defensa, no optimización |
| `STT_COMPUTE_TYPE` | `int8` | |
| `ALLOW_CLOUD_TTS_FALLBACK` | `true` | `false` fuerza pipeline 100% local |
| `PIPER_VOICE_MODEL` | `models/piper/es_AR-daniela-high.onnx` | Cualquier `.onnx` de esa carpeta sirve |

## Puesta en marcha

```bash
sudo systemctl start ollama          # backend NLU

mkdir -p ~/.config/systemd/user/
cp niri.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now niri.service
```

Diagnóstico:

```bash
systemctl --user status niri.service
journalctl --user -fu niri.service
journalctl --user -u niri.service | grep "Descartando segmento"   # baja confianza STT
journalctl -u ollama | grep "inference compute"                   # debe decir Vulkan
```

> **El micrófono es de acceso exclusivo.** Si el servicio está corriendo, ya tiene el
> dispositivo abierto: cualquier prueba manual de audio en paralelo falla. Hay que
> parar el servicio, probar, y volver a levantarlo.

## Rendimiento

Medido en el equipo de desarrollo (laptop con RTX 3050 de 4 GB):

| Etapa | Antes | Después |
|---|---|---|
| NLU, primera llamada en frío | 92.5 s | 25.3 s |
| NLU, turno normal (modelo caliente) | 4.3 s | 0.6–1.3 s |
| STT | 2.11 s (`small`) | 0.66 s (`base`) |
| TTS (generación) | 1.2 s | 0.6–1.2 s |
| **Total por turno** | **~7.7 s** | **~2.5 s** |

En reposo (`IDLE`, el 99% del tiempo) el consumo de CPU ronda el 17%: incluye el
polling de ALSA, el modelo de wake word y el remuestreo en tiempo real.

`keep_alive="30m"` en `src/nlu.py` es lo que evita pagar la carga en frío en cada
orden, a costa de mantener ~2 GB de VRAM reservados mientras el modelo está caliente.

## Decisiones técnicas

Cada una salió de un problema concreto y está verificada contra el hardware real:

**Python 3.14 sin TensorFlow.** `openwakeword` requiere `tflite-runtime`, que no tiene
wheels para 3.14. Se usa `pyopen-wakeword` y el clasificador propio corre con
`onnxruntime` en vez de TFLite, evitando tener que bajar a Python 3.12.

**Silero VAD integrado a mano.** `silero-vad-lite` fallaba al compilar vía CMake en
este entorno, así que se consume el `.onnx` nativo directamente con `onnxruntime`
(interfaz combinada v4/v5: entrada de 512 muestras @16 kHz, estado recurrente
`[2,1,128]`).

**Remuestreo al vuelo.** El micrófono UAC rechaza captura nativa a 16 kHz (ALSA
`PaErrorCode -9997`). Se captura a 48 kHz —soportado universalmente por hardware
USB— y se reduce en memoria con `scipy.signal.resample_poly` antes de las redes
neuronales, con penalidad mínima de CPU.

**La ganancia de audio solo afecta al VAD.** Se verificó transcribiendo la misma
grabación a 1x, 2x, 3x y normalizada a -3 dBFS: la salida de Whisper es idéntica en
los cuatro casos, porque trabaja sobre características log-mel esencialmente
invariantes a la amplitud. Silero VAD sí es sensible al nivel absoluto, que es
exactamente donde la ganancia está aplicada. **Los errores de transcripción de este
proyecto no se arreglan subiendo volumen**; se atacaron desde las opciones de
decodificación.

**Opciones de decodificación de Whisper.** `vad_filter=True` recorta el silencio y
elimina las alucinaciones sobre audio vacío (en los logs viejos aparecían frases del
set de entrenamiento como "¡Suscríbete!"). El `initial_prompt` usa **frases completas
con la misma estructura que habla el usuario**: contra una lista de vocabulario suelto
acertó 2 de 5 casos, con frases estructuradas 5 de 5 — Whisper imita el formato del
prompt. Eso corrigió errores persistentes como "archivo" → "al chivo" y "fotos" →
"Photos". `condition_on_previous_text=False` evita propagar errores entre turnos.

**Wake word entrenado con voces sintéticas.** `oye_niri.onnx` se entrenó con 8 voces
de Piper (Argentina/España/México) diciendo "oye niri", negativos sintéticos (el
vocabulario de comandos del asistente y frases parecidas tipo "oye mari") y
grabaciones reales del usuario como negativos de audio real. Validación en streaming:
15/15 positivos detectados, 32/33 negativos en silencio. En vivo dispara con scores de
0.902 a 0.989 sin falsos positivos. El pipeline de entrenamiento **no está en el
repositorio**: reentrenarlo requiere rearmarlo.

**El paquete `ollama` de Arch es solo-CPU.** No incluye ningún backend de GPU
(`/usr/lib/ollama/` trae únicamente `libggml-cpu-*.so`): con la GPU presente y su
driver funcionando, igual reportaba `library=cpu` y generaba a ~8 tokens/s. Se
resuelve instalando `ollama-vulkan` (~7 MB, usa el driver ya instalado). Se eligió
sobre `ollama-cuda` deliberadamente: el backend CUDA arrastra el paquete `cuda`
completo (~2.2 GB de descarga, 4.71 GB en disco) para una ganancia marginal en este
caso de uso. Por la misma razón se descartó acelerar el STT por GPU: `faster-whisper`
usa CTranslate2, que necesita cuDNN/cuBLAS reales, y en 4 GB de VRAM —donde el NLU ya
reserva ~2 GB— no vale la pena comprometer memoria que el equipo necesita para otras
cosas.

**`CPUQuota` demasiado bajo rompe el VAD.** Con `CPUQuota=25%` el cgroup throttleaba
en pleno ciclo de audio y el asistente "no escuchaba" pese a que el wake word sí
disparaba. Quedó en `CPUQuota=300%` con `CPUWeight=20`, que baja la prioridad relativa
cuando el equipo está bajo carga pero no estrangula el pipeline en tiempo real.

## Desarrollo y pruebas

```bash
PY=.venv/bin/python

# Pipeline completo por texto, sin hablarle al micrófono
$PY test_pipeline.py "Crea una carpeta llamada pruebas_nlu"

# Consumo de CPU en reposo
$PY test_phase1.py

# Captura + VAD (requiere el servicio detenido)
$PY test_mic_vad.py
```

## Estado actual y pendientes

El pipeline está completo y funcionando end-to-end. Lo único que falta para que sea
**100% local** es instalar Piper: hoy, con la configuración por defecto, la voz sale
por `edge-tts` (nube) cuando hay internet. Es el único punto del sistema donde algo
sale de la máquina, y lo único que se envía es el texto de la respuesta.

## Privacidad

El repositorio **excluye deliberadamente** los datos que genera el uso real:
`recordings/` (grabaciones de voz), `logs/audit.log` (transcripciones de todo lo
dicho), `data/` (notas y recordatorios personales). Si clonás esto, esos directorios
se crean vacíos.

## Créditos

Construido sobre [openWakeWord](https://github.com/dscripka/openWakeWord) /
`pyopen-wakeword`, [Silero VAD](https://github.com/snakers4/silero-vad),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[Ollama](https://ollama.com) con Qwen2.5, [Piper](https://github.com/rhasspy/piper) y
`edge-tts`. Consultá la licencia de cada proyecto upstream —y la de las voces de Piper
usadas para entrenar el wake word— antes de redistribuir los modelos incluidos acá.

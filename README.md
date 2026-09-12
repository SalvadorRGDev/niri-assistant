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

### Cómo se nombra una ubicación por voz

Las acciones de archivos ocurren siempre bajo una de las dos raíces de la whitelist,
`~/Proyectos` o `~/Clases` (ver *Modelo de seguridad*). Cualquier otra carpeta del home
—`~/Descargas`, `~/Documentos`— no resuelve **a propósito**: el asistente vuelve a
preguntar en vez de escribir fuera de la frontera.

Para llegar a una **subcarpeta** alcanza con nombrarla junto a la raíz. Estas tres formas
resuelven a `~/Clases/ADA`, las tres verificadas de punta a punta:

> «…en Clases, dentro de la carpeta ADA» · «…de Proyectos, carpeta trabajo» ·
> «…en la carpeta ADA de Clases»

La raíz tiene que aparecer: decir sólo «ADA» no alcanza, porque el asistente no sabe de
qué carpeta parte. El resolvedor admite varios niveles (`Clases, carpeta ADA, carpeta
parciales` → `~/Clases/ADA/parciales`) y corrige el nombre contra el disco: «ada» resuelve
a `ADA` si esa carpeta ya existe, y se conserva tal cual si todavía no existe, que es lo
que permite crearla.

Si no se dice ninguna ubicación, el asistente la pregunta, y **esa respuesta no pasa por
el LLM**: la interpreta directamente `src/paths.py`. Antes de crear algo repite en voz
alta la ruta final —«Voy a crear 'parciales' en Clases, carpeta ADA»—, así que conviene
escucharla: el NLU a veces traduce un nombre («trabajo» → `work`, en 1 de 8 frases
probadas) y esa carpeta no existiría.

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

**Medir el STT exige el conjunto completo y conocer el piso de ruido.** Whisper no es
determinista: la misma configuración corrida dos veces sobre las mismas 54 grabaciones
difiere en una. Cualquier comparación por debajo de ese piso no dice nada. Se probó
`beam_size=1` sobre 8 grabaciones —parecía 14% más rápido y 8/8 idéntico— y con las 54
dio 36/54 y encima más lento en total. Quedó en 5 con `cpu_threads=8`, que baja el
total un 8% sin salirse del ruido. `eval/test_stt.py` fija el conjunto y compara por
hashes para no guardar transcripciones.

**Las similitudes del codificador se leen por orden, no por valor.** Con e5 todo cae
cerca de 0.8: sobre las carpetas reales, el mejor resultado supera a la mediana por
+0.036 cuando el archivo existe y +0.033 cuando no. Ningún umbral separa "lo encontré"
de "no está", así que la señal es el ranking más el dato binario del BM25 (si ninguna
palabra dicha aparece en ningún documento, no hay nada). De ahí el campo `literal` de
cada resultado.

**En Piper el costo es cargar el modelo, no sintetizar.** Por subproceso son ~826 ms
por frase y casi no dependen de su largo, porque el `.onnx` se relee en cada llamada.
Cargándolo una vez en memoria, la síntesis baja a 62 ms con la voz mexicana y 241 ms
con la argentina. Por eso `src/tts.py` usa la API en proceso y no el binario, y por eso
no hace falta cachear las respuestas fijas.

**Comparar modelos en Ollama exige liberar la VRAM primero.** El 1.5B medido con el 3B
todavía cargado corrió 68% en CPU y dio 1272 ms; con la GPU libre, 194 ms. `ollama ps`
lo muestra en la columna PROCESSOR: si no dice "100% GPU", la medición no vale.

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

Los comandos van con la ruta completa al intérprete del venv en vez de una variable
`PY=...`: la shell del proyecto es fish, donde esa asignación es un error de sintaxis.

```bash
# Verificación: ninguna de estas toca el micrófono ni la red
.venv/bin/python eval/run.py              # 69 órdenes contra el baseline; sale 1 si hay regresiones
.venv/bin/python eval/test_seguridad.py   # 46 checks de los invariantes de seguridad
.venv/bin/python eval/test_audio.py       # 22 checks de la selección de dispositivo de entrada
.venv/bin/python eval/test_stt.py         # regresión del STT sobre las grabaciones reales
.venv/bin/python eval/resumen_metricas.py # p50/p95 por etapa sobre el uso real

# Pipeline completo por texto, sin hablarle al micrófono
.venv/bin/python test_pipeline.py "Crea una carpeta llamada pruebas_nlu"

# Consumo de CPU en reposo
.venv/bin/python test_phase1.py

# Captura + VAD (requiere el servicio detenido)
.venv/bin/python test_mic_vad.py
```

### Diagnóstico del wake word

**Síntoma:** el asistente se despierta solo y no entiende nada. Medido con el
micrófono interno: **38 de 40 activaciones sin ninguna transcripción**, con puntajes
de hasta 0.986 — más altos que muchos aciertos reales, así que subir
`WAKE_WORD_THRESHOLD` no alcanza. `AUDIO_GAIN` tampoco: con 1.0 o 3.0, Silero
clasifica ese ruido como voz igual. La causa es que `oye_niri.onnx` se entrenó con
voces sintéticas y negativos del micrófono USB, y el interno tiene otra respuesta.

**El síntoma inverso es el mismo problema.** El 2026-09-08, con el micrófono interno,
los logs de una mañana muestran **32 frames por encima de 0.9 y una sola activación**:
el modelo roza el umbral en picos aislados y casi nunca sostiene los tres frames
seguidos que exige `WAKE_WORD_TRIGGER_LEVEL`. Desde el lado del usuario eso se ve
como "el umbral está muy alto y no me reconoce", y desde los datos es el mismo
desajuste entre el modelo y este micrófono, no un valor mal elegido. En la única
activación de esa mañana el VAD tampoco encontró voz (prob. máx. 0.351), y el nivel
de captura venía cayendo: las utterances pasaron de ~-21 dBFS (28-ago a 5-sep) a
-28/-32 dBFS (6 y 7-sep). Los controles de ALSA explican parte: `Capture` e
`Internal Mic Boost` están al máximo, pero `Digital` está a mitad de rango (0 dB de
30 disponibles). Subirlo recupera nivel para el VAD y el STT **y también amplifica el
ruido que dispara los falsos positivos**, así que conviene tocarlo después de medir,
no antes, para no mover dos variables a la vez.

**Trampa: una muestra puede tener buen nivel y no contener voz.** Las dos primeras
grabaciones de `recordings/wakeword_positivos/` marcaban -19 y -13 dBFS —niveles
sanos— pero tenían el 70% y el 93% de su energía por debajo de 100 Hz: rumble de
manipular la laptop. Whisper las transcribe vacías y el wake word les da 0.15, o sea
que parecían "positivos que el modelo no reconoce" y llevaban directo a la conclusión
"hay que reentrenar". Desde 2026-09-08, `eval/calidad_audio.py` decide por forma
espectral (≥15% de la energía en 300-1000 Hz, ≤40% por debajo de 100 Hz):
`grabar_wakeword.py` descarta y repite la toma, y `analizar_wakeword.py` excluye del
análisis las muestras mudas que ya estén guardadas.

El procedimiento de abajo decide **con datos** entre las dos únicas salidas: ajustar
dos valores, o reentrenar.

**Veredicto (2026-09-08): hay que reentrenar.** Con 32 positivos válidos del
micrófono interno contra 18 falsos disparos reales, el ordenamiento da **AUC 0.307**:
tomados un positivo y un negativo al azar, el modelo le da más puntaje al ruido el
69% de las veces. Eso no es un umbral mal elegido —ningún umbral reordena una
lista— y se ve en las medianas de pico, 0.961 los positivos contra 0.983 los
negativos. La configuración de hoy (0.9 / 3 frames) detecta 7 de 32 "oye niri" y
deja pasar 10 de 18 falsos, que es exactamente el síntoma doble que reporta el
usuario. Y no hay adónde moverse: en toda la grilla, **ningún** par que no deje pasar
falsos detecta un solo positivo. Endurecer a 0.9 / 4 frames baja los falsos a 2 de 18
pero también los aciertos a 3 de 32.

Se descartaron dos explicaciones más baratas antes de firmarlo:

- **No es el nivel de captura.** Normalizar las grabaciones a -20 dBFS RMS mueve el
  AUC de 0.307 a 0.316. Los positivos se grabaron a -25/-34 dBFS y el modelo los
  sigue ordenando por debajo del ruido.
- **No es el modelo, es a qué se parece.** Sintetizando "oye niri" con las voces
  Piper del repo, el mismo `oye_niri.onnx` puntúa **1.000** con 3-4 frames seguidos
  por encima de 0.9. El clasificador funciona perfecto: aprendió las voces
  sintéticas con las que se lo entrenó, no la palabra dicha por una persona ante
  este micrófono. Reentrenar con positivos reales es exactamente lo que falta.

**Defecto encontrado de paso, ya corregido (2026-09-08):** `src/audio.py` bajaba de
48 kHz a 16 kHz con `raw_data[::factor]`, o sea quedándose con una de cada tres
muestras **sin filtro antialias**. Todo lo que el micrófono captaba por encima de
8 kHz no se perdía: se plegaba sobre la banda de voz, donde ya no se distingue de la
señal real. Como cuánto se pliega depende de la respuesta de agudos de cada
micrófono, esto además volvía el audio del interno distinto del USB con el que se
entrenó el wake word.

Ahora el callback filtra antes de decimar, con un FIR de 161 coeficientes y corte en
7.5 kHz: deja la banda que se pliega **53.8 dB abajo** con 0.02 dB de rizado hasta
6 kHz, y cuesta **20 µs por bloque, 0.20% de un núcleo** (medido; el costo lo domina
la llamada, no la cantidad de coeficientes). El estado del filtro y la fase de la
decimación se conservan entre bloques, porque PortAudio no garantiza un tamaño de
bloque múltiplo de 3 y perder la fase correría la rejilla de muestreo en cada borde.
Cubierto por cinco verificaciones nuevas en `eval/test_audio.py` (27 en total), que
no abren el micrófono: un tono de 10 kHz que antes reaparecía en 6 kHz ahora queda
67.6 dB más abajo, y una grabación real de voz atraviesa el filtro con correlación
0.994 y sin cambio de nivel.

**Lo que sigue sin medirse** es cuánto ensuciaba realmente ese alias en este
micrófono. No se puede sacar de las grabaciones guardadas: están todas en 16 kHz, o
sea ya decimadas, y un alias no se deshace. Tampoco sirve la prueba con Piper — una
voz sintética a 22 kHz casi no tiene energía por encima de 8 kHz, y por eso puntúa
1.000 igual por los dos caminos. Hace falta capturar a 48 kHz sin decimar, que es lo
que hace `eval/medir_alias.py`:

```bash
systemctl --user stop niri.service; .venv/bin/python eval/medir_alias.py; systemctl --user start niri.service
```

Graba dos escenas de 10 s —**ambiente** sin hablar, que es la condición de los falsos
disparos, y **voz** diciendo "oye niri", que es la que tiene que funcionar—, las deja
en `recordings/captura_48k/` y, sobre el mismo audio, compara los dos caminos. De
cada escena informa qué porcentaje de la energía estaba por encima de 8 kHz, cuánto
de eso caía sobre la banda de voz al plegarse, la relación alias/señal en dB, y los
puntajes del wake word por los dos caminos —que es lo que dice si esa basura movía la
decisión o no—. Las capturas se reanalizan sin micrófono con `--solo-analizar`.

**Resultado (2026-09-08): el alias no era el problema.** Sobre 10 s de voz real, la
basura plegada quedaba **37.1 dB por debajo** de la señal —solo el 0.02% de la energía
estaba por encima de 8 kHz— y el wake word puntúa **0.997 por los dos caminos**,
idéntico. Sobre 10 s de ambiente la relación sube a -11.2 dB, pero es el cociente
entre dos números diminutos: la captura está a -53.6 dBFS, o sea silencio, el pico de
agudos cae en 9.9 kHz y al plegarse aterriza en 6.1 kHz, fuera de la banda de voz. El
puntaje del wake word se mueve de 0.595 a 0.601. Sacar el alias fue higiene necesaria
antes de reentrenar, pero no explica ni los falsos disparos ni las detecciones
perdidas.

**Lo que sí destapó la captura de voz.** Diez segundos continuos, que es la condición
real de producción y no la de los clips de 2 s, contienen 6 tramos con habla según
Silero (el último cortado por el final de la grabación). Los picos del wake word,
tramo por tramo: 0.997, 0.942, 0.876, 0.899, 0.954, 0.014. Tres pasan de 0.9 pero
**uno solo sostiene los tres frames seguidos**, así que hay **1 detección de 6**. Es
el mismo 22% que dieron los 32 positivos grabados por separado, ahora sin ningún
artefacto de recorte de por medio: el veredicto de reentrenar no dependía de cómo se
midió.

```bash
# 1. Los negativos se juntan solos: cada disparo que no produce una orden guarda
#    los 2 s previos en recordings/falsos_positivos/. Cuanto más corra, mejor.

# 2. Grabar los positivos (~3 min). Es lo único que hace falta de tu parte.
systemctl --user stop niri.service
.venv/bin/python eval/grabar_wakeword.py        # o --cantidad 40
systemctl --user start niri.service

# 3. Decidir.
.venv/bin/python eval/analizar_wakeword.py
```

El análisis busca en una grilla de umbral × frames consecutivos si existe algún par
que detecte **todos** los positivos sin dejar pasar **ninguno** de los negativos:

- **"Hay separación limpia"** → imprime los dos valores para `src/config.py`
  (`WAKE_WORD_THRESHOLD` y `WAKE_WORD_TRIGGER_LEVEL`) y ahí termina.
- **"No hay ningún par que separe"** → queda demostrado que hay que reentrenar con
  negativos de este micrófono, que son justo los que se están juntando. El pipeline
  de entrenamiento **no está en el repo**: rearmarlo es una sesión completa.

Con menos de ~25 positivos el resultado no es concluyente. Una muestra chica ya
produjo una conclusión falsa en este proyecto (ver `beam_size` más arriba).

**Pista registrada:** los negativos recolectados tienen picos de 0.944 a 0.997,
mientras que grabaciones de puro ruido ambiente quedan en 0.15-0.40, y los disparos
ocurren cada ~5 minutos. No es "cualquier ruido": hay algo específico y periódico del
ambiente que el modelo confunde con "oye niri". Identificarlo es media respuesta.

**Dos trampas al puntuar clips** (de `eval/analizar_wakeword.py`, 2026-09-08). El primer
chunk devuelve 76 puntajes de golpe, porque el buffer del modelo llega pre-rellenado:
suponer una cadencia fija desalinea cualquier eje de tiempo que se construya con ellos
—los picos y las rachas, en cambio, no se ven afectados—. Y la última ventana termina
donde termina el archivo, así que una palabra pegada al final nunca se ve entera: el
mismo clip pasa de 0.002 a 0.962 agregándole 1 s de silencio a cada lado. Por eso el
analizador rellena los clips antes de puntuarlos.

## Estado actual y pendientes

El pipeline está completo y funcionando end-to-end, con 142 verificaciones
automáticas cubriéndolo (69 casos de comprensión, 46 de seguridad, 27 de audio, más
la regresión del STT).

**Lo que está abierto, por orden de importancia:**

1. **Reentrenar `oye_niri.onnx` con voz real de este micrófono.** Es lo único que hoy
   impide usar el asistente cuando el USB no está conectado, y desde el 2026-09-08
   está medido, no supuesto: ver el veredicto en *Diagnóstico del wake word*. Los
   dos lados del conjunto de entrenamiento ya existen —32 positivos en
   `recordings/wakeword_positivos/`, 18 negativos en `recordings/falsos_positivos/`,
   y los negativos siguen juntándose solos—. Lo que falta es el pipeline de
   entrenamiento, que **no está en el repo**: rearmarlo es una sesión completa.
   La decimación sin antialias de `src/audio.py`, que habría congelado una
   distorsión dentro del modelo nuevo, ya está corregida.
2. **`VAD_SILENCE_TIMEOUT_MS` en 800 ms.** Es el bloque más grande de cada turno y
   es espera pura. Bajarlo a 600 son 200 ms fijos de mejora, a riesgo de cortar
   frases con pausa natural; hay que probarlo con `test_mic_vad.py`.
3. **`qwen2.5:1.5b` como alternativa.** Ahorra 969 MiB de VRAM y es 39% más rápido,
   pero contesta 65/69 en vez de 69/69 y, sobre todo, inventa una acción en vez de
   decir `ninguna` ante pedidos fuera de alcance. Está medido y documentado; el
   cambio es de una línea si algún día la VRAM importa más.

**Sobre la frontera de nube:** Piper está instalado y anda (62 ms con la voz mexicana,
241 ms con la rioplatense), pero `edge-tts` sigue siendo la voz primaria mientras haya
internet, por preferencia de voz. Poniendo `ALLOW_CLOUD_TTS_FALLBACK=false` el sistema
queda **100% local** sin perder ninguna función. Lo único que sale de la máquina hoy es
el texto de la respuesta hablada.

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

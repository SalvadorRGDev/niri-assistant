# Documentación Final — Asistente de Voz "Salvador"

Este documento cubre la conclusión de las fases 2 a la 6 de la arquitectura de Salvador, logrando un asistente de voz completamente funcional, en español, privado y 100% local.

## Estado de Módulos Implementados

### 1. Wake Word y VAD (Fase 1)
Implementados y operando a <20% de CPU de manera continua a través de remuestreo (48kHz a 16kHz al vuelo).
- **VAD:** `src/vad.py` usa **Silero VAD real** vía `onnxruntime` puro sobre `models/silero_vad.onnx` (interfaz combinada v4/v5: input 512 muestras @16kHz, estado recurrente `[2,1,128]`). Antes de esta revisión el código usaba `webrtcvad` pese a que este documento y `DOCUMENTACION_FASE1.md` ya describían Silero — quedó corregido y validado con audio real de `recordings/`.
- **Wake Word:** en producción con el modelo custom **`models/oye_niri.onnx`** (ya no el genérico `hey_jarvis`, ni el `ey_salvador.onnx` que figuraba en el plan original). Entrenado con voces sintéticas de Piper en español y ejecutado vía `onnxruntime`. **Validado en vivo con la voz real del usuario** (scores 0.902–0.989, sin falsos positivos), con `WAKE_WORD_THRESHOLD=0.9` y `WAKE_WORD_TRIGGER_LEVEL=3`. Detalle de entrenamiento en `README_PENDIENTES.md`.

### 2. STT (Fase 2)
Implementado en `src/stt.py` usando `faster-whisper`.
- Modelo configurado: **`base`** (antes `small`). Medido en este equipo: `base`
  transcribe una orden típica en **0.66s** contra los **2.11s** de `small`, y
  evita los picos de CPU a ~90°C que `small` provocaba en el laptop.
- Modo `cpu` con computación `int8`.
- Transcribe audio en fragmentos (utterances) capturados tras la palabra de activación.

**Opciones de decodificación** (todas medidas contra grabaciones reales de `recordings/`):

- **`vad_filter=True`**: recorta el silencio antes de decodificar. Además de
  acelerar, elimina las alucinaciones de Whisper sobre audio vacío — en los logs
  viejos aparecían frases de sus datos de entrenamiento como "¡Suscríbete!" y
  "Subtítulos realizados por la comunidad de Amara.org".
- **`initial_prompt`**: sesga la decodificación hacia las órdenes reales. Se
  probaron cinco variantes: una lista de vocabulario suelto ("carpeta, archivo,
  fotos…") acertó 2 de 5 casos, mientras que **frases completas con la misma
  estructura que habla el usuario** acertaron 5 de 5. La estructura importa más
  que la cantidad de palabras: Whisper imita el formato del prompt. Esto corrigió
  el error persistente donde "archivo" se decodificaba como "al chivo" y "fotos"
  como "Photos".
- **`condition_on_previous_text=False`**: cada orden se decodifica independiente,
  para no propagar errores de un turno al siguiente.
- **Filtro de confianza `STT_MIN_AVG_LOGPROB` (-0.5)**: descarta segmentos por
  debajo de ese `avg_logprob`. Es una **defensa necesaria**, no un extra: el
  `initial_prompt` mejora la precisión pero hace que el modelo alucine frases
  *del propio prompt* cuando el audio está vacío, y una de ellas es "Elimina el
  archivo", que es destructiva. Medido sobre audio real, el habla clara cae entre
  -0.08 y -0.30 mientras que las alucinaciones caen en -0.65 o peor, así que el
  corte separa ambos grupos con margen. Los segmentos descartados se registran
  como `WARNING`; `src/main_loop.py` trata el resultado vacío como "no entendí" y
  vuelve a `IDLE` sin actuar.

Nota sobre el nivel de audio: `AUDIO_GAIN` **no** se aplica a la ruta de STT, solo
al VAD (`src/vad.py`), y eso es correcto. Se verificó experimentalmente que
transcribir la misma grabación a 1x, 2x, 3x o normalizada a -3 dBFS produce
salida idéntica — Whisper usa características log-mel que son esencialmente
invariantes a la amplitud. Silero VAD sí es sensible al nivel, que es por lo que
la ganancia existe ahí. Ver `DOCUMENTACION_FASE1.md` para las mediciones.

### 3. NLU y Schema Pydantic (Fase 3)
Implementado en `src/nlu.py` y `src/schemas.py`.
- Llama localmente a `Ollama` utilizando el modelo **`qwen2.5:3b-instruct`**
  (este documento decía antes `7b`, pero el código siempre usó el `3b`).
- **Requiere que Ollama corra sobre GPU** para tener latencia usable — ver la
  sección "Rendimiento y aceleración por GPU" más abajo.
- Fuerzamente tipado: `Ollama` devuelve una cadena JSON estrictamente validada contra `FileAction` (Pydantic), limitando las intenciones posibles y proveyendo un marco altamente seguro.

### 4. Ejecutor y Seguridad (Fase 4)
Implementado en `src/executor.py`.
- **Restricción de Path:** Solo se permite operar en subcarpetas de `~/Proyectos` y `~/Clases`. Los intentos de escape de directorio `../` son neutralizados resolviendo la ruta con `pathlib.Path.resolve()`. Tampoco se permite eliminar/mover un directorio raíz completo (`~/Proyectos` o `~/Clases`), aunque técnicamente pase el chequeo de whitelist.
- **Confirmación por voz obligatoria en `eliminar`/`mover`:** Salvador pregunta y espera la respuesta ("sí"/"no", interpretada en `src/confirm.py`) antes de tocar disco — orquestado en `src/main_loop.py`. Ante silencio, timeout o respuesta ambigua, la acción se **cancela** (diseño fail-safe: nunca asume "sí").
- **Resolución de ubicación por voz** (`src/paths.py`): el NLU ya no adivina una carpeta por defecto — si el usuario no dijo dónde (p.ej. "crea una carpeta llamada fotos" sin más), Salvador pregunta "¿en qué carpeta?" (Proyectos/Clases, con subcarpetas opcionales) y **confirma repitiendo la ruta completa** antes de crear algo. Para `eliminar`/`mover`/`leer` sobre algo que ya existe, primero lo busca en las raíces permitidas (nivel superior); si aparece en más de una o en ninguna, pregunta dónde está antes de continuar. El mensaje de confirmación de `eliminar`/`mover` ahora incluye la ruta completa, no solo el nombre.
- **Acciones `mover` y `leer`** implementadas (antes devolvían "todavía no sé hacer eso"). `mover` siempre coloca el origen *dentro* de la carpeta destino (creándola si no existe) y nunca sobrescribe algo con el mismo nombre que ya esté ahí.
- **Log de auditoría append-only** (`src/audit.py` → `logs/audit.log`, JSON Lines): registra timestamp + transcripción + acción + resultado de *todo* intento, incluidos los bloqueados por seguridad y las confirmaciones pendientes/canceladas.

### 5. Text-To-Speech (Fase 5)
Implementado en `src/tts.py`, en cadena de motores:
1. **edge-tts** (nube, voz de mujer "Elena", Argentina): preferido por defecto (`ALLOW_CLOUD_TTS_FALLBACK=true`) cuando hay internet, para sonar más amigable. Timeout de 8s: si la red falla o está lenta, cae al siguiente motor.
2. **Piper** (local): calidad neuronal, totalmente offline. Requiere instalar el paquete y descargar una voz ES — ver `README_PENDIENTES.md`. Se usa si no hay internet o `ALLOW_CLOUD_TTS_FALLBACK=false`.
3. **espeak-ng** (último recurso): siempre local, calidad robótica.
- Se integra con `mpv` o `ffplay` sin mostrar ventanas, garantizando una latencia percibida muy baja.

## Rendimiento y aceleración por GPU

### El paquete de Ollama por defecto no usa la GPU

En Arch/CachyOS, el paquete `ollama` es la build **solo-CPU**: no incluye ningún
backend de GPU (`/usr/lib/ollama/` trae únicamente `libggml-cpu-*.so`). Con una
RTX 3050 presente y su driver funcionando, Ollama igual reportaba
`inference compute id=cpu library=cpu` en sus logs y generaba a ~8 tokens/segundo.

La solución fue instalar el backend por separado:

```bash
sudo pacman -S ollama-vulkan     # ~7 MB, usa el driver NVIDIA ya instalado
sudo systemctl restart ollama
```

Se eligió **`ollama-vulkan` sobre `ollama-cuda`** deliberadamente: el backend CUDA
arrastra el paquete `cuda` completo (~2.2 GB de descarga, 4.71 GB en disco) para
una ganancia marginal sobre Vulkan en este caso de uso. Por la misma razón se
descartó acelerar el STT por GPU: `faster-whisper` usa CTranslate2, que necesita
cuDNN/cuBLAS reales (Vulkan no le sirve), y en una tarjeta de solo 4 GB de VRAM
—donde el modelo NLU ya reserva ~2 GB mientras está caliente— no vale la pena
comprometer la memoria de video que el equipo necesita para otras tareas.

Para verificar que la GPU está en uso:
```bash
journalctl -u ollama | grep "inference compute"
# debe decir library=Vulkan y nombrar la GPU, no library=cpu
```

### Latencias medidas

| Etapa | Antes | Después |
|---|---|---|
| NLU (Ollama), primera llamada en frío | 92.5s | 25.3s |
| NLU, turno normal (modelo caliente) | 4.3s | 0.6–1.3s |
| STT | 2.11s (`small`) | 0.66s (`base`) |
| TTS (generación edge-tts) | 1.2s | 0.6–1.2s |
| **Total por turno** | **~7.7s** | **~2.5s** |

El `keep_alive="30m"` en `src/nlu.py` es lo que evita pagar la carga en frío en
cada orden: mantiene el modelo residente en VRAM entre invocaciones.

## Puesta en Marcha en Producción (Fase 6)

Para habilitar a Niri como un servicio persistente transparente que se ejecute en el fondo (sin interfaz gráfica y tolerante a fallos):

### 1. Iniciar los Servicios Dependientes
Asegúrate de que el backend NLU está corriendo:
```bash
sudo systemctl start ollama
```
*(Opcional: puedes habilitarlo permanentemente con `sudo systemctl enable ollama`)*

### 2. Configurar el Servicio de Usuario `systemd`
El archivo `niri.service` ya posee las reglas de hardening de sistema, limitando la memoria y acceso de escritura solo a las carpetas configuradas.
Ejecuta:
```bash
# Crear directorio si no existe
mkdir -p ~/.config/systemd/user/

# Copiar el servicio
cp niri.service ~/.config/systemd/user/

# Recargar demonios
systemctl --user daemon-reload

# Habilitar para que inicie en el arranque de sesión y levantar ahora
systemctl --user enable --now niri.service
```

### 3. Diagnósticos y Logs
Si algo falla o Niri no responde:
```bash
# Ver el estado actual
systemctl --user status niri.service

# Leer el registro (logs) en tiempo real
journalctl --user -fu niri.service

# Confirmar qué modelo de Whisper cargó la instancia en ejecución
journalctl --user -u niri.service | grep "Loading Whisper model"

# Ver qué transcripciones se descartaron por baja confianza
journalctl --user -u niri.service | grep "Descartando segmento"
```

**Importante para pruebas manuales:** si el servicio está activo, ya tiene el
micrófono abierto. Correr `python main.py` a mano en paralelo produce un
conflicto por el dispositivo de audio. Hay que parar el servicio primero
(`systemctl --user stop niri.service`), probar, y levantarlo al terminar.

## Pruebas de Desarrollo
Para evitar tener que hablar al micrófono al diagnosticar la interpretación o síntesis, usa el pipeline de texto:
```bash
python test_pipeline.py "Crea una carpeta llamada pruebas_nlu"
```

# Plan de Implementación — Asistente de Voz "Salvador"

> ⚠️ **DOCUMENTO HISTÓRICO — no describe el estado actual del sistema.**
>
> Este es el plan original con el que arrancó el proyecto, y se conserva como
> registro de las decisiones de diseño y su justificación. Varias cosas cambiaron
> durante la implementación:
>
> | En este plan | En el sistema real hoy |
> |---|---|
> | Asistente "Salvador" | Se llama **Niri** (wake word "oye niri", `niri.service`) |
> | Wake word `ey_salvador.onnx` | `models/oye_niri.onnx`, umbral 0.9 + 3 frames |
> | STT modelo `small` | Modelo **`base`**, con `vad_filter`, `initial_prompt` y filtro de confianza |
> | NLU `qwen2.5:7b-instruct` | **`qwen2.5:3b-instruct`**, sobre GPU vía `ollama-vulkan` |
> | JSON forzado con GBNF/`outlines` | `format=` del cliente de Ollama + validación Pydantic |
> | 100% local | El TTS usa `edge-tts` (nube) por defecto; Piper sigue pendiente |
>
> **Para el estado actual, ver `DOCUMENTACION_FINAL.md`** (arquitectura, rendimiento
> y puesta en marcha), `DOCUMENTACION_FASE1.md` (audio, wake word y VAD) y
> `README_PENDIENTES.md` (lo que falta).

**Stack:** Python 3.11+ | CachyOS (Arch) | 100% local | systemd user service

## 0. Arquitectura

```
Mic → [Wake Word] → [VAD] → [STT] → [NLU→JSON] → [Validador] → [Ejecutor] → [TTS]
      (idle 24/7)    └────────── activo solo tras wake word ──────────┘
```

---

## 1. Wake Word — `openWakeWord`
- **Por qué:** ONNX ~1-2MB, CPU <5% en reposo, Apache-2.0, entrenable con pocas muestras sintéticas.
- **Riesgo:** falsos positivos → umbral ≥0.85 + confirmación VAD antes de disparar STT.
- **Config:** modelo custom `ey_salvador.onnx`, chunk 1280 (80ms @16kHz).

## 2. VAD — `Silero VAD`
- **Por qué:** ~1MB, MIT, más preciso que WebRTC-VAD con ruido de fondo.
- **Config:** corte tras 800ms de silencio continuo.

## 3. STT — `faster-whisper` (modelo `small`, int8, `language="es"`)
- **Por qué:** mejor precisión/latencia en español que Vosk; CPU-friendly vs Whisper original.
- **Optimización:** carga perezosa (solo tras wake word), descarga tras timeout de inactividad.

## 4. NLU → JSON — `Ollama` + `qwen2.5:7b-instruct`
- **Por qué:** local, buen español, soporta salida estructurada.
- **Regla de oro:** el LLM **nunca** genera shell. Solo JSON validado contra schema Pydantic cerrado (enum de acciones: `crear_carpeta`, `crear_archivo`, `mover`, `eliminar`, `listar`, `leer`).
- **Técnica:** forzar grammar/JSON schema (GBNF vía `llama-cpp-python` u `outlines`) para impedir campos fuera del enum.
- **Riesgo:** prompt injection por voz → mitigado en capa 5, no aquí. El LLM no es la autoridad de seguridad.

```python
class FileAction(BaseModel):
    action: Literal["crear_carpeta","crear_archivo","mover","eliminar","listar","leer"]
    ruta_base: str
    nombre: Optional[str] = None
    destino: Optional[str] = None
    contenido: Optional[str] = None
```

## 5. Ejecución Validada — capa crítica de seguridad
- **Nunca** `eval`/`exec`/`subprocess(shell=True)` con datos del LLM. Solo `pathlib`/`os` directo.
- **Whitelist de directorios raíz** (`~/Clases`, `~/Documentos`, …) + `Path.resolve()` para neutralizar `../`.
- **Confirmación por voz obligatoria** en `eliminar`/`mover`.
- **Log de auditoría** append-only (timestamp + transcripción + acción).

```python
def resolve_safe_path(raw: str) -> Path:
    p = Path(raw).expanduser().resolve()
    if not any(str(p).startswith(str(r)) for r in ALLOWED_ROOTS):
        raise UnsafePathError(p)
    return p
```

## 6. TTS — `Piper`
- **Por qué:** casi tiempo real en CPU, voces `es_MX`/`es_ES` ligeras (20-60MB), MIT.
- Texto del LLM va como **dato de entrada** al proceso TTS, nunca como comando interpretado.

## 7. Autostart — `systemd --user`
- Sin root. Hardening como segunda barrera independiente del código:

```ini
[Service]
ExecStart=/home/USUARIO/salvador/.venv/bin/python main.py
Restart=on-failure
CPUQuota=15%
MemoryMax=800M
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/home/USUARIO/Clases /home/USUARIO/Documentos
NoNewPrivileges=true

[Install]
WantedBy=default.target
```
`enable --now` + `loginctl enable-linger USUARIO` para arranque sin sesión gráfica.

---

## Resumen de stack

| Capa | Herramienta | Criterio clave |
|---|---|---|
| Wake word | openWakeWord | CPU-only, libre, entrenable |
| VAD | Silero VAD | Ligero, preciso |
| STT | faster-whisper (small, int8) | Precisión ES en CPU |
| NLU | Ollama + Qwen2.5 + JSON schema forzado | Local, estructurado |
| Validación | Pydantic + whitelist rutas | Autoridad real de seguridad |
| Ejecución | pathlib/os directo | Sin inyección de shell |
| TTS | Piper | Rápido, voces ES |
| Autostart | systemd --user + sandboxing | Persistente, hardened |

## Roadmap sugerido
1. Wake word + VAD funcionando en loop (validar consumo CPU en reposo).
2. STT + captura de utterance.
3. Schema Pydantic + prompt NLU + grammar forzado.
4. Ejecutor con whitelist + confirmación por voz.
5. TTS + integración completa end-to-end.
6. systemd service + hardening + pruebas de arranque.

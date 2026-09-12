"""Captura explícita de sesiones nuevas para validar candidatos de wake word.

El WAV nunca se sobrescribe; cada sesión genera su propio manifiesto editable.
No detiene servicios, no cambia el dispositivo por fallback, no revisa etiquetas
automáticamente. El operador debe escuchar los WAV y revisar el manifiesto.
"""
import argparse
import datetime
import hashlib
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from training.wakeword import CHUNK, FORMAT_VERSION, SAMPLE_RATE, json_write, sha256  # noqa: E402


def check_service_stopped():
    """Estado desconocido también bloquea: no implica micrófono disponible."""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show", "niri.service", "--property=ActiveState", "--value"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("No se pudo comprobar niri.service; no se abrirá el micrófono.") from exc
    state = result.stdout.strip()
    if result.returncode != 0 or state not in ("inactive", "failed"):
        raise ValueError(f"niri.service no está confirmado inactivo (estado: {state or 'desconocido'}). "
                         "Detenerlo antes de capturar.")


def explicit_device(index):
    import sounddevice as sd
    from src.audio import _es_hardware_real

    if index < 0:
        raise ValueError("El índice de dispositivo debe ser explícito y no negativo.")
    try:
        device = sd.query_devices(index)
    except sd.PortAudioError as exc:
        raise ValueError("No se pudo consultar el dispositivo seleccionado.") from exc
    if device["max_input_channels"] < 1 or not _es_hardware_real(device["name"]):
        raise ValueError("Seleccioná un micrófono físico '(hw:...)' con entrada. "
                         "Los plugins/default no identifican qué micrófono se está midiendo.")
    return device["name"]


def capture(stream, path, seconds):
    """Escribir mientras se captura; conservar también un WAV parcial si falla."""
    wanted = round(seconds * SAMPLE_RATE)
    captured = 0
    digest = hashlib.sha256()
    with path.open("xb") as raw:
        with wave.open(raw, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            while captured < wanted:
                audio = stream.read_chunk(min(CHUNK, wanted - captured))
                if len(audio) == 0:
                    raise ValueError("El micrófono devolvió un bloque vacío.")
                audio = audio[:wanted - captured]
                content = audio.astype("<i2", copy=False).tobytes()
                handle.writeframesraw(content)
                digest.update(content)
                captured += len(audio)
    return captured / SAMPLE_RATE, digest.hexdigest()


def run(args):
    from src.audio import AudioStream

    check_service_stopped()
    name = explicit_device(args.device)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.session):
        raise ValueError("La sesión debe tener 1–64 letras, números, guiones o guiones bajos.")
    if args.count < 1 or args.count > 200 or not 0 < args.seconds <= 7200:
        raise ValueError("Usá 1–200 clips y una duración mayor que 0 y hasta 7200 segundos.")
    if args.kind == "negative-continuous" and args.count != 1:
        raise ValueError("Una captura negativa continua debe ser un solo WAV (count=1).")
    output = Path(args.manifest).resolve()
    if not output.is_relative_to((BASE_DIR / "data" / "wakeword").resolve()) or output.exists():
        raise ValueError("El manifiesto debe ser nuevo y quedar bajo data/wakeword/.")
    batch = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = BASE_DIR / "recordings" / "wakeword_validacion" / args.session / batch
    folder.mkdir(parents=True, exist_ok=False)
    stream = AudioStream(device=args.device)
    # start() admite fallback: aquí se abre exclusivamente el índice que eligió
    # el operador para impedir etiquetar un micrófono USB como interno.
    if not stream._abrir(args.device):
        raise ValueError("No se pudo abrir el dispositivo exacto; no se probó otro.")
    print(f"Micrófono: [{args.device}] {name}; etiqueta: {args.microphone}; sesión: {args.session}")
    print("Audio local en WAV PCM mono 16 kHz/16 bit. Las etiquetas quedan sin revisar.")
    entries = []
    interrupted = False
    try:
        for index in range(args.count):
            instruction = ('Decí "oye niri" después de empezar.' if args.kind == "positive" else
                           'Dejá audio ambiente continuo sin decir "oye niri"; no se corta por detecciones.')
            input(f"[{index + 1}/{args.count}] {instruction} Enter para grabar {args.seconds:g} s: ")
            # Comprobar otra vez antes de cada clip evita seguir si el servicio
            # fue iniciado mientras el operador respondía al aviso.
            check_service_stopped()
            stream.reset_buffers()
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = folder / f"{args.kind}_{stamp}.wav"
            try:
                duration, pcm_digest = capture(stream, path, args.seconds)
            except (KeyboardInterrupt, OSError, RuntimeError, ValueError):
                interrupted = True
                if path.exists():
                    # wave cierra y corrige la cabecera incluso al interrumpir.
                    with wave.open(str(path), "rb") as handle:
                        content = handle.readframes(handle.getnframes())
                    duration = len(content) / (2 * SAMPLE_RATE)
                    pcm_digest = hashlib.sha256(content).hexdigest()
                else:
                    raise
            entries.append({
                "path": str(path.relative_to(BASE_DIR)), "sha256": sha256(path),
                "pcm_sha256": pcm_digest, "duration_seconds": duration,
                "sample_rate": SAMPLE_RATE, "channels": 1, "sample_width": 2,
                "label": int(args.kind == "positive"), "label_reviewed": False,
                "source": "validation_positive" if args.kind == "positive" else "continuous_candidate",
                "session": args.session, "session_reviewed": True,
                "device": args.microphone, "capture_device_name": name,
                "continuous_negative": False,
                "usable": duration > 0 and not interrupted,
                "capture_complete": not interrupted,
                "exclusion_reason": "Captura interrumpida; revisar antes de usar." if interrupted else "",
            })
            print(f"Guardado {path.relative_to(BASE_DIR)} ({duration:.2f} s).")
            if interrupted:
                break
    except (KeyboardInterrupt, EOFError):
        interrupted = True
    finally:
        try:
            stream.stop()
        finally:
            if entries:
                json_write(output, {"format_version": FORMAT_VERSION, "entries": entries,
                                   "notes": "Escuchar localmente; luego revisar label/label_reviewed/usable. "
                                            "Solo si todo un negativo continuo excluye la wake word, "
                                            "marcar también continuous_negative=true. No editar hashes/WAV."})
    if not entries:
        raise ValueError("No se capturaron clips; no se creó manifiesto.")
    print(json.dumps({"manifest": str(output), "clips": len(entries),
                      "label_reviewed": False, "interrupted": interrupted}, ensure_ascii=False))
    return 1 if interrupted else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True, help="Índice físico de sounddevice")
    parser.add_argument("--microphone", choices=("internal", "usb"), required=True)
    parser.add_argument("--session", required=True, help="Mismo ID para clips de la misma sesión")
    parser.add_argument("--kind", choices=("positive", "negative-continuous"), required=True)
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"No se completó: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

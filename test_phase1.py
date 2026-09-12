"""Medición de captura y wake word en reposo, sin STT ni ejecución de acciones.

No guarda audio. Detener niri.service antes de usar el micrófono.
Uso: .venv/bin/python test_phase1.py --seconds 15
"""
import argparse
import subprocess
import sys
import time


def test_idle_cpu(seconds=15.0, device=None):
    try:
        status = subprocess.run(["systemctl", "--user", "is-active", "niri.service"],
                                capture_output=True, text=True, timeout=5)
        if status.returncode != 3 or status.stdout.strip() not in {"inactive", "failed"}:
            print("No se confirmó que niri.service esté detenido; no abro el micrófono.", file=sys.stderr)
            return 2
    except (OSError, subprocess.TimeoutExpired):
        print("No pude consultar el servicio; no abro el micrófono.", file=sys.stderr)
        return 2

    # PortAudio puede abrir conexiones al importar: hacerlo después de comprobar
    # el servicio y únicamente en esta prueba explícita de hardware.
    from src.audio import AudioStream
    from src.wake_word import WakeWordDetector
    audio = AudioStream(device=device)
    try:
        detector = WakeWordDetector()
        audio.start(esperar=False)
        start, cpu_start = time.monotonic(), time.process_time()
        chunks, activations = 0, 0
        while time.monotonic() - start < seconds:
            activations += bool(detector.feed(audio.read_chunk(1280)))
            chunks += 1
        elapsed = time.monotonic() - start
        cpu = 100 * (time.process_time() - cpu_start) / elapsed
        print(f"Captura + wake word: {elapsed:.1f} s, {chunks} bloques, CPU {cpu:.1f}% de un núcleo.")
        print(f"Activaciones sin verificar: {activations}. No se transcribió ni ejecutó ninguna orden.")
        return 0
    except Exception as exc:
        print(f"Prueba de audio incompleta: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        audio.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--device", help="nombre del dispositivo PortAudio; omitir usa la selección del asistente")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300:
        parser.error("--seconds debe estar entre 1 y 300")
    raise SystemExit(test_idle_cpu(args.seconds, args.device))

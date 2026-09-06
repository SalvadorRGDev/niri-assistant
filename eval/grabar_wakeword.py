"""
Graba muestras de "oye niri" con el micrófono que haya conectado.

Para qué: hoy no existe un solo positivo grabado con el micrófono interno. El
wake word se entrenó con voces sintéticas de Piper y negativos del micrófono
USB, y con el interno dispara sobre ruido ambiente con puntajes de hasta 0.986
— más altos que muchos aciertos reales. Sin positivos de ESTE micrófono no se
puede saber si el problema se arregla subiendo el umbral o si hay que
reentrenar: tocar el umbral a ojo es adivinar.

Los negativos ya se juntan solos en `recordings/falsos_positivos/` (los 2 s
previos a cada disparo que no produjo ninguna orden). Esto junta el otro lado.

Uso:
    systemctl --user stop niri.service          # el micrófono es de acceso exclusivo
    .venv/bin/python eval/grabar_wakeword.py    # o: --cantidad 40
    systemctl --user start niri.service

Las grabaciones van a recordings/wakeword_positivos/, que no se versiona.
"""
import argparse
import datetime
import subprocess
import sys
import time
import wave
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np  # noqa: E402

from src.audio import AudioStream, _descripcion  # noqa: E402
from src.config import RECORDINGS_DIR, SAMPLE_RATE, CHANNELS  # noqa: E402

DESTINO = RECORDINGS_DIR / "wakeword_positivos"
SEGUNDOS_POR_MUESTRA = 2.0
CHUNK = 1280  # 80 ms, el mismo tamaño que consume el wake word en producción

VERDE, ROJO, AMARILLO, GRIS, FIN = "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[0m"

# Variaciones pedidas a lo largo de la sesión. Un modelo afinado con muestras
# todas iguales (misma distancia, mismo tono) funciona solo en esa situación;
# el uso real tiene al usuario lejos, de espaldas o hablando bajo.
VARIACIONES = [
    "como le hablarías normalmente",
    "un poco más rápido",
    "un poco más lento y marcando las sílabas",
    "más bajo, casi susurrando",
    "desde más lejos, o girando la cabeza",
]


def servicio_activo() -> bool:
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", "niri.service"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() == "active"
    except Exception:
        return False


def nivel_dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    return 20 * np.log10(max(rms, 1e-9) / 32768)


def grabar_una(stream: AudioStream, segundos: float) -> np.ndarray:
    trozos, objetivo = [], int(segundos * SAMPLE_RATE / CHUNK)
    for _ in range(objetivo):
        trozos.append(stream.read_chunk(CHUNK))
    return np.concatenate(trozos)


def main() -> int:
    parser = argparse.ArgumentParser(description="Graba positivos del wake word")
    parser.add_argument("--cantidad", type=int, default=30,
                        help="cuántas muestras grabar (por defecto 30)")
    args = parser.parse_args()

    if servicio_activo():
        print(f"{ROJO}niri.service está corriendo y tiene el micrófono tomado.{FIN}")
        print("El micrófono es de acceso exclusivo, así que hay que pararlo antes:\n")
        print("  systemctl --user stop niri.service")
        print("  .venv/bin/python eval/grabar_wakeword.py")
        print("  systemctl --user start niri.service")
        return 1

    DESTINO.mkdir(parents=True, exist_ok=True)
    ya_habia = len(list(DESTINO.glob("*.wav")))

    stream = AudioStream()
    try:
        stream.start(esperar=False)
    except Exception as e:
        print(f"{ROJO}No pude abrir el micrófono: {e}{FIN}")
        return 1

    print(f"Entrada: {_descripcion(stream.device)}")
    print(f"Voy a grabar {args.cantidad} muestras de {SEGUNDOS_POR_MUESTRA:.0f} segundos.")
    print(f"En cada una decí {VERDE}\"oye niri\"{FIN} cuando aparezca GRABANDO, y nada más.")
    if ya_habia:
        print(f"{GRIS}(ya hay {ya_habia} muestras de antes; estas se suman){FIN}")
    print(f"\n{GRIS}Ctrl+C para cortar; lo grabado hasta ahí se conserva.{FIN}")
    input("\nEnter para empezar... ")

    niveles, guardadas = [], 0
    try:
        for i in range(1, args.cantidad + 1):
            variacion = VARIACIONES[(i - 1) * len(VARIACIONES) // args.cantidad]
            print(f"\n[{i}/{args.cantidad}] {variacion}")
            for cuenta in (3, 2, 1):
                print(f"  {cuenta}...", end="\r", flush=True)
                time.sleep(0.7)
            # Descartar lo que se acumuló durante la cuenta regresiva, que es
            # ruido de fondo y no la muestra.
            stream.reset_buffers()
            print(f"  {VERDE}GRABANDO{FIN}          ", end="\r", flush=True)

            audio = grabar_una(stream, SEGUNDOS_POR_MUESTRA)
            nivel = nivel_dbfs(audio)
            niveles.append(nivel)

            sello = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            destino = DESTINO / f"oye_niri_{sello}.wav"
            with wave.open(str(destino), "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio.tobytes())
            guardadas += 1

            if nivel < -45:
                aviso = f"  {AMARILLO}muy bajo ({nivel:.0f} dBFS) — ¿dijiste algo?{FIN}"
            elif nivel > -12:
                aviso = f"  {AMARILLO}muy fuerte ({nivel:.0f} dBFS) — puede estar saturando{FIN}"
            else:
                aviso = f"  {VERDE}ok{FIN} ({nivel:.0f} dBFS)"
            print(f"  guardada        {aviso}")
    except KeyboardInterrupt:
        print(f"\n\n{AMARILLO}Cortado. Se conservan las {guardadas} muestras grabadas.{FIN}")
    finally:
        stream.stop()

    if guardadas:
        print(f"\n{guardadas} muestras en {DESTINO.relative_to(BASE_DIR)}/ "
              f"(nivel mediano {sorted(niveles)[len(niveles)//2]:.0f} dBFS)")
        negativos = len(list((RECORDINGS_DIR / 'falsos_positivos').glob('*.wav')))
        print(f"Negativos disponibles para comparar: {negativos}")
        print(f"\n{GRIS}Ahora: systemctl --user start niri.service{FIN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

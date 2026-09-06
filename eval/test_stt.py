"""
Banco de regresión del STT sobre las grabaciones reales del usuario.

Existe por un error concreto de este proyecto: se midió `beam_size=1` sobre 8
grabaciones, dio "8/8 idéntico y 14% más rápido" y se lo dio por bueno. Con las
54 el resultado era 36/54 y encima más lento. Sin un banco fijo, cada medición
del STT se hace sobre un conjunto distinto —`recordings/` crece solo, y hoy
crece sobre todo con falsos positivos del wake word— y no se puede comparar
nada con nada.

**No guarda transcripciones, solo hashes.** El contenido es la voz real del
usuario (§3.1.4): un hash alcanza para detectar que algo cambió, sin dejar el
texto en un archivo que después alguien lea, copie o publique. Por la misma
razón este script nunca imprime lo que se dijo.

Sobre el piso de ruido: Whisper no es determinista. La MISMA configuración
corrida dos veces sobre las mismas 54 grabaciones difiere en una. Cualquier
cambio por debajo de ese piso no significa nada, y por eso el umbral de fallo
no es cero.

Uso:
    .venv/bin/python eval/test_stt.py --crear-baseline   # fija el conjunto y sus hashes
    .venv/bin/python eval/test_stt.py                    # compara contra el baseline
"""
import argparse
import glob
import hashlib
import json
import statistics
import sys
import time
import wave
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np  # noqa: E402

from src.stt import SpeechToText  # noqa: E402

BASELINE = BASE_DIR / "eval" / "resultados" / "audio_baseline.json"
RECORDINGS = BASE_DIR / "recordings"

# Diferencias toleradas antes de declarar regresión, en fracción del total.
# El piso medido es 1/54 (1.9%); 5% deja margen para dos o tres sin alarmar.
TOLERANCIA = 0.05

VERDE, ROJO, AMARILLO, GRIS, FIN = "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[0m"


def cargar_audio(ruta: Path) -> np.ndarray:
    with wave.open(str(ruta)) as w:
        crudo = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return crudo.astype(np.float32) / 32768.0


def transcribir(archivos: list[Path]) -> tuple[dict, list[float]]:
    """Devuelve {nombre: hash del texto} y los tiempos. Nunca retorna el texto."""
    stt = SpeechToText()
    hashes, tiempos = {}, []
    for ruta in archivos:
        inicio = time.perf_counter()
        texto = stt.transcribe(cargar_audio(ruta))
        tiempos.append(time.perf_counter() - inicio)
        hashes[ruta.name] = hashlib.sha256(texto.strip().lower().encode()).hexdigest()[:16]
    return hashes, tiempos


def duracion_total(archivos: list[Path]) -> float:
    total = 0.0
    for ruta in archivos:
        with wave.open(str(ruta)) as w:
            total += w.getnframes() / w.getframerate()
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Regresión del STT sobre grabaciones reales")
    parser.add_argument("--crear-baseline", action="store_true",
                        help="fija el conjunto actual de grabaciones y sus hashes")
    args = parser.parse_args()

    if args.crear_baseline:
        archivos = sorted(RECORDINGS.glob("*.wav"))
        if not archivos:
            print(f"No hay grabaciones en {RECORDINGS}.", file=sys.stderr)
            return 2
        print(f"Transcribiendo {len(archivos)} grabaciones para fijar el baseline...")
        hashes, tiempos = transcribir(archivos)
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps({
            "creado": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "audio_s": round(duracion_total(archivos), 1),
            "mediana_ms": round(statistics.median(tiempos) * 1000, 1),
            "total_s": round(sum(tiempos), 2),
            "hashes": hashes,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Baseline fijado: {len(hashes)} grabaciones, "
              f"{round(duracion_total(archivos),1)}s de audio, "
              f"mediana {statistics.median(tiempos)*1000:.0f} ms.")
        print(f"{GRIS}{BASELINE.relative_to(BASE_DIR)} (no se versiona: son datos del usuario){FIN}")
        return 0

    if not BASELINE.exists():
        print("No hay baseline todavía. Crealo con --crear-baseline.")
        return 0

    referencia = json.loads(BASELINE.read_text(encoding="utf-8"))
    archivos = [RECORDINGS / n for n in referencia["hashes"]]
    faltantes = [a.name for a in archivos if not a.exists()]
    archivos = [a for a in archivos if a.exists()]
    if not archivos:
        print("Ninguna grabación del baseline sigue en disco.", file=sys.stderr)
        return 2

    print(f"Regresión del STT — {len(archivos)} grabaciones "
          f"({referencia['audio_s']}s de audio)")
    if faltantes:
        print(f"  {AMARILLO}{len(faltantes)} grabaciones del baseline ya no están{FIN}")

    hashes, tiempos = transcribir(archivos)
    distintas = [n for n, h in hashes.items() if referencia["hashes"].get(n) != h]
    mediana = statistics.median(tiempos) * 1000
    delta = mediana - referencia["mediana_ms"]

    print(f"\n  transcripciones idénticas  {len(archivos) - len(distintas)}/{len(archivos)}")
    print(f"  latencia mediana           {mediana:.0f} ms "
          f"({'+' if delta >= 0 else ''}{delta:.0f} ms vs baseline)")
    print(f"  {GRIS}piso de ruido conocido: la misma config difiere en ~1 de 54{FIN}")

    fraccion = len(distintas) / len(archivos)
    if fraccion > TOLERANCIA:
        print(f"\n  {ROJO}REGRESIÓN: cambió el {fraccion:.0%} de las transcripciones, "
              f"por encima del {TOLERANCIA:.0%} tolerado.{FIN}")
        print(f"  {GRIS}Archivos afectados (sin su contenido): "
              f"{', '.join(sorted(distintas)[:5])}{'...' if len(distintas) > 5 else ''}{FIN}")
        return 1

    print(f"\n  {VERDE}sin regresión{FIN} ({len(distintas)} diferencias, dentro del ruido)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

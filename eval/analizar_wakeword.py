"""
Compara cómo puntúa el wake word los positivos reales contra los falsos disparos.

Responde la única pregunta que importa sobre los falsos positivos: ¿existe algún
par (umbral, frames consecutivos) que separe un "oye niri" de verdad del ruido
ambiente de este micrófono?

- Si existe, el arreglo son dos valores en `src/config.py` y se termina acá.
- Si no existe, queda demostrado que hay que reentrenar el modelo, que es caro
  (`CLAUDE.md` dice que cuesta una sesión) pero al menos ya no es una corazonada.

Entradas:
    recordings/wakeword_positivos/   lo que grabaste con eval/grabar_wakeword.py
    recordings/falsos_positivos/     los 2 s previos a cada disparo sin orden,
                                     que el asistente junta solo

No imprime ni reproduce audio: solo puntajes.

Uso:  .venv/bin/python eval/analizar_wakeword.py
"""
import glob
import statistics
import sys
import wave
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np  # noqa: E402

from eval.calidad_audio import tiene_voz  # noqa: E402
from src.wake_word import WakeWordDetector  # noqa: E402
from src.config import (RECORDINGS_DIR, SAMPLE_RATE, WAKE_WORD_THRESHOLD,  # noqa: E402
                        WAKE_WORD_TRIGGER_LEVEL)

POSITIVOS = RECORDINGS_DIR / "wakeword_positivos"
NEGATIVOS = RECORDINGS_DIR / "falsos_positivos"
CHUNK = 1280

# Silencio que se agrega antes y después de cada grabación. El clasificador
# necesita 16 embeddings (1.28 s) más el contexto del extractor de rasgos, así
# que sobre un archivo de 2 s la última ventana que alcanza a formarse termina
# donde termina el archivo: si el "oye niri" quedó pegado al final —y queda,
# porque entre el aviso de GRABANDO y la voz hay reacción humana—, el modelo
# nunca ve la palabra completa. Sin este relleno, muestras buenas puntuaban
# 0.002 y con él 0.962. En producción el stream es continuo y ese corte no
# existe: el relleno reproduce esa condición, no la maquilla.
RELLENO = np.zeros(SAMPLE_RATE, dtype=np.int16)

VERDE, ROJO, AMARILLO, GRIS, FIN = "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[0m"


def puntajes_de(detector: WakeWordDetector, ruta: Path) -> list[float]:
    """Todos los puntajes por frame de un archivo, como los vería en streaming."""
    with wave.open(str(ruta)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = np.concatenate([RELLENO, audio, RELLENO])

    detector.oww_features.reset()
    detector.oww_model.reset()
    puntajes = []
    for i in range(0, len(audio) - CHUNK + 1, CHUNK):
        for rasgos in detector.oww_features.process_streaming(audio[i:i + CHUNK].tobytes()):
            for p in detector.oww_model.process_streaming(rasgos):
                puntajes.append(float(p))
    return puntajes


def racha_maxima(puntajes: list[float], umbral: float) -> int:
    """Frames consecutivos por encima del umbral: es lo que exige WAKE_WORD_TRIGGER_LEVEL."""
    mejor = actual = 0
    for p in puntajes:
        actual = actual + 1 if p >= umbral else 0
        mejor = max(mejor, actual)
    return mejor


def resumen(nombre: str, archivos: list[Path], detector) -> list[list[float]]:
    todos = [puntajes_de(detector, a) for a in archivos]
    picos = [max(p) if p else 0.0 for p in todos]
    print(f"\n{nombre} — {len(archivos)} grabaciones")
    if picos:
        print(f"  puntaje pico: mediana {statistics.median(picos):.3f} · "
              f"mín {min(picos):.3f} · máx {max(picos):.3f}")
    return todos


def con_voz(archivos: list[Path]) -> tuple[list[Path], list[tuple[Path, str]]]:
    """
    Separa los positivos utilizables de los que no contienen voz.

    Un positivo mudo (un golpe a la laptop, un roce del micrófono) puntúa bajo
    igual que un "oye niri" que el modelo no reconoció, y la grilla no puede
    distinguirlos: uno solo alcanza para concluir "hay que reentrenar" cuando
    en realidad la muestra estaba mal. Las dos primeras muestras grabadas en
    este proyecto eran exactamente eso.
    """
    utiles, descartados = [], []
    for a in archivos:
        with wave.open(str(a)) as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        ok, motivo = tiene_voz(audio)
        (utiles if ok else descartados).append(a if ok else (a, motivo))
    return utiles, descartados


def main() -> int:
    pos_archivos = sorted(POSITIVOS.glob("*.wav"))
    neg_archivos = sorted(NEGATIVOS.glob("*.wav"))
    if not pos_archivos:
        print(f"No hay positivos en {POSITIVOS.relative_to(BASE_DIR)}/.\n"
              "Grabalos primero con: .venv/bin/python eval/grabar_wakeword.py")
        return 2
    if not neg_archivos:
        print(f"No hay negativos en {NEGATIVOS.relative_to(BASE_DIR)}/.\n"
              "Se juntan solos con el asistente andando: son los 2 s previos a "
              "cada disparo que no produjo ninguna orden.")
        return 2

    pos_archivos, sin_voz = con_voz(pos_archivos)
    if sin_voz:
        print(f"{AMARILLO}Descarto {len(sin_voz)} positivo(s) que no contienen voz:{FIN}")
        for a, motivo in sin_voz:
            print(f"  {GRIS}{a.name}: {motivo}{FIN}")
        print(f"{GRIS}Los archivos quedan donde están; simplemente no entran al "
              f"análisis.{FIN}")
    if not pos_archivos:
        print(f"\n{ROJO}No queda ningún positivo utilizable.{FIN} Volvé a grabarlos "
              "con: .venv/bin/python eval/grabar_wakeword.py")
        return 2

    detector = WakeWordDetector()
    print(f"\nConfiguración actual: umbral {WAKE_WORD_THRESHOLD}, "
          f"{WAKE_WORD_TRIGGER_LEVEL} frames consecutivos")

    pos = resumen("POSITIVOS (vos diciendo 'oye niri')", pos_archivos, detector)
    neg = resumen("NEGATIVOS (ruido que disparó el wake word)", neg_archivos, detector)

    # AUC: probabilidad de que un positivo puntúe más alto que un negativo tomados
    # al azar. Responde "¿alcanza con mover el umbral?" sin depender de la grilla:
    # 1.0 = separables con algún umbral, 0.5 = azar, <0.5 = el modelo prefiere el
    # ruido, y ahí ningún umbral puede arreglarlo porque el orden ya está mal.
    picos_pos = [max(p) if p else 0.0 for p in pos]
    picos_neg = [max(n) if n else 0.0 for n in neg]
    auc = statistics.fmean(
        1.0 if a > b else 0.5 if a == b else 0.0
        for a in picos_pos for b in picos_neg)
    color = VERDE if auc >= 0.9 else AMARILLO if auc > 0.5 else ROJO
    print(f"\nOrdenamiento — AUC {color}{auc:.3f}{FIN} "
          f"{GRIS}(1.0 = separables con algún umbral · 0.5 = azar · "
          f"<0.5 = puntúa más alto el ruido){FIN}")

    # Grilla: para cada par (umbral, racha), cuántos positivos se detectan y
    # cuántos negativos se cuelan. El objetivo es 100% de positivos con 0 falsos.
    print(f"\n{'umbral':>7} {'frames':>7} {'detecta':>9} {'falsos':>8}   veredicto")
    print("  " + "-" * 48)
    mejores = []
    for umbral in (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99):
        for racha in (2, 3, 4, 5, 6, 8):
            detectados = sum(1 for p in pos if racha_maxima(p, umbral) >= racha)
            colados = sum(1 for n in neg if racha_maxima(n, umbral) >= racha)
            mejores.append((detectados / len(pos), -colados / len(neg), umbral, racha,
                            detectados, colados))
    mejores.sort(reverse=True)

    perfectos = [m for m in mejores if m[0] == 1.0 and m[4] and m[5] == 0]
    for tasa, _, umbral, racha, det, col in mejores[:8]:
        marca = f"{VERDE}separa limpio{FIN}" if (tasa == 1.0 and col == 0) else (
            f"{AMARILLO}parcial{FIN}" if col < len(neg) else f"{ROJO}no sirve{FIN}")
        print(f"  {umbral:>5.2f} {racha:>7} {det:>4}/{len(pos)} {col:>6}/{len(neg)}   {marca}")

    print()
    if perfectos:
        _, _, umbral, racha, det, col = perfectos[0]
        print(f"{VERDE}Hay separación limpia.{FIN} Con umbral {umbral} y {racha} frames "
              f"consecutivos se detectan los {det} positivos y no se cuela ninguno "
              f"de los {len(neg)} negativos.")
        print(f"\nEn src/config.py (requiere confirmación, §3.2):")
        print(f"  WAKE_WORD_THRESHOLD={umbral}")
        print(f"  WAKE_WORD_TRIGGER_LEVEL={racha}")
    else:
        mejor = mejores[0]
        print(f"{ROJO}No hay ningún par que separe.{FIN} Lo mejor de la grilla detecta "
              f"{mejor[4]}/{len(pos)} positivos dejando pasar {mejor[5]}/{len(neg)} falsos "
              f"(umbral {mejor[2]}, {mejor[3]} frames).")
        print("Queda demostrado que ajustar los umbrales no alcanza: hay que reentrenar "
              "el wake word con negativos de este micrófono, que son justo los que se "
              "están juntando en recordings/falsos_positivos/.")
        if auc < 0.5:
            print(f"Con AUC {auc:.3f} no es que el umbral esté mal puesto: el modelo "
                  "ordena el ruido por encima de la voz real, y ningún umbral "
                  "reordena una lista.")
    print(f"\n{GRIS}Muestras chicas engañan: con menos de ~25 positivos, tomar esto como "
          f"definitivo es repetir el error del beam_size (ver README.md).{FIN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Mide cuánto ensuciaba el alias del remuestreo en ESTE micrófono.

Hasta el 2026-09-08 `src/audio.py` bajaba de 48 kHz a 16 kHz quedándose con una
de cada tres muestras, sin filtrar antes. Todo lo que el micrófono capta por
encima de 8 kHz no se perdía: se plegaba sobre la banda de voz. El filtro ya
está puesto, pero **cuánto** ensuciaba nunca se midió, y sin ese número no se
sabe si era un detalle o media explicación de los falsos disparos.

No se puede medir desde las grabaciones guardadas: todas están en 16 kHz, o sea
ya decimadas, y un alias no se deshace. Hace falta capturar a 48 kHz sin decimar
y comparar los dos caminos sobre el mismo audio, que es lo que hace esto.

Graba dos escenas y las analiza:
  ambiente — sin hablar; es la condición en la que aparecen los falsos disparos
  voz      — diciendo "oye niri"; es la condición que tiene que funcionar

De cada una informa la energía por encima de 8 kHz, dónde cae al plegarse, la
relación alias/señal en dB, y los puntajes del wake word por los dos caminos.

Las capturas quedan en recordings/captura_48k/ (no se versiona) y se pueden
reanalizar sin micrófono con --solo-analizar.

Uso:
    systemctl --user stop niri.service; .venv/bin/python eval/medir_alias.py; systemctl --user start niri.service
    .venv/bin/python eval/medir_alias.py --solo-analizar
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
from scipy.signal import lfilter  # noqa: E402

from src.audio import AudioStream, FACTOR_DECIMACION, FIR_ANTIALIAS, _descripcion  # noqa: E402
from src.config import CAPTURE_SAMPLE_RATE, CHANNELS, RECORDINGS_DIR, SAMPLE_RATE  # noqa: E402
from src.wake_word import WakeWordDetector  # noqa: E402

DESTINO = RECORDINGS_DIR / "captura_48k"
CHUNK = 4800  # 100 ms a 48 kHz
NYQUIST = SAMPLE_RATE / 2  # 8 kHz: por encima de acá, todo se pliega

VERDE, ROJO, AMARILLO, GRIS, FIN = "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[0m"

ESCENAS = (
    ("ambiente", "No digas nada. Dejá el ambiente como está cuando el asistente "
                 "dispara solo."),
    ("voz", 'Decí "oye niri" cada dos o tres segundos, como le hablarías normalmente.'),
)


class CapturaCruda(AudioStream):
    """
    AudioStream que entrega el audio tal como llega, a CAPTURE_SAMPLE_RATE.

    Hereda toda la elección de dispositivo (que no es trivial: el micrófono USB
    no siempre está, ver src/audio.py) y solo cambia el callback, porque el de
    producción justamente filtra y deciman, que es lo que acá hay que evitar.
    """

    def callback(self, indata, frames, time, status):
        if status:
            print(f"  {AMARILLO}estado del stream: {status}{FIN}")
        self.q.put(indata.copy()[:, 0])


def servicio_activo() -> bool:
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", "niri.service"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() == "active"
    except Exception:
        return False


def nivel_dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return 20 * np.log10(max(rms, 1e-9) / 32768)


def separar(crudo48: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Parte la captura en (lo que sobrevive, lo que se pliega), ya a 16 kHz.

    El FIR es de fase lineal, así que retrasa la señal exactamente la mitad de
    sus coeficientes. Sin compensar ese retraso, restar las dos versiones daría
    la diferencia entre dos señales desalineadas y no el alias.
    """
    filtrado = lfilter(FIR_ANTIALIAS, 1.0, crudo48.astype(np.float64))
    retardo = (len(FIR_ANTIALIAS) - 1) // 2
    limpio = filtrado[retardo:]
    crudo = crudo48[:len(limpio)].astype(np.float64)
    alto = crudo - limpio  # lo que está por encima del corte, que es lo que se pliega
    return limpio[::FACTOR_DECIMACION], alto[::FACTOR_DECIMACION]


def reparto_espectral(crudo48: np.ndarray) -> dict:
    """Dónde está la energía de la captura, y adónde cae la que se pliega."""
    ventana = np.hanning(len(crudo48))
    espectro = np.abs(np.fft.rfft(crudo48.astype(np.float64) * ventana)) ** 2
    freqs = np.fft.rfftfreq(len(crudo48), 1 / CAPTURE_SAMPLE_RATE)
    total = espectro.sum()
    if total <= 0:
        return {}

    arriba = freqs >= NYQUIST
    # Al decimar, cada frecuencia f de arriba reaparece en esta otra. Es el
    # plegado del muestreo: 10 kHz vuelve como 6 kHz, 17 kHz como 1 kHz.
    plegada = np.abs(((freqs + NYQUIST) % SAMPLE_RATE) - NYQUIST)
    en_voz = arriba & (plegada >= 300) & (plegada <= 3400)
    pct_arriba = 100 * espectro[arriba].sum() / total
    # El pico solo se informa si hay algo real arriba: si no, el argmax elige
    # ruido numérico y señalaría una frecuencia inventada.
    pico = (float(freqs[arriba][np.argmax(espectro[arriba])])
            if arriba.any() and pct_arriba > 0.01 else None)
    return {
        "pct_arriba": pct_arriba,
        "pct_cae_en_voz": 100 * espectro[en_voz].sum() / total,
        "pico_arriba_hz": pico,
    }


def puntajes_wake_word(detector: WakeWordDetector, audio16: np.ndarray) -> tuple[float, int]:
    """(puntaje pico, frames seguidos por encima de 0.9) de una señal de 16 kHz."""
    detector.oww_features.reset()
    detector.oww_model.reset()
    puntajes = []
    muestras = np.clip(np.rint(audio16), -32768, 32767).astype(np.int16)
    for i in range(0, len(muestras) - 1280 + 1, 1280):
        for rasgos in detector.oww_features.process_streaming(muestras[i:i + 1280].tobytes()):
            for p in detector.oww_model.process_streaming(rasgos):
                puntajes.append(float(p))
    if not puntajes:
        return 0.0, 0
    mejor = actual = 0
    for p in puntajes:
        actual = actual + 1 if p >= 0.9 else 0
        mejor = max(mejor, actual)
    return max(puntajes), mejor


def analizar(ruta: Path, detector: WakeWordDetector) -> None:
    with wave.open(str(ruta)) as w:
        if w.getframerate() != CAPTURE_SAMPLE_RATE:
            print(f"  {AMARILLO}{ruta.name}: está a {w.getframerate()} Hz, no a "
                  f"{CAPTURE_SAMPLE_RATE}. Me la salteo.{FIN}")
            return
        crudo48 = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    limpio, alias = separar(crudo48)
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    señal, ruido = rms(limpio), rms(alias)
    relacion = 20 * np.log10(max(ruido, 1e-9) / max(señal, 1e-9))
    reparto = reparto_espectral(crudo48)

    print(f"\n{ruta.name}  {GRIS}({len(crudo48)/CAPTURE_SAMPLE_RATE:.0f} s, "
          f"{nivel_dbfs(crudo48):.1f} dBFS){FIN}")
    pico = reparto.get("pico_arriba_hz")
    print(f"  energía por encima de {NYQUIST/1000:.0f} kHz: "
          f"{reparto.get('pct_arriba', 0):.2f}% del total"
          + (f" {GRIS}(pico en {pico/1000:.1f} kHz){FIN}" if pico else ""))
    print(f"  la parte que se pliega sobre la voz (300-3400 Hz): "
          f"{reparto.get('pct_cae_en_voz', 0):.2f}% del total")
    color = ROJO if relacion > -20 else AMARILLO if relacion > -35 else VERDE
    print(f"  alias / señal en la salida de 16 kHz: {color}{relacion:+.1f} dB{FIN}")

    # El número que decide: si el alias movía el wake word, se ve acá.
    pico_v, racha_v = puntajes_wake_word(detector, limpio + alias)  # camino viejo
    pico_n, racha_n = puntajes_wake_word(detector, limpio)          # camino nuevo
    print(f"  wake word — antes (con alias): pico {pico_v:.3f}, {racha_v} frames ≥0.9")
    print(f"              ahora (filtrado):  pico {pico_n:.3f}, {racha_n} frames ≥0.9")


def grabar(segundos: float) -> int:
    if servicio_activo():
        print(f"{ROJO}niri.service está corriendo y tiene el micrófono tomado.{FIN}")
        print("El micrófono es de acceso exclusivo, así que hay que pararlo antes.")
        # Una sola línea: entre un `stop` suelto y la grabación hay una ventana
        # en la que el servicio puede volver a arrancar y tomar el micrófono. El
        # `;` final lo levanta aunque esto falle o lo cortes. Vale en fish y bash.
        print(f"\n  {VERDE}systemctl --user stop niri.service; "
              f".venv/bin/python eval/medir_alias.py; "
              f"systemctl --user start niri.service{FIN}\n")
        return 1

    DESTINO.mkdir(parents=True, exist_ok=True)
    stream = CapturaCruda()
    try:
        stream.start(esperar=False)
    except Exception as e:
        print(f"{ROJO}No pude abrir el micrófono: {e}{FIN}")
        return 1

    print(f"Entrada: {_descripcion(stream.device)} "
          f"{GRIS}(captura cruda a {CAPTURE_SAMPLE_RATE} Hz, sin decimar){FIN}")
    print(f"Dos escenas de {segundos:.0f} segundos cada una.\n")

    try:
        for nombre, instruccion in ESCENAS:
            print(f"{VERDE}{nombre}{FIN} — {instruccion}")
            input("  Enter cuando estés listo... ")
            for cuenta in (3, 2, 1):
                print(f"  {cuenta}...", end="\r", flush=True)
                time.sleep(0.7)
            stream.reset_buffers()
            print(f"  {VERDE}GRABANDO{FIN}          ", end="\r", flush=True)

            trozos = [stream.read_chunk(CHUNK)
                      for _ in range(int(segundos * CAPTURE_SAMPLE_RATE / CHUNK))]
            audio = np.concatenate(trozos)

            sello = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            destino = DESTINO / f"{nombre}_{sello}.wav"
            with wave.open(str(destino), "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(CAPTURE_SAMPLE_RATE)
                wf.writeframes(audio.tobytes())
            print(f"  guardada        {GRIS}{destino.name} "
                  f"({nivel_dbfs(audio):.0f} dBFS){FIN}")
    except KeyboardInterrupt:
        print(f"\n{AMARILLO}Cortado. Se conserva lo que ya se grabó.{FIN}")
        return 1
    finally:
        stream.stop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Mide el alias del remuestreo a 16 kHz")
    parser.add_argument("--segundos", type=float, default=10.0,
                        help="duración de cada escena (por defecto 10)")
    parser.add_argument("--solo-analizar", action="store_true",
                        help="no graba: analiza las capturas que ya estén guardadas")
    args = parser.parse_args()

    if FIR_ANTIALIAS is None:
        print(f"{AMARILLO}Captura y salida están a la misma frecuencia: no hay "
              f"decimación y no hay alias que medir.{FIN}")
        return 0

    if not args.solo_analizar:
        codigo = grabar(args.segundos)
        if codigo:
            return codigo

    capturas = sorted(DESTINO.glob("*.wav"))
    if not capturas:
        print(f"\nNo hay capturas en {DESTINO.relative_to(BASE_DIR)}/.")
        return 2

    print(f"\n{'=' * 60}\nAnálisis de {len(capturas)} captura(s)")
    detector = WakeWordDetector()
    for ruta in capturas:
        analizar(ruta, detector)

    print(f"\n{GRIS}Cómo leerlo: la relación alias/señal es cuánta basura plegada "
          f"había en la salida de 16 kHz\nrespecto de la señal real. Por debajo de "
          f"-35 dB era irrelevante; por encima de -20 dB era\nmedia explicación de "
          f"los falsos disparos. Los puntajes del wake word dicen si esa\nbasura "
          f"movía la decisión o no.{FIN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

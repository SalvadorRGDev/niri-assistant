"""
Pruebas de la selección de dispositivo de entrada.

Ninguna abre el micrófono de verdad: se reemplazan `_abrir` y la enumeración por
dobles, así que corren con niri.service levantado y sin robarle el dispositivo.

Existen por un incidente real: al desconectar el micrófono USB, el `default` del
sistema dejó de resolver a una fuente de audio, `AudioStream.start()` murió y
systemd entró en un bucle de reinicio que recargaba Whisper cada 12 segundos. El
asistente tiene que elegir el dispositivo que haya en cada arranque, y esperar
sin morir cuando no hay ninguno.

Uso:  .venv/bin/python eval/test_audio.py
      .venv/bin/python eval/test_audio.py --portaudio

La suite predeterminada no inicializa PortAudio. --portaudio agrega un sondeo
real de un dispositivo inexistente, con timeout; requiere acceso al servidor
de sonido. Salidas: 0 aprobado, 1 fallo, 2 sondeo no disponible.
"""
import argparse
import importlib.util
import os
import queue
import subprocess
import sys
import time
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np  # noqa: E402
from scipy.signal import lfilter, resample_poly  # noqa: E402

# sounddevice inicializa PortAudio al importarse y puede bloquearse en un
# sandbox sin acceso a PipeWire/PulseAudio. Los tests del callback y selección
# deben poder correr allí; solo el sondeo opcional importa el backend real.
def _dispositivo_invalido(*args, **kwargs):
    raise ValueError("dispositivo inexistente en el doble de PortAudio")


_sd_doble = SimpleNamespace(
    query_devices=lambda *a, **k: DISPOSITIVOS_FALSOS[a[0]] if a else DISPOSITIVOS_FALSOS,
    check_input_settings=_dispositivo_invalido,
    InputStream=_dispositivo_invalido,
)
with patch.dict(sys.modules, {"sounddevice": _sd_doble}):
    # Cargar con otro nombre evita dejar un src.audio con backend falso en el
    # proceso si un runner importa varias suites juntas.
    _spec_audio = importlib.util.spec_from_file_location("audio_en_evaluacion", BASE_DIR / "src" / "audio.py")
    audio_mod = importlib.util.module_from_spec(_spec_audio)
    _spec_audio.loader.exec_module(audio_mod)
from src.config import CAPTURE_SAMPLE_RATE, RECORDINGS_DIR, SAMPLE_RATE  # noqa: E402

VERDE, ROJO, FIN = "\033[32m", "\033[31m", "\033[0m"
_fallos, _corridas = [], 0


def verificar(descripcion: str, condicion: bool, detalle: str = "") -> None:
    global _corridas
    _corridas += 1
    if condicion:
        print(f"  {VERDE}OK {FIN} {descripcion}")
    else:
        print(f"  {ROJO}MAL{FIN} {descripcion}" + (f"\n       {detalle}" if detalle else ""))
        _fallos.append(descripcion)


DISPOSITIVOS_FALSOS = [
    {"name": "HDA Intel PCH: ALC256 Analog (hw:0,0)", "max_input_channels": 2},
    {"name": "salida solamente", "max_input_channels": 0},
    {"name": "lavrate", "max_input_channels": 128},
    {"name": "upmix", "max_input_channels": 8},
    {"name": "pipewire", "max_input_channels": 128},
    {"name": "pulse", "max_input_channels": 32},
    {"name": "sysdefault", "max_input_channels": 128},
    {"name": "Mi Micrófono USB (hw:1,0)", "max_input_channels": 1},
]


def probar_orden_de_candidatos():
    print("\nOrden de preferencia de dispositivos")
    candidatos = audio_mod._candidatos(None)
    verificar("el default del sistema se prueba primero", candidatos[0] is None,
              f"dio {candidatos[:3]}")

    con_preferido = audio_mod._candidatos(7)
    verificar("un dispositivo configurado a mano gana al default",
              con_preferido[0] == 7 and None in con_preferido, f"dio {con_preferido[:3]}")

    verificar("no hay candidatos repetidos", len(candidatos) == len(set(map(repr, candidatos))))


def probar_filtro_de_plugins():
    print("\nLos plugins de ALSA no son micrófonos")
    for nombre, esperado in (("HDA Intel PCH: ALC256 Analog (hw:0,0)", True),
                              ("Mi Micrófono USB (hw:1,0)", True),
                              ("lavrate", False), ("upmix", False),
                              ("vdownmix", False), ("speexrate", False)):
        verificar(f"{nombre[:34]!r} -> {'placa' if esperado else 'plugin'}",
                  audio_mod._es_hardware_real(nombre) is esperado)

    original = audio_mod.sd.query_devices
    audio_mod.sd.query_devices = lambda *a, **k: DISPOSITIVOS_FALSOS
    try:
        indices = [c for c in audio_mod._candidatos(None) if c is not None]
        nombres = [DISPOSITIVOS_FALSOS[i]["name"] for i in indices]
    finally:
        audio_mod.sd.query_devices = original
    verificar("ningún plugin entra a la lista de candidatos",
              not any(n in ("lavrate", "upmix") for n in nombres), f"dio {nombres}")
    verificar("las dos placas reales sí entran",
              sum("(hw:" in n for n in nombres) == 2, f"dio {nombres}")


def probar_descarte_y_fallback():
    print("\nSi un dispositivo no abre, se prueba el siguiente")
    intentados = []

    def abrir_falso(self, dispositivo):
        intentados.append(dispositivo)
        # Falla el default (el caso real al desconectar el USB) y anda el segundo.
        if dispositivo is None:
            return False
        self.stream, self.device = object(), dispositivo
        return True

    original_abrir, original_cand = audio_mod.AudioStream._abrir, audio_mod._candidatos
    audio_mod.AudioStream._abrir = abrir_falso
    audio_mod._candidatos = lambda pref: [None, 19, 0]
    try:
        stream = audio_mod.AudioStream()
        stream.start()
        verificar("el default fallido no detiene el arranque", stream.device == 19,
                  f"eligió {stream.device}")
        verificar("se probó el default antes que el resto", intentados == [None, 19],
                  f"intentó {intentados}")
    finally:
        audio_mod.AudioStream._abrir = original_abrir
        audio_mod._candidatos = original_cand


def probar_sin_dispositivos():
    print("\nSin ningún dispositivo utilizable")
    original_abrir, original_cand = audio_mod.AudioStream._abrir, audio_mod._candidatos
    audio_mod.AudioStream._abrir = lambda self, d: False
    audio_mod._candidatos = lambda pref: [None]
    try:
        try:
            audio_mod.AudioStream().start(esperar=False)
            verificar("esperar=False levanta AudioDeviceUnavailable", False, "no levantó nada")
        except audio_mod.AudioDeviceUnavailable:
            verificar("esperar=False levanta AudioDeviceUnavailable", True)

        # esperar=True reintenta: al tercer intento aparece un dispositivo.
        intentos = {"n": 0}

        def aparece_al_tercero(self, dispositivo):
            intentos["n"] += 1
            if intentos["n"] < 3:
                return False
            self.stream, self.device = object(), dispositivo
            return True

        audio_mod.AudioStream._abrir = aparece_al_tercero
        t0 = time.perf_counter()
        stream = audio_mod.AudioStream()
        stream.start(esperar=True, intervalo_espera=0.05)
        verificar("esperar=True reintenta hasta que aparece un micrófono",
                  stream.device is None and intentos["n"] == 3,
                  f"intentos={intentos['n']}, tardó {time.perf_counter()-t0:.2f}s")
    finally:
        audio_mod.AudioStream._abrir = original_abrir
        audio_mod._candidatos = original_cand


def probar_perdida_en_caliente():
    print("\nEl dispositivo desaparece con el asistente andando")
    original = audio_mod.TIMEOUT_DISPOSITIVO_S
    audio_mod.TIMEOUT_DISPOSITIVO_S = 0.2
    try:
        stream = audio_mod.AudioStream()
        stream.q = queue.Queue()
        t0 = time.perf_counter()
        try:
            stream.read_chunk(1280)
            verificar("read_chunk levanta AudioDeviceLost en vez de colgar", False, "no levantó")
        except audio_mod.AudioDeviceLost:
            tardanza = time.perf_counter() - t0
            verificar("read_chunk levanta AudioDeviceLost en vez de colgar", True)
            verificar("no espera más que el timeout configurado", tardanza < 1.0,
                      f"tardó {tardanza:.2f}s")
    finally:
        audio_mod.TIMEOUT_DISPOSITIVO_S = original


def probar_silencio_de_alsa():
    print("\nEl ruido de PortAudio no puede inundar el journal")
    leer, escribir = os.pipe()
    original = os.dup(2)
    try:
        os.dup2(escribir, 2)
        with audio_mod._sin_ruido_de_alsa():
            os.write(2, b"esto no tiene que aparecer\n")
        os.write(2, b"esto si\n")
    finally:
        os.dup2(original, 2)
        os.close(original)
        os.close(escribir)
    salida = os.read(leer, 4096)
    os.close(leer)
    verificar("lo escrito al fd 2 dentro del bloque se descarta",
              b"no tiene que aparecer" not in salida, f"salida={salida!r}")
    verificar("el fd 2 queda restaurado al salir", b"esto si" in salida, f"salida={salida!r}")

    verificar("un dispositivo inválido se rechaza sin abrir un stream",
              audio_mod.AudioStream()._abrir(99999) is False)


def probar_portaudio() -> bool:
    """Sondeo independiente; nunca intenta abrir un micrófono existente."""
    print("\nSondeo opcional del backend PortAudio real", flush=True)
    guion = (
        "import sys; sys.path.insert(0, %r)\n"
        "from src.audio import AudioStream\n"
        "assert AudioStream()._abrir(99999) is False\n"
        "print('PORTAUDIO_PROBADO')\n"
    ) % str(BASE_DIR)
    try:
        proceso = subprocess.run([sys.executable, "-c", guion], capture_output=True, timeout=10)
    except subprocess.TimeoutExpired:
        print("  Sondeo incompleto: PortAudio no respondió en 10 s. "
              "Ejecutá --portaudio con acceso al servidor de sonido.", file=sys.stderr)
        return False
    lineas = len(proceso.stderr.decode(errors="replace").splitlines())
    verificar("PortAudio real rechaza un dispositivo inválido sin ruido en stderr",
              proceso.returncode == 0 and b"PORTAUDIO_PROBADO" in proceso.stdout and lineas == 0,
              f"salida={proceso.returncode}, stderr={lineas} líneas")
    return True


def probar_backoff():
    print("\nLa espera crece en vez de barrer cada 5 s para siempre")
    esperas = []
    original_sleep, original_abrir = audio_mod.time.sleep, audio_mod.AudioStream._abrir
    audio_mod.time.sleep = esperas.append
    audio_mod.AudioStream._abrir = lambda self, d: len(esperas) >= 4
    try:
        audio_mod.AudioStream().start(esperar=True, intervalo_espera=5.0)
    finally:
        audio_mod.time.sleep = original_sleep
        audio_mod.AudioStream._abrir = original_abrir
    verificar("cada reintento espera el doble que el anterior",
              esperas == [5.0, 10.0, 20.0, 40.0], f"esperó {esperas}")
    verificar("el tope está en ESPERA_MAXIMA_S",
              audio_mod.ESPERA_MAXIMA_S == 60.0 and max(esperas) <= audio_mod.ESPERA_MAXIMA_S)


def _por_callback(muestras, bloques):
    """Pasa `muestras` (int16 a CAPTURE_SAMPLE_RATE) por el callback real, de a bloques."""
    stream = audio_mod.AudioStream()
    salida, i = [], 0
    for largo in bloques:
        trozo = muestras[i:i + largo]
        if not len(trozo):
            break
        i += len(trozo)
        stream.callback(trozo.reshape(-1, 1), len(trozo), None, None)
        while not stream.q.empty():
            salida.append(stream.q.get())
    return np.concatenate(salida) if salida else np.array([], dtype=np.int16)


def _pico_en(senal, frecuencia, ancho=200):
    """Amplitud máxima del espectro alrededor de `frecuencia`, en Hz de salida."""
    freqs = np.fft.rfftfreq(len(senal), 1 / SAMPLE_RATE)
    espectro = np.abs(np.fft.rfft(senal.astype(np.float64) * np.hanning(len(senal))))
    return espectro[(freqs > frecuencia - ancho) & (freqs < frecuencia + ancho)].max()


def probar_antialias():
    """
    El remuestreo a 16 kHz tiene que filtrar antes de decimar.

    Hasta el 2026-09-08 se quedaba con una de cada tres muestras sin filtro, así
    que todo lo que el micrófono captaba por encima de 8 kHz se plegaba sobre la
    banda de voz. Estas pruebas no abren el micrófono: le dan audio sintético al
    callback real.
    """
    print("\nAntialias del remuestreo")
    factor = audio_mod.FACTOR_DECIMACION
    fir = audio_mod.FIR_ANTIALIAS
    if fir is None:
        verificar("no hace falta filtrar (captura y salida a la misma frecuencia)",
                  CAPTURE_SAMPLE_RATE == SAMPLE_RATE)
        return

    rng = np.random.default_rng(0)
    ruido = rng.normal(0, 3000, CAPTURE_SAMPLE_RATE).astype(np.int16)

    # 1. Filtrar de a bloques tiene que dar lo mismo que filtrar todo junto: si el
    #    estado del filtro no se conservara, habría un transitorio por bloque.
    referencia = np.clip(
        np.rint(lfilter(fir, 1.0, ruido.astype(np.float64))[::factor]), -32768, 32767
    ).astype(np.int16)
    obtenido = _por_callback(ruido, [1024] * (len(ruido) // 1024 + 1))
    n = min(len(referencia), len(obtenido))
    verificar("el filtrado en streaming es idéntico a filtrar la señal entera",
              n > 0 and np.array_equal(referencia[:n], obtenido[:n]),
              f"{n} muestras comparadas")

    # 2. La rejilla de decimación no puede depender del tamaño de bloque, que
    #    PortAudio elige y no tiene por qué ser múltiplo del factor.
    irregular = _por_callback(ruido, [479, 1013, 257, 2048, 71] * 40)
    n = min(len(obtenido), len(irregular))
    verificar("el resultado no depende del tamaño de bloque",
              n > 0 and np.array_equal(obtenido[:n], irregular[:n]),
              f"bloques de 1024 vs irregulares, {n} muestras")

    # 3. Un tono por encima de Nyquist tiene que desaparecer, no reaparecer abajo.
    t = np.arange(CAPTURE_SAMPLE_RATE) / CAPTURE_SAMPLE_RATE
    tono = (20000 * np.sin(2 * np.pi * 10000 * t)).astype(np.int16)
    alias_sin_filtro = _pico_en(tono[::factor], 6000)   # lo que hacía el código viejo
    alias_con_filtro = _pico_en(_por_callback(tono, [1024] * 47), 6000)
    atenuacion = 20 * np.log10(max(alias_con_filtro, 1e-9) / max(alias_sin_filtro, 1e-9))
    verificar("un tono de 10 kHz ya no se pliega sobre los 6 kHz", atenuacion < -40,
              f"{atenuacion:.1f} dB respecto de decimar sin filtrar")

    # 4. Y la voz real tiene que pasar intacta: subir una grabación de 16 kHz a la
    #    frecuencia de captura y volver por el callback devuelve la misma señal.
    grabaciones = sorted(RECORDINGS_DIR.glob("utterance_*.wav"))
    if not grabaciones:
        print("       (sin grabaciones en recordings/: me salteo la prueba con voz real)")
        return
    with wave.open(str(grabaciones[-1])) as w:
        voz = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    subida = np.clip(resample_poly(voz.astype(np.float64), factor, 1), -32768, 32767).astype(np.int16)
    vuelta = _por_callback(subida, [1024] * (len(subida) // 1024 + 1))
    m = min(len(voz), len(vuelta))
    # El FIR retrasa 80 muestras de captura: a 16 kHz son 26 2/3, no 27.
    # Buscar solo retardos enteros daba 0.9583 sobre este mismo WAV aunque la
    # señal estaba intacta. Se compensa el retardo conocido en la rejilla de
    # captura, donde es entero, antes de medir la correlación (0.9995).
    retardo_captura = (len(fir) - 1) // 2
    vuelta_alineada = resample_poly(vuelta.astype(np.float64), factor, 1)[retardo_captura::factor]
    n_alineadas = min(len(voz), len(vuelta_alineada))
    correlacion = np.corrcoef(voz[:n_alineadas], vuelta_alineada[:n_alineadas])[0, 1]
    verificar("la voz real atraviesa el filtro sin deformarse", correlacion > 0.99,
              f"correlación {correlacion:.4f} con retardo compensado {retardo_captura / factor:.4f} muestras")

    nivel = lambda a: 20 * np.log10(max(float(np.sqrt(np.mean(a.astype(float) ** 2))), 1e-9) / 32768)
    verificar("el filtro no cambia el nivel de la voz",
              abs(nivel(voz[:m]) - nivel(vuelta[:m])) < 0.5,
              f"{nivel(voz[:m]):.2f} -> {nivel(vuelta[:m]):.2f} dBFS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portaudio", action="store_true",
                        help="agrega un sondeo real del backend, sin grabar voz")
    args = parser.parse_args()
    print("Pruebas de entrada de audio — ninguna abre el micrófono real")
    probar_orden_de_candidatos()
    probar_filtro_de_plugins()
    probar_descarte_y_fallback()
    probar_sin_dispositivos()
    probar_perdida_en_caliente()
    probar_silencio_de_alsa()
    probar_backoff()
    probar_antialias()
    backend_disponible = probar_portaudio() if args.portaudio else True
    print(f"\n{len(_fallos)} fallos de {_corridas} verificaciones")
    for f in _fallos:
        print(f"  {ROJO}-{FIN} {f}")
    return 1 if _fallos else (0 if backend_disponible else 2)


if __name__ == "__main__":
    raise SystemExit(main())

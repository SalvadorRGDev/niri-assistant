"""
Pruebas de la selección de dispositivo de entrada.

Ninguna abre el micrófono de verdad: se reemplazan `_abrir` y la enumeración por
dobles, así que corren con niri.service levantado y sin robarle el dispositivo.

Existen por un incidente real: al desconectar el micrófono USB, el `default` del
sistema dejó de resolver a una fuente de audio, `AudioStream.start()` murió y
systemd entró en un bucle de reinicio que recargaba Whisper cada 12 segundos. El
asistente tiene que elegir el dispositivo que haya en cada arranque, y esperar
sin morir cuando no hay ninguno.

Uso:  .venv/bin/python eval/test_audio.py     (código de salida 1 si algo falla)
"""
import os
import queue
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import audio as audio_mod  # noqa: E402

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

    # Abrir un dispositivo inexistente es el camino que imprimía ~10.000 líneas.
    guion = (
        "import sys; sys.path.insert(0, %r)\n"
        "from src.audio import AudioStream\n"
        "AudioStream()._abrir(99999)\n"
    ) % str(BASE_DIR)
    proceso = subprocess.run([sys.executable, "-c", guion], capture_output=True, timeout=60)
    lineas = len(proceso.stderr.decode(errors="replace").splitlines())
    verificar("abrir un dispositivo inválido no escribe nada en stderr", lineas == 0,
              f"escribió {lineas} líneas")


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


def main() -> int:
    print("Pruebas de entrada de audio — ninguna abre el micrófono real")
    probar_orden_de_candidatos()
    probar_filtro_de_plugins()
    probar_descarte_y_fallback()
    probar_sin_dispositivos()
    probar_perdida_en_caliente()
    probar_silencio_de_alsa()
    probar_backoff()
    print(f"\n{len(_fallos)} fallos de {_corridas} verificaciones")
    for f in _fallos:
        print(f"  {ROJO}-{FIN} {f}")
    return 1 if _fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())

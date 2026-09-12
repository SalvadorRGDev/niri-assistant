import os
import queue
import time
from contextlib import contextmanager

import sounddevice as sd
import numpy as np
from scipy.signal import firwin, lfilter
from src.config import SAMPLE_RATE, CAPTURE_SAMPLE_RATE, CHANNELS, AUDIO_DEVICE
from src.logger import get_logger

logger = get_logger("AudioStream")

# --- Remuestreo de CAPTURE_SAMPLE_RATE a SAMPLE_RATE -------------------------
#
# Quedarse con una de cada `FACTOR_DECIMACION` muestras es correcto solo si
# antes se saca todo lo que está por encima de la nueva frecuencia de Nyquist
# (8 kHz). Sin ese filtro —que es lo que hacía este módulo hasta el 2026-09-08—
# el contenido de 8 a 24 kHz no se pierde: se *pliega* sobre la banda de voz,
# donde ya no se distingue de la señal real. Es ruido que depende de la
# respuesta de agudos de cada micrófono, así que además vuelve el audio del
# micrófono interno distinto del USB con el que se entrenó el wake word.
#
# 161 coeficientes dejan la banda que se pliega 53.8 dB abajo con un rizado de
# 0.02 dB hasta 6 kHz, y cuestan 20 us por bloque: 0.20% de un núcleo, medido
# acá. El corte va en 7.5 kHz para llegar atenuado a 8 kHz sin comerse los
# agudos de las fricativas.
FACTOR_DECIMACION = CAPTURE_SAMPLE_RATE // SAMPLE_RATE
if CAPTURE_SAMPLE_RATE % SAMPLE_RATE:
    # Con una razón no entera la decimación daría una frecuencia de muestreo
    # equivocada sin ningún síntoma visible: todo el pipeline seguiría andando
    # con el audio acelerado. Mejor no arrancar.
    raise ValueError(
        f"CAPTURE_SAMPLE_RATE ({CAPTURE_SAMPLE_RATE}) tiene que ser múltiplo "
        f"entero de SAMPLE_RATE ({SAMPLE_RATE}) para decimar."
    )
FIR_ANTIALIAS = (firwin(161, 7500, fs=CAPTURE_SAMPLE_RATE)
                 if FACTOR_DECIMACION > 1 else None)

# Nombres de los dispositivos mediados por el servidor de sonido. Se prueban
# después del default porque siguen la fuente que el usuario eligió en el
# escritorio, así que cuando el micrófono USB está enchufado, es el que usan.
_DISPOSITIVOS_DE_SERVIDOR = ("pipewire", "pulse", "sysdefault")

# Segundos sin un solo chunk del callback antes de dar por perdido el dispositivo.
# Un chunk llega cada pocos milisegundos mientras la placa esté viva, así que 5 s
# solo se cumplen si el micrófono desapareció (desenchufado, o el servidor de
# sonido lo soltó).
TIMEOUT_DISPOSITIVO_S = 5.0


# Tope del tiempo de espera entre barridos. Sin él, un dispositivo que falla de
# forma persistente se reintentaría cada 5 s para siempre.
ESPERA_MAXIMA_S = 60.0


@contextmanager
def _sin_ruido_de_alsa():
    """
    Silencia el descriptor 2 mientras PortAudio toca el dispositivo.

    ALSA y PortAudio escriben sus errores directo al fd 2 desde C, sin pasar por
    `logging`. Un solo `start()` sobre un dispositivo en mal estado llegó a
    imprimir ~10.000 líneas, que es exactamente el `RateLimitBurst` de journald:
    el resultado fue que systemd descartó TODOS los logs del servicio, los
    nuestros incluidos, y el asistente quedó ciego para diagnosticar.

    Nuestro logger escribe a stdout (ver src/logger.py), así que esto no tapa
    ningún mensaje propio. Aun así, adentro del bloque no se loguea nada: la
    excepción se guarda y se reporta afuera.
    """
    original = os.dup(2)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 2)
            yield
        finally:
            os.dup2(original, 2)
            os.close(devnull)
    finally:
        # Anidado a propósito: si `os.open` falla, `original` ya existe y sin
        # este finally quedaría abierto para siempre.
        os.close(original)


class AudioDeviceUnavailable(RuntimeError):
    """No hay ningún dispositivo de entrada utilizable en este momento."""


class AudioDeviceLost(RuntimeError):
    """El dispositivo estaba abierto y dejó de entregar audio."""


def _es_hardware_real(nombre: str) -> bool:
    """
    True si el nombre corresponde a una placa de captura y no a un plugin de ALSA.

    ALSA expone como "dispositivos de entrada" un montón de plugins de proceso
    (lavrate, speexrate, upmix, vdownmix...) que no son micrófonos. Además de
    inútiles son peligrosos: abrirlos a ciegas para probarlos hace segfaultear a
    PortAudio (verificado en este equipo), así que nunca entran a la lista.
    """
    return "(hw:" in nombre


def _candidatos(preferido) -> list:
    """
    Dispositivos de entrada a probar, en orden de preferencia.

    El orden importa: primero lo que el usuario configuró, después el default del
    sistema —que es el que sigue al micrófono USB cuando está enchufado— y recién
    al final las placas concretas. Así el USB gana cuando está, y el micrófono
    interno queda como red cuando no.
    """
    candidatos = []
    if preferido is not None:
        candidatos.append(preferido)
    candidatos.append(None)  # default del sistema

    try:
        dispositivos = list(sd.query_devices())
    except Exception as e:
        logger.warning(f"No se pudo enumerar dispositivos de audio: {e}")
        return candidatos

    por_nombre = {}
    for indice, d in enumerate(dispositivos):
        if d["max_input_channels"] < 1:
            continue
        por_nombre.setdefault(d["name"].lower(), indice)

    for nombre in _DISPOSITIVOS_DE_SERVIDOR:
        if nombre in por_nombre:
            candidatos.append(por_nombre[nombre])

    for indice, d in enumerate(dispositivos):
        if d["max_input_channels"] >= 1 and _es_hardware_real(d["name"]):
            candidatos.append(indice)

    # Sin duplicados, conservando el orden.
    vistos, unicos = set(), []
    for c in candidatos:
        clave = repr(c)
        if clave not in vistos:
            vistos.add(clave)
            unicos.append(c)
    return unicos


def _resolver_por_nombre(nombre: str):
    """Traduce un nombre parcial de dispositivo a su índice, o None si no está."""
    try:
        for indice, d in enumerate(sd.query_devices()):
            if nombre.lower() in d["name"].lower() and d["max_input_channels"] > 0:
                return indice
    except Exception as e:
        logger.warning(f"No se pudo buscar el dispositivo {nombre!r}: {e}")
    return None


def _descripcion(dispositivo) -> str:
    if dispositivo is None:
        return "default del sistema"
    try:
        return f"[{dispositivo}] {sd.query_devices(dispositivo)['name']}"
    except Exception:
        return str(dispositivo)


class AudioStream:
    def __init__(self, device=AUDIO_DEVICE):
        # `device` es una preferencia, no una promesa: el dispositivo real se
        # elige en cada start(), porque el micrófono USB no siempre está
        # conectado y el default del sistema falla cuando no hay ninguna fuente.
        self.preferido = device
        self.device = None
        self.q = queue.Queue()
        self.stream = None
        self.buffer = np.array([], dtype=np.int16)

        # Estado del filtro antialias y fase de la decimación. Los dos viven
        # entre llamadas porque el callback ve el audio de a bloques y el
        # filtrado tiene que ser continuo: reiniciarlos en cada bloque metería
        # un transitorio cada pocos milisegundos, y perder la fase cambiaría el
        # espaciado entre muestras justo en el borde. Solo los escribe el
        # callback, que corre en un único hilo de PortAudio.
        self._zi_antialias = np.zeros(len(FIR_ANTIALIAS) - 1) if FIR_ANTIALIAS is not None else None
        self._fase_decimacion = 0

        if isinstance(self.preferido, str):
            self.preferido = _resolver_por_nombre(self.preferido)

    def callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio status: {status}")

        # Obtenemos la data original (shape: frames, channels)
        raw_data = indata.copy()[:, 0]

        # Antialias + decimación, manteniendo la continuidad entre bloques.
        if FIR_ANTIALIAS is not None:
            filtrado, self._zi_antialias = lfilter(
                FIR_ANTIALIAS, 1.0, raw_data.astype(np.float64), zi=self._zi_antialias)
            muestras = filtrado[self._fase_decimacion::FACTOR_DECIMACION]
            # Dónde cae la próxima muestra a conservar, ya en coordenadas del
            # bloque siguiente. Sin esto, un bloque de largo no múltiplo de
            # FACTOR_DECIMACION correría la rejilla y el audio saldría con
            # saltos de una muestra en cada borde.
            self._fase_decimacion = (self._fase_decimacion - len(raw_data)) % FACTOR_DECIMACION
            # El filtro puede sobrepasar el rango en un transitorio (Gibbs), así
            # que se recorta antes de volver a int16 en vez de dar la vuelta.
            resampled_data = np.clip(np.rint(muestras), -32768, 32767).astype(np.int16)
        else:
            resampled_data = raw_data

        self.q.put(resampled_data)

    def _abrir(self, dispositivo) -> bool:
        """Intenta abrir un dispositivo concreto. True si quedó andando."""
        stream, error = None, None
        with _sin_ruido_de_alsa():
            try:
                # check_input_settings valida sin abrir el stream. Se hace primero
                # porque abrir un dispositivo inválido puede tumbar el proceso entero.
                sd.check_input_settings(device=dispositivo, samplerate=CAPTURE_SAMPLE_RATE,
                                         channels=CHANNELS, dtype="int16")
                stream = sd.InputStream(
                    device=dispositivo,
                    samplerate=CAPTURE_SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="int16",
                    callback=self.callback,
                )
                stream.start()
            except Exception as e:
                error = e
                if stream is not None:
                    # El stream se creó pero no arrancó: cerrarlo o queda colgado.
                    try:
                        stream.close()
                    except Exception:
                        pass
                    stream = None

        if error is not None:
            logger.debug(f"Descartado {_descripcion(dispositivo)}: {str(error).splitlines()[0]}")
            return False

        self.stream = stream
        self.device = dispositivo
        return True

    def start(self, esperar: bool = True, intervalo_espera: float = 5.0):
        """
        Abre el primer dispositivo de entrada que realmente funcione.

        Con `esperar=True` no se rinde si no hay ninguno: reintenta cada
        `intervalo_espera` segundos, volviendo a enumerar cada vez, así que
        enchufar el micrófono lo levanta solo. Sin esto, el servicio moría con
        `Restart=on-failure` y entraba en un bucle que recargaba Whisper cada
        pocos segundos (visto en producción al desconectar el micrófono USB).
        """
        aviso_emitido = False
        espera = intervalo_espera
        while True:
            for dispositivo in _candidatos(self.preferido):
                if self._abrir(dispositivo):
                    logger.info(
                        f"Entrada de audio: {_descripcion(dispositivo)} "
                        f"(captura {CAPTURE_SAMPLE_RATE}Hz, salida {SAMPLE_RATE}Hz)"
                    )
                    return

            if not esperar:
                raise AudioDeviceUnavailable(
                    "No hay ningún dispositivo de entrada disponible a "
                    f"{CAPTURE_SAMPLE_RATE}Hz."
                )
            if not aviso_emitido:
                aviso_emitido = True
                logger.warning(
                    "No encontré ningún micrófono utilizable. Esperando a que "
                    "aparezca uno."
                )
            time.sleep(espera)
            # Backoff: un micrófono que no vuelve no justifica barrer los
            # dispositivos cada 5 s indefinidamente.
            espera = min(espera * 2, ESPERA_MAXIMA_S)

    def reset_buffers(self):
        """
        Descarta el audio pendiente (cola del callback y buffer parcial).

        Lo usa MainLoop al reabrir el dispositivo: lo que quedó en la cola es de
        antes del corte y mezclarlo con lo nuevo produciría una orden partida al
        medio.
        """
        self.q.queue.clear()
        self.buffer = np.array([], dtype=np.int16)

    def stop(self):
        logger.info("Stopping audio stream...")
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.warning(f"Error cerrando el stream de audio: {e}")
            self.stream = None

    def read_chunk(self, size: int) -> np.ndarray:
        """
        Read exactly `size` samples from the audio stream.

        Si el dispositivo deja de entregar audio (típicamente porque lo
        desenchufaron), la cola se seca y esta función colgaría para siempre.
        Por eso el get() tiene timeout y levanta AudioDeviceLost: MainLoop lo
        atrapa y vuelve a elegir dispositivo en vez de quedarse mudo.
        """
        while len(self.buffer) < size:
            try:
                new_data = self.q.get(timeout=TIMEOUT_DISPOSITIVO_S)
            except queue.Empty:
                raise AudioDeviceLost(
                    f"El dispositivo de entrada dejó de entregar audio "
                    f"({TIMEOUT_DISPOSITIVO_S:.0f}s sin datos)."
                ) from None
            self.buffer = np.concatenate((self.buffer, new_data))

        chunk = self.buffer[:size]
        self.buffer = self.buffer[size:]
        return chunk

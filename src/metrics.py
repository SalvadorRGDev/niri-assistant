"""
Métricas de latencia por etapa, en formato append-only (logs/metrics.jsonl).

Complementa a src/audit.py sin pisarlo: el log de auditoría responde "qué se
intentó hacer y con qué texto", y este responde "cuánto tardó cada etapa". Por
eso acá NO se escribe ni la transcripción ni el contenido de la orden — solo el
nombre de la acción, tiempos y conteos de tokens. Ese recorte es deliberado:
metrics.jsonl está pensado para poder mirarse y compartirse sin exponer nada de
lo que el usuario dijo frente al micrófono.

Existe porque las cifras de rendimiento del README salieron de mediciones
manuales y no se recalculan solas: una regresión de 300 ms puede vivir semanas
sin que nadie la note.
"""
import json
import os
import datetime
from contextlib import contextmanager
from time import perf_counter
from typing import Optional

from src.config import LOGS_DIR
from src.logger import get_logger

logger = get_logger("Metrics")

# La ruta no vive en src/config.py a propósito: ese archivo es configuración de
# producción con valores medidos (umbrales de wake word, VAD y STT) y se toca lo
# menos posible. Igual se puede redirigir por entorno para pruebas.
METRICS_PATH = os.getenv("NIRI_METRICS_PATH", str(LOGS_DIR / "metrics.jsonl"))

# Interruptor para poder correr pruebas manuales sin ensuciar la serie histórica.
METRICS_ENABLED = os.getenv("NIRI_METRICS", "1").lower() not in ("0", "false", "no")

# Techo de tamaño: a ~200 bytes por turno son varios cientos de miles de turnos.
# Al superarlo se rota a un único .1, sin borrar nada más (el archivo viejo se
# reemplaza recién en la rotación siguiente).
MAX_BYTES = 5 * 1024 * 1024


class TurnoMetricas:
    """
    Acumula los tiempos de un turno de voz y escribe una línea al terminar.

    Uso típico:
        turno = TurnoMetricas()
        with turno.etapa("stt"):
            texto = stt.transcribe(audio)
        turno.registrar(accion="eliminar", tokens_salida=36)

    Nunca lanza excepciones hacia el llamador: perder una métrica no puede
    tumbar el asistente en medio de una orden.
    """

    def __init__(self):
        self._inicio = perf_counter()
        self.etapas: dict[str, float] = {}

    @contextmanager
    def etapa(self, nombre: str):
        """Mide un bloque y lo suma a la etapa `nombre` (acumula si se repite:
        en un turno con diálogo de ubicación, el STT corre más de una vez)."""
        inicio = perf_counter()
        try:
            yield
        finally:
            self.sumar(nombre, (perf_counter() - inicio) * 1000)

    def sumar(self, nombre: str, ms: float) -> None:
        self.etapas[nombre] = round(self.etapas.get(nombre, 0.0) + ms, 1)

    def registrar(self, accion: Optional[str] = None, **extra) -> None:
        if not METRICS_ENABLED:
            return
        try:
            entrada = {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "accion": accion,
                "total_ms": round((perf_counter() - self._inicio) * 1000, 1),
                **{f"{k}_ms": v for k, v in self.etapas.items()},
            }
            entrada.update({k: v for k, v in extra.items() if v is not None})
            _escribir(entrada)
        except Exception as e:
            logger.error(f"No se pudo registrar la métrica del turno: {e}")


def _escribir(entrada: dict) -> None:
    ruta = METRICS_PATH
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    try:
        if os.path.getsize(ruta) > MAX_BYTES:
            os.replace(ruta, f"{ruta}.1")
    except FileNotFoundError:
        pass
    with open(ruta, "a", encoding="utf-8") as f:
        f.write(json.dumps(entrada, ensure_ascii=False) + "\n")

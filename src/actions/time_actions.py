"""
Tiempo: hora, temporizador, alarma, recordatorio, nota.

'temporizador'/'alarma' son avisos programados: se guardan con su hora de
disparo y el loop principal los chequea periódicamente (ver check_due_timers,
llamado desde MainLoop.run() en el estado IDLE). 'recordatorio'/'nota' no
tienen hora — son texto que se guarda y se lee de vuelta cuando lo piden,
sin ningún mecanismo de disparo activo.

Persistencia simple a JSON (mismo espíritu que logs/audit.log) para
sobrevivir reinicios del servicio.
"""
import json
import re
import datetime
import threading
from pathlib import Path
from typing import List, Optional

from src.schemas import FileAction, ExecutionResult
from src.config import TIMERS_STATE_PATH, NOTES_STATE_PATH, DATA_DIR
from src.logger import get_logger

logger = get_logger("Actions.Time")

_lock = threading.Lock()


def _load_json(path: Path) -> list:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error leyendo {path}: {e}")
        return []


def _save_json(path: Path, data: list) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error guardando {path}: {e}")


# --- hora --------------------------------------------------------------

def hora(action: FileAction) -> ExecutionResult:
    ahora = datetime.datetime.now().strftime("%H:%M")
    return ExecutionResult(text=f"Son las {ahora}.")


# --- temporizador / alarma ----------------------------------------------
# Se guardan como timestamp absoluto (epoch, datetime.now().timestamp()) para
# que sobrevivan un reinicio del servicio sin desfasarse.

def _add_pending(disparo_epoch: float, mensaje: str) -> None:
    with _lock:
        pendientes = _load_json(TIMERS_STATE_PATH)
        pendientes.append({"disparo": disparo_epoch, "mensaje": mensaje})
        _save_json(TIMERS_STATE_PATH, pendientes)


def temporizador(action: FileAction) -> ExecutionResult:
    minutos_str = (action.cantidad or "").strip()
    match = re.search(r"\d+(\.\d+)?", minutos_str)
    if not match:
        return ExecutionResult(text="¿En cuántos minutos querés que te avise?")

    minutos = float(match.group())
    disparo = datetime.datetime.now().timestamp() + minutos * 60
    mensaje = f"¡Se cumplió el temporizador de {minutos_str} minutos!"
    if action.contenido:
        mensaje += f" ({action.contenido})"
    _add_pending(disparo, mensaje)
    logger.info(f"Temporizador creado: {minutos} min -> dispara a las {datetime.datetime.fromtimestamp(disparo):%H:%M:%S}")
    return ExecutionResult(text=f"Listo, te aviso en {minutos_str} minutos.")


def alarma(action: FileAction) -> ExecutionResult:
    hora_str = (action.cantidad or "").strip()
    match = re.match(r"^(\d{1,2})[:h.](\d{2})$", hora_str)
    if not match:
        return ExecutionResult(text="¿A qué hora querés la alarma? Decime, por ejemplo, 7:30.")

    h, m = int(match.group(1)), int(match.group(2))
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return ExecutionResult(text="Esa hora no es válida.")

    ahora = datetime.datetime.now()
    disparo_dt = ahora.replace(hour=h, minute=m, second=0, microsecond=0)
    if disparo_dt <= ahora:
        disparo_dt += datetime.timedelta(days=1)

    mensaje = f"¡Es la hora de tu alarma de las {hora_str}!"
    if action.contenido:
        mensaje += f" ({action.contenido})"
    _add_pending(disparo_dt.timestamp(), mensaje)
    logger.info(f"Alarma creada para las {disparo_dt:%Y-%m-%d %H:%M}")
    return ExecutionResult(text=f"Listo, alarma puesta para las {hora_str}.")


def check_due_timers() -> List[str]:
    """
    Llamado desde el loop principal (estado IDLE): devuelve los mensajes de
    los temporizadores/alarmas que ya vencieron, y los saca de la lista de
    pendientes. Barato de llamar seguido: si no hay pendientes, no toca disco.
    """
    with _lock:
        pendientes = _load_json(TIMERS_STATE_PATH)
        if not pendientes:
            return []

        ahora = datetime.datetime.now().timestamp()
        vencidos = [p for p in pendientes if p["disparo"] <= ahora]
        if not vencidos:
            return []

        restantes = [p for p in pendientes if p["disparo"] > ahora]
        _save_json(TIMERS_STATE_PATH, restantes)
        return [v["mensaje"] for v in vencidos]


# --- recordatorio / nota --------------------------------------------------
# Sin hora de disparo: se guardan y se leen de vuelta cuando el usuario
# pregunta (contenido vacío = "léeme lo que tengo guardado").

def _add_or_list(path: Path, action: FileAction, etiqueta: str) -> ExecutionResult:
    texto = (action.contenido or "").strip()

    with _lock:
        items = _load_json(path)
        if texto:
            items.append({"texto": texto, "fecha": datetime.datetime.now().isoformat()})
            _save_json(path, items)
            logger.info(f"{etiqueta.capitalize()} guardado: {texto!r}")
            return ExecutionResult(text="Anotado.")

        if not items:
            return ExecutionResult(text=f"No tenés {etiqueta}s guardados.")
        ultimos = items[-5:]
        listado = "; ".join(i["texto"] for i in ultimos)
        return ExecutionResult(text=f"Tus {etiqueta}s: {listado}.")


def recordatorio(action: FileAction) -> ExecutionResult:
    return _add_or_list(NOTES_STATE_PATH, action, "recordatorio")


def nota(action: FileAction) -> ExecutionResult:
    return _add_or_list(NOTES_STATE_PATH, action, "nota")

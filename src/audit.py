"""
Log de auditoría append-only para Niri.

Registra cada intento de acción del Executor (permitida, bloqueada o fallida)
con timestamp, transcripción original y resultado. Este archivo nunca se abre
en modo escritura ('w'): solo se agrega ('a'), para preservar el historial
completo como exige la capa 5 del plan de implementación.
"""
import json
import datetime
from typing import Optional

from src.config import AUDIT_LOG_PATH
from src.logger import get_logger

logger = get_logger("Audit")


def log_action(raw_text: Optional[str], action_data: dict, resultado: str, extra: Optional[dict] = None) -> None:
    """
    Escribe una entrada de auditoría. Nunca lanza excepción hacia el llamador:
    un fallo al auditar no debe tumbar el asistente, pero sí se reporta.

    `extra` permite anexar metadatos puntuales (p.ej. si una acción destructiva
    fue confirmada por voz) sin cambiar la forma base de cada entrada.
    """
    try:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "transcripcion": raw_text,
            "accion": action_data,
            "resultado": resultado,
        }
        if extra:
            entry.update(extra)
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"No se pudo escribir en el log de auditoría: {e}")

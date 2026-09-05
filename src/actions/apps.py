"""
Abrir aplicaciones, enfocar ventanas y ejecutar rutinas.

Todo pasa por src.app_registry: el NLU solo aporta un alias (`nombre`), nunca
un comando — igual filosofía de seguridad que ALLOWED_ROOTS en paths.py.
"""
import json
import subprocess

from src.schemas import FileAction, ExecutionResult
from src.app_registry import resolve_app, resolve_routine
from src.logger import get_logger

logger = get_logger("Actions.Apps")


def _launch(argv: list) -> bool:
    try:
        subprocess.Popen(argv, start_new_session=True,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        logger.error(f"Error lanzando {argv}: {e}")
        return False


def abrir_aplicacion(action: FileAction) -> ExecutionResult:
    nombre = (action.nombre or "").strip()
    argv = resolve_app(nombre)
    if not argv:
        logger.warning(f"App no reconocida en la whitelist: {nombre!r}")
        return ExecutionResult(text=f"No tengo '{nombre}' registrada como aplicación conocida.")

    if _launch(argv):
        logger.info(f"App abierta: {nombre} -> {argv}")
        return ExecutionResult(text=f"Abriendo {nombre}.")
    return ExecutionResult(text=f"Hubo un error al intentar abrir {nombre}.")


def ejecutar_rutina(action: FileAction) -> ExecutionResult:
    nombre = (action.nombre or "").strip()
    apps = resolve_routine(nombre)
    if not apps:
        logger.warning(f"Rutina no reconocida: {nombre!r}")
        return ExecutionResult(text=f"No tengo ninguna rutina llamada '{nombre}'.")

    abiertas = []
    for alias in apps:
        argv = resolve_app(alias)
        if argv and _launch(argv):
            abiertas.append(alias)

    if not abiertas:
        return ExecutionResult(text=f"No pude abrir nada de la rutina '{nombre}'.")
    logger.info(f"Rutina ejecutada: {nombre} -> {abiertas}")
    return ExecutionResult(text=f"Listo, activé {nombre}: abrí {', '.join(abiertas)}.")


def _find_window_address(nombre: str) -> str:
    """Busca la primera ventana cuyo `class` contenga `nombre` (case-insensitive)."""
    try:
        out = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True,
                              text=True, timeout=3, check=True)
        clients = json.loads(out.stdout)
    except Exception as e:
        logger.error(f"Error consultando hyprctl clients: {e}")
        return ""

    needle = nombre.strip().lower()
    for c in clients:
        if needle in (c.get("class") or "").lower():
            return c.get("address", "")
    return ""


def enfocar_ventana(action: FileAction) -> ExecutionResult:
    nombre = (action.nombre or "").strip()
    if not nombre:
        return ExecutionResult(text="¿Qué ventana querés que enfoque?")

    address = _find_window_address(nombre)
    if not address:
        return ExecutionResult(text=f"No encontré ninguna ventana abierta de '{nombre}'.")

    try:
        subprocess.run(["hyprctl", "dispatch", "focuswindow", f"address:{address}"],
                        capture_output=True, timeout=3, check=True)
        logger.info(f"Ventana enfocada: {nombre} -> {address}")
        return ExecutionResult(text=f"Listo, enfoqué {nombre}.")
    except Exception as e:
        logger.error(f"Error enfocando ventana: {e}")
        return ExecutionResult(text="Hubo un error al intentar cambiar de ventana.")

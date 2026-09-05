"""
Ruteo de acciones que NO son de archivos hacia su módulo correspondiente.

Executor (src/executor.py) sigue siendo exclusivamente para archivos —
estas acciones no tienen `base_dir`/`is_safe_path`, así que viven en un
dispatcher separado para no mezclar los dos modelos de seguridad.
"""
from src.schemas import FileAction, ExecutionResult
from src.logger import get_logger
from src.actions import apps as apps_actions
from src.actions import system as system_actions
from src.actions import smalltalk as smalltalk_actions
from src.actions import utils as utils_actions
from src.actions import media as media_actions
from src.actions import time_actions

logger = get_logger("Actions.Dispatch")

# Acciones que MainLoop debe rutear acá en vez de a Executor.
# Se va completando a medida que se implementa cada fase del plan.
NON_FILE_ACTIONS = {
    "abrir_aplicacion", "enfocar_ventana", "ejecutar_rutina",
    "volumen", "brillo", "captura_pantalla", "wifi", "bluetooth", "energia",
    "saludo", "chiste", "despedida",
    "calculo", "conversion", "traduccion",
    "control_musica",
    "hora", "temporizador", "alarma", "recordatorio", "nota",
}

_HANDLERS = {
    "abrir_aplicacion": apps_actions.abrir_aplicacion,
    "enfocar_ventana": apps_actions.enfocar_ventana,
    "ejecutar_rutina": apps_actions.ejecutar_rutina,
    "volumen": system_actions.volumen,
    "brillo": system_actions.brillo,
    "captura_pantalla": system_actions.captura_pantalla,
    "wifi": system_actions.wifi,
    "bluetooth": system_actions.bluetooth,
    "energia": system_actions.energia,
    "saludo": smalltalk_actions.saludo,
    "chiste": smalltalk_actions.chiste,
    "despedida": smalltalk_actions.despedida,
    "calculo": utils_actions.calculo,
    "conversion": utils_actions.conversion,
    "traduccion": utils_actions.traduccion,
    "control_musica": media_actions.control_musica,
    "hora": time_actions.hora,
    "temporizador": time_actions.temporizador,
    "alarma": time_actions.alarma,
    "recordatorio": time_actions.recordatorio,
    "nota": time_actions.nota,
}


def dispatch(action: FileAction) -> ExecutionResult:
    handler = _HANDLERS.get(action.action)
    if handler is None:
        logger.warning(f"Acción sin handler todavía: {action.action}")
        return ExecutionResult(text="Todavía no sé cómo hacer esa acción.")
    return handler(action)

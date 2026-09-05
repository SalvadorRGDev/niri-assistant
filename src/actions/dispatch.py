"""
Ruteo de acciones que NO son de archivos hacia su módulo correspondiente.

Executor (src/executor.py) sigue siendo exclusivamente para archivos —
estas acciones no tienen `base_dir`/`is_safe_path`, así que viven en un
dispatcher separado para no mezclar los dos modelos de seguridad.
"""
from typing import Optional

from src.schemas import FileAction, ExecutionResult
from src.audit import log_action
from src.logger import get_logger
from src.actions import apps as apps_actions
from src.actions import system as system_actions
from src.actions import smalltalk as smalltalk_actions
from src.actions import utils as utils_actions
from src.actions import media as media_actions
from src.actions import time_actions

logger = get_logger("Actions.Dispatch")

# Acciones que no tocan archivos pero SÍ son irreversibles: exigen la misma
# confirmación por voz que 'eliminar' y 'mover' en el Executor (§4.4 del harness).
#
# 'energia' está acá por un hallazgo del banco de evaluación: a "apagá el
# bluetooth" el NLU le contestaba {"action": "energia", "cantidad": "apagar"},
# que llegaba directo a `systemctl poweroff` sin preguntar nada. El prompt ya
# advertía en mayúsculas que no confundiera bluetooth con energía y el modelo lo
# hizo igual — de ahí que la defensa no pueda vivir solo en el prompt.
CONFIRMATION_REQUIRED = {"energia"}

_PREGUNTAS_ENERGIA = {
    "suspender": "¿Confirmás que suspenda el equipo?",
    "apagar": "¿Confirmás que apague la computadora?",
    "reiniciar": "¿Confirmás que reinicie la computadora?",
}

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


def _pregunta_de_confirmacion(action: FileAction) -> Optional[str]:
    """
    Devuelve la pregunta a hacer antes de ejecutar, o None si no hace falta.

    Está guiada por CONFIRMATION_REQUIRED y no por una lista de acciones escrita
    acá adentro: agregar una acción al conjunto tiene que bastar para que quede
    protegida. Si alguna vez se agrega una sin pregunta propia, se confirma igual
    con una genérica — el default es preguntar, nunca ejecutar.

    El único None con acción marcada es cuando todavía no se sabe QUÉ haría (la
    'cantidad' no se reconoce): ahí el handler repregunta por su cuenta y no
    ejecuta nada, así que confirmar sería pedir dos veces lo mismo.
    """
    if action.action not in CONFIRMATION_REQUIRED:
        return None

    if action.action == "energia":
        operacion = system_actions.normalizar_energia(action.cantidad)
        if operacion is None:
            return None
        return _PREGUNTAS_ENERGIA[operacion]

    return f"¿Confirmás que ejecute la acción {action.action}?"


def dispatch(action: FileAction, raw_text: Optional[str] = None,
             confirmed: bool = False) -> ExecutionResult:
    """
    Ejecuta una acción que no es de archivos, o prepara su confirmación.

    Con `confirmed=False` (el caso normal), las acciones de CONFIRMATION_REQUIRED
    NO se ejecutan: se devuelve la pregunta con `needs_confirmation=True` y es
    MainLoop quien la hace por voz. Solo vuelve acá con `confirmed=True` si el
    usuario dijo que sí de forma inequívoca (src/confirm.py falla hacia NO).

    Todo intento queda en el log de auditoría, se haya ejecutado o no: hasta
    ahora solo se auditaban las acciones de archivos, así que un apagado por voz
    no dejaba ningún rastro (§4.6).
    """
    handler = _HANDLERS.get(action.action)

    if handler is None:
        logger.warning(f"Acción sin handler todavía: {action.action}")
        resultado = ExecutionResult(text="Todavía no sé cómo hacer esa acción.")
    else:
        pregunta = _pregunta_de_confirmacion(action) if not confirmed else None
        if pregunta is not None:
            logger.info(f"Acción irreversible pendiente de confirmación: {action.action}/{action.cantidad}")
            resultado = ExecutionResult(text=pregunta, needs_confirmation=True)
        else:
            resultado = handler(action)

    log_action(
        raw_text,
        action.model_dump(),
        resultado.text,
        extra={
            "confirmado": confirmed,
            "requiere_confirmacion": resultado.needs_confirmation,
            "origen": "dispatch",
        },
    )
    return resultado

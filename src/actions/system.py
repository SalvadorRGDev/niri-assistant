"""
Control del sistema: volumen, brillo, captura de pantalla, wifi, bluetooth, energía.

Herramientas verificadas en el entorno real (Hyprland + PipeWire, CachyOS):
wpctl (volumen), brightnessctl (brillo), grim (captura), nmcli (wifi),
bluetoothctl (bluetooth), systemctl (suspender/apagar/reiniciar).
"""
import subprocess
import datetime
from pathlib import Path

from src.schemas import FileAction, ExecutionResult
from src.logger import get_logger

logger = get_logger("Actions.System")

SCREENSHOTS_DIR = Path.home() / "Pictures" / "Screenshots"

_STEP = "10%"


def _run(argv: list, timeout: float = 5.0) -> bool:
    try:
        subprocess.run(argv, capture_output=True, timeout=timeout, check=True)
        return True
    except Exception as e:
        logger.error(f"Error ejecutando {argv}: {e}")
        return False


def _direction(cantidad: str) -> str:
    """
    Normaliza 'cantidad' a 'subir'/'bajar'/'silenciar', default 'subir'.

    El default 'subir' existe porque para volumen y brillo el prompt del NLU
    pide asumir esa dirección cuando el usuario no la dice. Pero ese mismo
    default era un bug en wifi y bluetooth: el banco de evaluación mostró que
    a "apagá el wifi" el NLU le pone a veces cantidad="apagar" en vez de
    "bajar", y sin las formas de apagar listadas abajo eso caía en 'subir' y
    PRENDÍA justo lo que se pedía apagar. Se reconocen acá en vez de confiar
    en que el modelo respete el vocabulario del prompt.
    """
    c = (cantidad or "").strip().lower()
    if any(w in c for w in ("baj", "-", "menos", "reduc",
                             "apag", "desconect", "desactiv")):
        return "bajar"
    if any(w in c for w in ("silenci", "mute", "mut")):
        return "silenciar"
    return "subir"


def volumen(action: FileAction) -> ExecutionResult:
    d = _direction(action.cantidad)
    if d == "silenciar":
        ok = _run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"])
        return ExecutionResult(text="Listo." if ok else "No pude cambiar el silencio.")
    delta = f"{_STEP}+" if d == "subir" else f"{_STEP}-"
    ok = _run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", delta])
    return ExecutionResult(text=f"Volumen {'subido' if d == 'subir' else 'bajado'}." if ok
                            else "No pude cambiar el volumen.")


def brillo(action: FileAction) -> ExecutionResult:
    d = _direction(action.cantidad)
    delta = f"{_STEP}+" if d == "subir" else f"{_STEP}-"
    ok = _run(["brightnessctl", "set", delta])
    return ExecutionResult(text=f"Brillo {'subido' if d == 'subir' else 'bajado'}." if ok
                            else "No pude cambiar el brillo.")


def captura_pantalla(action: FileAction) -> ExecutionResult:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = SCREENSHOTS_DIR / f"captura_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    ok = _run(["grim", str(filename)], timeout=8.0)
    if ok:
        logger.info(f"Captura guardada en {filename}")
        return ExecutionResult(text="Listo, guardé la captura de pantalla.")
    return ExecutionResult(text="No pude tomar la captura de pantalla.")


def wifi(action: FileAction) -> ExecutionResult:
    d = _direction(action.cantidad)
    estado = "off" if d == "bajar" or d == "silenciar" else "on"
    ok = _run(["nmcli", "radio", "wifi", estado])
    return ExecutionResult(text=f"Wifi {'activado' if estado == 'on' else 'desactivado'}." if ok
                            else "No pude cambiar el estado del wifi.")


def bluetooth(action: FileAction) -> ExecutionResult:
    d = _direction(action.cantidad)
    estado = "off" if d == "bajar" or d == "silenciar" else "on"
    ok = _run(["bluetoothctl", "power", estado])
    return ExecutionResult(text=f"Bluetooth {'activado' if estado == 'on' else 'desactivado'}." if ok
                            else "No pude cambiar el estado del bluetooth.")


_ENERGIA_CMDS = {
    "suspender": ["systemctl", "suspend"],
    "apagar": ["systemctl", "poweroff"],
    "reiniciar": ["systemctl", "reboot"],
}
# El NLU devuelve 'cantidad' en lenguaje natural (ver ejemplos few-shot en nlu.py);
# este mapa cubre sinónimos comunes hacia las 3 claves soportadas arriba.
_ENERGIA_ALIASES = {
    "suspender": "suspender", "suspende": "suspender", "duerme": "suspender",
    "dormir": "suspender", "hiberna": "suspender",
    "apagar": "apagar", "apaga": "apagar", "apagate": "apagar",
    "reiniciar": "reiniciar", "reinicia": "reiniciar", "reboot": "reiniciar",
}


def normalizar_energia(cantidad) -> str | None:
    """
    Traduce la 'cantidad' hablada a 'suspender'/'apagar'/'reiniciar', o None si
    no se reconoce.

    Es pública porque src/actions/dispatch.py necesita saber QUÉ va a pasar
    antes de que pase, para poder pedir la confirmación por voz nombrando la
    operación concreta. Sin esto, la confirmación tendría que ser genérica o
    duplicar este mapa.
    """
    return _ENERGIA_ALIASES.get((cantidad or "").strip().lower())


def energia(action: FileAction) -> ExecutionResult:
    accion = normalizar_energia(action.cantidad)
    if accion is None:
        return ExecutionResult(text="¿Querés que suspenda, apague o reinicie el equipo?")

    logger.warning(f"Acción de energía solicitada por voz: {accion}")
    ok = _run(_ENERGIA_CMDS[accion], timeout=3.0)
    # Si el comando dispara la suspensión/apagado, es posible que ni lleguemos a
    # hablar la respuesta — no pasa nada, es aceptable para esta acción.
    return ExecutionResult(text=f"{accion.capitalize()}ando el equipo." if ok
                            else f"No pude {accion} el equipo.")

"""
Control básico de música vía MPRIS (D-Bus), usando `jeepney` (D-Bus puro en
Python, sin dependencias de sistema como playerctl). Controla lo que YA está
sonando en cualquier reproductor MPRIS activo (Spotify, navegador, etc.) —
no busca ni elige canciones, eso es 'reproducir_cancion' (fase Spotify API).
"""
from jeepney import DBusAddress, new_method_call
from jeepney.io.blocking import open_dbus_connection

from src.schemas import FileAction, ExecutionResult
from src.logger import get_logger

logger = get_logger("Actions.Media")

_DBUS_ADDR = DBusAddress("/org/freedesktop/DBus", bus_name="org.freedesktop.DBus",
                          interface="org.freedesktop.DBus")

_MPRIS_METHODS = {
    "reproducir": "Play",
    "pausar": "Pause",
    "alternar": "PlayPause",
    "siguiente": "Next",
    "anterior": "Previous",
}

# El NLU no siempre usa el vocabulario exacto del prompt: el banco de evaluación
# mostró que a "apagá la música" le pone cantidad="apagar", que no es ninguno de
# los métodos de arriba, y Niri terminaba repreguntando en vez de pausar. Estos
# alias absorben las formas habituales antes de buscar el método MPRIS.
_ALIAS_CANTIDAD = {
    "apagar": "pausar", "apaga": "pausar", "apagá": "pausar",
    "parar": "pausar", "para": "pausar", "pará": "pausar", "pausa": "pausar",
    "detener": "pausar", "detene": "pausar", "stop": "pausar",
    "play": "reproducir", "reanudar": "reproducir", "continuar": "reproducir",
    "seguir": "reproducir", "segui": "reproducir",
}


def _find_mpris_player(conn) -> str:
    msg = new_method_call(_DBUS_ADDR, "ListNames")
    reply = conn.send_and_get_reply(msg, timeout=3)
    names = reply.body[0]
    for name in names:
        if name.startswith("org.mpris.MediaPlayer2."):
            return name
    return ""


def control_musica(action: FileAction) -> ExecutionResult:
    cantidad = (action.cantidad or "").strip().lower()
    cantidad = _ALIAS_CANTIDAD.get(cantidad, cantidad)
    method = _MPRIS_METHODS.get(cantidad)
    if method is None:
        return ExecutionResult(text="¿Querés que reproduzca, pause, o pase a la siguiente/anterior canción?")

    try:
        conn = open_dbus_connection(bus="SESSION")
    except Exception as e:
        logger.error(f"Error conectando a D-Bus: {e}")
        return ExecutionResult(text="No pude conectarme al sistema de audio.")

    try:
        player = _find_mpris_player(conn)
        if not player:
            return ExecutionResult(text="No encontré ningún reproductor de música abierto.")

        player_addr = DBusAddress("/org/mpris/MediaPlayer2", bus_name=player,
                                   interface="org.mpris.MediaPlayer2.Player")
        msg = new_method_call(player_addr, method)
        conn.send_and_get_reply(msg, timeout=3)
        logger.info(f"MPRIS {method} enviado a {player}")
        return ExecutionResult(text="Listo.")
    except Exception as e:
        logger.error(f"Error controlando reproductor MPRIS: {e}")
        return ExecutionResult(text="Hubo un error al controlar la música.")
    finally:
        conn.close()

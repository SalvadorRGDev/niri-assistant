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

"""Movimiento atómico sin reemplazo para Linux, sin respaldo que sobrescriba.

La autorización de las rutas pertenece al Executor. Este módulo solo garantiza
que el destino no sea reemplazado si aparece antes de la operación del kernel.
Entre dispositivos o sin soporte de renameat2 se cancela con OSError: no se
intenta copiar/borrar ni usar os.rename o shutil.move como respaldo.
"""
import ctypes
import errno
import os
from pathlib import Path
import sys


# ABI comprobada en los headers locales: stdio.h declara renameat2(int,
# const char *, int, const char *, unsigned int); fcntl.h fija AT_FDCWD=-100
# y linux/fs.h define RENAME_NOREPLACE=(1 << 0).
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def _load_renameat2():
    if sys.platform != "linux":
        raise OSError(errno.ENOSYS, "Movimiento atómico sin reemplazo requiere Linux")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        function = libc.renameat2
    except (AttributeError, OSError) as exc:
        raise OSError(errno.ENOSYS, "renameat2 no está disponible; movimiento cancelado") from exc
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                        ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    return function


def move_no_replace(source: Path, destination: Path) -> None:
    """Mueve archivo/directorio al destino exacto, o lanza OSError sin reemplazar.

    Un destino existente, incluido un enlace roto, produce FileExistsError.
    EXDEV, ausencia de soporte y los demás errores del sistema se propagan.
    No resuelve enlaces: si el origen es un enlace, se mueve el propio enlace.
    """
    source_bytes, destination_bytes = os.fsencode(source), os.fsencode(destination)
    # c_char_p truncaría silenciosamente una ruta con NUL. Rechazarla antes
    # de entrar en libc evita operar sobre un nombre distinto al solicitado.
    if b"\x00" in source_bytes or b"\x00" in destination_bytes:
        raise OSError(errno.EINVAL, "La ruta contiene un carácter nulo")

    renameat2 = _load_renameat2()
    ctypes.set_errno(0)
    result = renameat2(_AT_FDCWD, source_bytes, _AT_FDCWD,
                       destination_bytes, _RENAME_NOREPLACE)
    if result != 0:
        error = ctypes.get_errno() or errno.EIO
        raise OSError(error, os.strerror(error), os.fspath(source), None,
                      os.fspath(destination))

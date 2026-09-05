"""
Suite de regresión de los invariantes de seguridad (§4 del harness).

Verifica los contratos que hacen seguro al asistente, sin ejecutar nada real:
`_run` se reemplaza por un doble que solo anota el comando que se habría corrido,
así que ninguna prueba apaga la máquina, toca el wifi ni borra archivos.

Nació de un hallazgo del banco de evaluación: "apagá el bluetooth" se clasificaba
como energia/apagar y llegaba a `systemctl poweroff` sin pedir confirmación.
Estas pruebas existen para que ese agujero no se pueda reabrir en silencio.

Uso:  .venv/bin/python eval/test_seguridad.py     (código de salida 1 si algo falla)
"""
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.schemas import FileAction                      # noqa: E402
from src.executor import Executor                       # noqa: E402
from src.confirm import interpret_confirmation          # noqa: E402
from src.paths import ALLOWED_ROOTS, DEFAULT_ROOT       # noqa: E402
from src.nlu import corregir_intencion                  # noqa: E402
from src.actions import system as system_actions        # noqa: E402
from src.actions import media as media_actions          # noqa: E402
from src.actions import dispatch as dispatch_mod        # noqa: E402
from src import audit as audit_mod                      # noqa: E402

VERDE, ROJO, FIN = "\033[32m", "\033[31m", "\033[0m"
_fallos = []
_corridas = 0

# Comandos que un doble de `_run` habría ejecutado. Ninguna prueba llega al sistema.
comandos_ejecutados: list[list] = []


def _run_falso(argv, timeout=5.0):
    comandos_ejecutados.append(list(argv))
    return True


def verificar(descripcion: str, condicion: bool, detalle: str = "") -> None:
    global _corridas
    _corridas += 1
    if condicion:
        print(f"  {VERDE}OK {FIN} {descripcion}")
    else:
        print(f"  {ROJO}MAL{FIN} {descripcion}" + (f"\n       {detalle}" if detalle else ""))
        _fallos.append(descripcion)


def accion(**campos) -> FileAction:
    campos.setdefault("ruta_base", "")
    return FileAction(**campos)


def probar_confirmacion_de_energia():
    print("\n§4.4 — Confirmación obligatoria antes de apagar/suspender/reiniciar")
    for cantidad, esperado in (("apagar", "poweroff"), ("suspender", "suspend"), ("reiniciar", "reboot")):
        comandos_ejecutados.clear()
        r = dispatch_mod.dispatch(accion(action="energia", cantidad=cantidad), raw_text="prueba")
        verificar(f"'{cantidad}' sin confirmar pide confirmación y NO ejecuta",
                  r.needs_confirmation and not comandos_ejecutados,
                  f"needs_confirmation={r.needs_confirmation}, comandos={comandos_ejecutados}")

        comandos_ejecutados.clear()
        r = dispatch_mod.dispatch(accion(action="energia", cantidad=cantidad),
                                  raw_text="prueba", confirmed=True)
        verificar(f"'{cantidad}' confirmado sí ejecuta ({esperado})",
                  comandos_ejecutados == [["systemctl", esperado]],
                  f"comandos={comandos_ejecutados}")

    comandos_ejecutados.clear()
    r = dispatch_mod.dispatch(accion(action="energia", cantidad="bluetooth"), raw_text="prueba")
    verificar("cantidad no reconocida repregunta sin ejecutar",
              not r.needs_confirmation and not comandos_ejecutados and "?" in r.text)


def probar_aparatos_no_son_energia():
    print("\nCorrección determinista — nombrar wifi/bluetooth descarta 'energia'")
    casos = [
        ("apagá el bluetooth", "bluetooth", "bajar"),
        ("apagá el wifi", "wifi", "bajar"),
        ("desconectá el wi-fi", "wifi", "bajar"),
        ("APAGÁ EL BLUETOOTH", "bluetooth", "bajar"),
        ("apagá el equipo", "energia", "apagar"),
        ("suspendé la compu", "energia", "suspender"),
    ]
    for texto, accion_esperada, cantidad_esperada in casos:
        corregida = corregir_intencion(texto, accion(action="energia", cantidad="apagar"
                                                      if "equipo" not in texto and "compu" not in texto
                                                      else ("suspender" if "compu" in texto else "apagar")))
        verificar(f"{texto!r} -> {accion_esperada}/{cantidad_esperada}",
                  corregida.action == accion_esperada and corregida.cantidad == cantidad_esperada,
                  f"dio {corregida.action}/{corregida.cantidad}")


def probar_direccion_de_aparatos():
    print("\nDirección de wifi/bluetooth — apagar tiene que apagar, no prender")
    casos = [
        ("wifi", "bajar", ["nmcli", "radio", "wifi", "off"]),
        ("wifi", "apagar", ["nmcli", "radio", "wifi", "off"]),
        ("wifi", "subir", ["nmcli", "radio", "wifi", "on"]),
        ("bluetooth", "bajar", ["bluetoothctl", "power", "off"]),
        ("bluetooth", "apagar", ["bluetoothctl", "power", "off"]),
        ("bluetooth", "prender", ["bluetoothctl", "power", "on"]),
    ]
    for act, cantidad, esperado in casos:
        comandos_ejecutados.clear()
        dispatch_mod.dispatch(accion(action=act, cantidad=cantidad), raw_text="prueba")
        verificar(f"{act} con cantidad {cantidad!r} -> {' '.join(esperado[-2:])}",
                  comandos_ejecutados == [esperado], f"dio {comandos_ejecutados}")


def probar_alias_de_musica():
    print("\nControl de música — 'apagar' es pausar, no un método inexistente")
    for cantidad, metodo in (("apagar", "Pause"), ("parar", "Pause"), ("pausar", "Pause"),
                              ("play", "Play"), ("siguiente", "Next")):
        canon = media_actions._ALIAS_CANTIDAD.get(cantidad, cantidad)
        verificar(f"{cantidad!r} -> {metodo}", media_actions._MPRIS_METHODS.get(canon) == metodo,
                  f"dio {media_actions._MPRIS_METHODS.get(canon)}")


def probar_confirmacion_de_archivos():
    print("\n§4.4 — Confirmación obligatoria en eliminar y mover (regresión)")
    ejecutor = Executor()
    r = ejecutor.execute(accion(action="eliminar", nombre="no_existe_xyz.txt"),
                          base_dir=DEFAULT_ROOT, raw_text="prueba")
    verificar("eliminar sin confirmar pide confirmación", r.needs_confirmation)

    r = ejecutor.execute(accion(action="mover", nombre="no_existe_xyz.txt", destino="clases"),
                          base_dir=DEFAULT_ROOT, destino_dir=ALLOWED_ROOTS[1], raw_text="prueba")
    verificar("mover sin confirmar pide confirmación", r.needs_confirmation)


def probar_whitelist_de_rutas():
    print("\n§4.2 — Whitelist de raíces y traversal")
    ejecutor = Executor()
    r = ejecutor.execute(accion(action="eliminar", nombre="../../.ssh/id_rsa"),
                          base_dir=DEFAULT_ROOT, raw_text="prueba", confirmed=True)
    verificar("traversal con '..' es rechazado",
              not r.needs_confirmation and "seguridad" in r.text.lower(), f"dio {r.text!r}")

    r = ejecutor.execute(accion(action="eliminar", nombre="."),
                          base_dir=DEFAULT_ROOT, raw_text="prueba")
    verificar("no se puede eliminar una raíz completa", "seguridad" in r.text.lower(), f"dio {r.text!r}")

    verificar("is_safe_path rechaza /etc/passwd", not ejecutor.is_safe_path(Path("/etc/passwd")))
    verificar("is_safe_path acepta una subcarpeta permitida",
              ejecutor.is_safe_path(DEFAULT_ROOT / "algo"))


def probar_fail_safe_de_confirmacion():
    print("\n§4.4 — Fail-safe: solo un sí inequívoco confirma")
    for respuesta in ("", None, "no", "no, mejor no", "cancelá", "quizás", "mmm", "no sé", "sí pero no"):
        verificar(f"{respuesta!r} NO confirma", interpret_confirmation(respuesta) is not True,
                  f"dio {interpret_confirmation(respuesta)}")
    for respuesta in ("sí", "si", "dale", "confirmo", "adelante"):
        verificar(f"{respuesta!r} confirma", interpret_confirmation(respuesta) is True)


def probar_auditoria():
    print("\n§4.6 — Toda acción que no es de archivos queda auditada")
    with tempfile.TemporaryDirectory() as tmp:
        original = audit_mod.AUDIT_LOG_PATH
        audit_mod.AUDIT_LOG_PATH = Path(tmp) / "audit.log"
        try:
            dispatch_mod.dispatch(accion(action="energia", cantidad="apagar"), raw_text="apagá todo")
            contenido = audit_mod.AUDIT_LOG_PATH.read_text(encoding="utf-8")
        finally:
            audit_mod.AUDIT_LOG_PATH = original
    verificar("el intento bloqueado por confirmación queda registrado",
              "energia" in contenido and "requiere_confirmacion" in contenido)
    verificar("el log guarda la transcripción original", "apagá todo" in contenido)


def main() -> int:
    system_actions._run = _run_falso  # ninguna prueba llega al sistema real
    print("Suite de seguridad de Niri — nada se ejecuta de verdad (`_run` es un doble)")
    probar_confirmacion_de_energia()
    probar_aparatos_no_son_energia()
    probar_direccion_de_aparatos()
    probar_alias_de_musica()
    probar_confirmacion_de_archivos()
    probar_whitelist_de_rutas()
    probar_fail_safe_de_confirmacion()
    probar_auditoria()

    print(f"\n{len(_fallos)} fallos de {_corridas} verificaciones")
    for f in _fallos:
        print(f"  {ROJO}-{FIN} {f}")
    return 1 if _fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())

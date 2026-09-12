"""
Resolución de ubicaciones habladas para el Executor.

Mapea nombres hablados de carpetas raíz ("Proyectos", "Clases") a las rutas
reales de la whitelist de seguridad, y arma subrutas cuando el usuario
menciona una carpeta intermedia (p.ej. "en Proyectos, carpeta trabajo").

Esta es la ÚNICA fuente de verdad sobre qué raíces existen — src/executor.py
importa ALLOWED_ROOTS/DEFAULT_ROOT de aquí para no duplicar la lista.
"""
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

# (alias hablado en minúsculas y sin acentos, ruta real). El primero de la
# lista es la raíz por defecto cuando no se puede resolver ninguna ubicación.
# Las rutas se derivan de Path.home() en vez de estar escritas a mano: son la
# frontera de seguridad del asistente, y un home hardcodeado apuntaría a un
# directorio inexistente en cualquier otra máquina. `resolve()` se mantiene
# porque es lo que neutraliza los enlaces simbólicos y los `..`.
ROOT_ALIASES: List[Tuple[str, Path]] = [
    ("proyectos", (Path.home() / "Proyectos").resolve()),
    ("clases", (Path.home() / "Clases").resolve()),
]

ALLOWED_ROOTS: List[Path] = [root for _, root in ROOT_ALIASES]
DEFAULT_ROOT: Path = ROOT_ALIASES[0][1]

# Palabras de relleno que se descartan al interpretar la subcarpeta dicha
# después del nombre de la raíz (p.ej. "en Proyectos, dentro de la carpeta
# trabajo" -> subcarpeta = ["trabajo"]).
_STOPWORDS = {"en", "la", "el", "los", "las", "carpeta", "subcarpeta",
              "dentro", "de", "del", "adentro", "por", "favor"}


def _normalize(text: str) -> str:
    """minúsculas, sin acentos, sin comas."""
    text = text.lower().strip().replace(",", " ")
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def _real_child_name(parent: Path, spoken: str) -> str:
    """
    Traduce un nombre de carpeta *hablado* (que llega siempre normalizado a
    minúsculas y sin acentos) al nombre real en disco.

    Existe porque `_normalize` destruye las mayúsculas y los acentos del nombre
    dicho, y en un sistema de archivos sensible a mayúsculas eso construye una
    ruta que no existe: "en Proyectos, carpeta Asistente" apuntaba a
    `~/Proyectos/asistente` en vez de `~/Proyectos/Asistente`. En `crear_carpeta`
    eso creaba un duplicado en minúscula al lado de la carpeta verdadera.

    Si ningún hijo de `parent` coincide, devuelve el nombre hablado tal cual:
    es el caso legítimo de crear una carpeta que todavía no existe.
    """
    try:
        for child in parent.iterdir():
            if _normalize(child.name) == spoken:
                return child.name
    except OSError:
        # parent no existe todavía, o no se puede leer: no hay nada que
        # traducir, y el nombre hablado es la mejor respuesta disponible.
        pass
    return spoken


def _subpath(root: Path, parts: List[str]) -> Optional[Path]:
    """
    Arma `root/parts...` traduciendo cada segmento al nombre real en disco.
    Devuelve None si algún segmento intenta salirse de la raíz.

    El chequeo de "/" ya no puede dispararse desde parse_location_speech (que
    separa por barras antes de llamar acá), pero se mantiene: es la última
    defensa si alguien vuelve a llamar a esta función con texto crudo.
    """
    if not parts:
        return root
    if any("/" in part or "\\" in part or part in (".", "..") for part in parts):
        return None
    path = root
    for part in parts:
        path = path / _real_child_name(path, part)
    return path


def parse_location_speech(text: Optional[str]) -> Optional[Path]:
    """
    Interpreta una frase (respuesta hablada, o el campo `ruta_base`/`destino`
    que devuelve el NLU) y devuelve la ruta base resultante dentro de la
    whitelist, o None si no reconoció ninguna raíz válida.

    Ejemplos:
      "proyectos"                         -> ~/Proyectos
      "en Clases"                         -> ~/Clases
      "Proyectos, carpeta trabajo"        -> ~/Proyectos/trabajo
      "clases/ada"  (ruta del NLU)        -> ~/Clases/ADA
      "clases/../etc"                     -> None (segmento inseguro)
      "la carpeta fotos" (sin raíz)       -> None (ambiguo, hay que preguntar)
      ""  /  None                         -> None

    Si la subcarpeta dicha ya existe en disco, se devuelve con su nombre real
    (mayúsculas y acentos incluidos): "Proyectos, carpeta asistente" resuelve a
    ~/Proyectos/Asistente. Si no existe, se conserva el nombre tal como se dijo,
    que es lo que permite crearla.
    """
    if not text:
        return None

    # Modo ruta. El NLU devuelve la ubicación anidada como una ruta con barras
    # ("clases/ada", y a veces con prefijo: "~/clases/ada"), y sin esto ninguna
    # raíz coincidía: "clases/ada" era UNA sola palabra, así que el asistente
    # volvía a preguntar la ubicación aunque la hubieras dicho. Medido con
    # qwen2.5:3b el 2026-09-11 sobre "crea una carpeta llamada parciales en
    # clases, dentro de la carpeta ada".
    #
    # Acá el orden SÍ importa: solo cuentan los segmentos posteriores a la raíz.
    # Si se aceptaran los anteriores, "/home/usuario/clases/ada" se convertiría
    # en ~/Clases/home/usuario/ada, es decir subcarpetas inventadas en silencio.
    if "/" in text:
        segments = [s for s in (_normalize(s) for s in text.split("/")) if s.strip()]
        for alias, root in ROOT_ALIASES:
            if alias in segments:
                posteriores = segments[segments.index(alias) + 1:]
                return _subpath(root, [w for seg in posteriores
                                       for w in seg.split() if w not in _STOPWORDS])
        return None

    # Modo frase hablada. Acá la raíz puede aparecer al final ("dentro de la
    # carpeta trabajo en proyectos"), así que vale en cualquier posición.
    words = _normalize(text).split()
    if not words:
        return None
    for alias, root in ROOT_ALIASES:
        if alias in words:
            return _subpath(root, [w for w in words if w != alias and w not in _STOPWORDS])
    return None


def speakable_path(path: Path) -> str:
    """
    Convierte una ruta resuelta (dentro de la whitelist) a una frase natural
    para que el TTS la lea en voz alta, p.ej.:
      ~/Proyectos              -> "Proyectos"
      ~/Proyectos/trabajo      -> "Proyectos, carpeta trabajo"
      ~/Proyectos/trabajo/x    -> "Proyectos, carpeta trabajo, carpeta x"
    Si la ruta no cae dentro de ninguna raíz conocida (no debería pasar tras
    is_safe_path), devuelve la ruta absoluta tal cual como último recurso.
    """
    resolved = path.resolve()
    for alias, root in ROOT_ALIASES:
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            continue
        label = alias.capitalize()
        if str(rel) == ".":
            return label
        return label + "".join(f", carpeta {part}" for part in rel.parts)
    return str(resolved)

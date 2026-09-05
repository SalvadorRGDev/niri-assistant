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


def parse_location_speech(text: Optional[str]) -> Optional[Path]:
    """
    Interpreta una frase (respuesta hablada, o el campo `ruta_base`/`destino`
    que devuelve el NLU) y devuelve la ruta base resultante dentro de la
    whitelist, o None si no reconoció ninguna raíz válida.

    Ejemplos:
      "proyectos"                         -> ~/Proyectos
      "en Clases"                         -> ~/Clases
      "Proyectos, carpeta trabajo"        -> ~/Proyectos/trabajo
      "la carpeta fotos" (sin raíz)       -> None (ambiguo, hay que preguntar)
      ""  /  None                         -> None

    Si la subcarpeta dicha ya existe en disco, se devuelve con su nombre real
    (mayúsculas y acentos incluidos): "Proyectos, carpeta asistente" resuelve a
    ~/Proyectos/Asistente. Si no existe, se conserva el nombre tal como se dijo,
    que es lo que permite crearla.
    """
    if not text:
        return None
    words = _normalize(text).split()
    if not words:
        return None

    for alias, root in ROOT_ALIASES:
        if alias in words:
            remainder = [w for w in words if w != alias and w not in _STOPWORDS]
            if not remainder:
                return root
            # Ningún segmento hablado puede contener separadores ni ser "."/"..".
            # El ejecutor igual lo rechazaría con is_safe_path(), pero no tiene
            # sentido construir una ruta de escape para que la validen abajo.
            if any("/" in w or "\\" in w or w in (".", "..") for w in remainder):
                return None
            path = root
            for part in remainder:
                path = path / _real_child_name(path, part)
            return path
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

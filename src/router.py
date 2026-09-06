"""
Router determinista: resuelve por expresiones regulares las órdenes de
vocabulario cerrado, sin pasar por el LLM.

Por qué existe: "subí el volumen" o "qué hora es" no tienen ninguna ambigüedad
que justifique 300 ms de GPU y un modelo capaz de alucinar una intención. El
router las resuelve en microsegundos, sin VRAM y sin margen de error.

Dos reglas de diseño, y las dos son de seguridad:

1. **Solo acciones sin parámetros libres.** Nada de archivos: `crear_carpeta`,
   `eliminar`, `mover`, `listar` y `leer` necesitan extraer un nombre y una
   ubicación del habla, y una regex que se equivoque ahí borra el archivo
   equivocado. Esas siguen siendo del NLU, siempre.

2. **Ante la duda, no contesta.** Toda regla usa `fullmatch`: la orden entera
   tiene que encajar, no alcanza con que contenga las palabras. Si sobra o
   falta algo, `route()` devuelve None y la orden sigue su camino normal hacia
   el NLU. Un falso negativo cuesta 300 ms; un falso positivo ejecuta algo que
   el usuario no pidió.

`energia` (apagar/suspender/reiniciar) queda deliberadamente afuera: es rara en
el uso real y es la única de este grupo que es irreversible, así que no gana
nada con ahorrarse 300 ms y sí pierde si una regex la dispara de más.
"""
import re
import unicodedata
from typing import Optional

from src.schemas import FileAction
from src.logger import get_logger

logger = get_logger("Router")

# Muletillas que pueden rodear a la orden sin cambiar su sentido. Se permiten
# antes y después para que "che niri, subí el volumen por favor" encaje igual
# que "subí el volumen", sin tener que escribir la combinatoria a mano.
_PRE = r"(?:(?:che|niri|hola niri|oye|eh|dale|ok|okay|por favor|porfa|a ver)[,\s]+)*"
_POST = (r"(?:[,\s]+(?:por favor|porfa|gracias|dale|niri|ahora|ya|un poco|un poquito"
         r"|de nuevo|otra vez|de vuelta))*")

_UN_POCO = r"(?:un poco\s+|un poquito\s+|algo\s+)?"
_EL = r"(?:el\s+|la\s+|lo\s+)?"

# (patrón sin anclar, acción, cantidad). El orden importa solo para la
# legibilidad: las reglas son mutuamente excluyentes por construcción.
_REGLAS = [
    # --- volumen ---
    (rf"(?:subi|sube|subime|subile|aumenta|aumentame|levanta|mas)\s+{_UN_POCO}{_EL}(?:volumen|audio|sonido)",
     "volumen", "subir"),
    (rf"(?:baja|bajame|bajale|reduci|reduce|disminui|menos)\s+{_UN_POCO}{_EL}(?:volumen|audio|sonido)",
     "volumen", "bajar"),
    (rf"(?:silencia|silenciame|mutea|muteame|calla|callate)\s*{_EL}?(?:volumen|audio|sonido|musica)?",
     "volumen", "silenciar"),

    # --- brillo (par confundible con volumen: el sustantivo es lo que decide) ---
    (rf"(?:subi|sube|subime|subile|aumenta|aumentame|levanta|mas)\s+{_UN_POCO}{_EL}brillo"
     rf"(?:\s+de\s+{_EL}pantalla)?",
     "brillo", "subir"),
    (rf"(?:baja|bajame|bajale|reduci|reduce|disminui|menos)\s+{_UN_POCO}{_EL}brillo"
     rf"(?:\s+de\s+{_EL}pantalla)?",
     "brillo", "bajar"),

    # --- wifi y bluetooth (nunca 'energia': ver src/nlu.py) ---
    (rf"(?:apaga|apagame|desconecta|desconectame|desactiva|corta)\s+{_EL}wi\s?fi",
     "wifi", "bajar"),
    (rf"(?:prende|prendeme|encende|enciende|activa|conecta|conectame)\s+{_EL}wi\s?fi",
     "wifi", "subir"),
    (rf"(?:apaga|apagame|desconecta|desconectame|desactiva|corta)\s+{_EL}bluetooth",
     "bluetooth", "bajar"),
    (rf"(?:prende|prendeme|encende|enciende|activa|conecta|conectame)\s+{_EL}bluetooth",
     "bluetooth", "subir"),

    # --- música (control de lo que ya suena) ---
    (rf"(?:pausa|pausame|para|parame|frena|detene|deten|apaga)\s+{_EL}(?:musica|cancion|tema|reproduccion)",
     "control_musica", "pausar"),
    (rf"(?:segui|sigue|continua|reanuda|reproduci|reproduce|play)"
     rf"(?:\s+{_EL}(?:musica|cancion|tema|reproduccion|reproduciendo))?",
     "control_musica", "reproducir"),
    (rf"(?:pasa|pasame|poneme|pone|cambia)\s+a\s+{_EL}(?:siguiente|proxima)(?:\s+(?:cancion|tema|pista))?",
     "control_musica", "siguiente"),
    (r"(?:siguiente|proxima)(?:\s+(?:cancion|tema|pista))",
     "control_musica", "siguiente"),
    (rf"(?:volve|vuelve|volvete|regresa|pasa)\s+a\s+{_EL}(?:cancion\s+)?anterior",
     "control_musica", "anterior"),
    (r"(?:cancion|tema)\s+anterior",
     "control_musica", "anterior"),

    # --- hora ---
    (rf"(?:que hora es|que hora tenes|decime {_EL}hora|dame {_EL}hora|{_EL}hora)",
     "hora", None),

    # --- captura de pantalla ---
    (rf"(?:saca|sacame|toma|tomame|hace|haceme|hazme)\s+(?:una\s+)?captura(?:\s+de\s+{_EL}pantalla)?",
     "captura_pantalla", None),
    (r"screenshot",
     "captura_pantalla", None),

    # --- charla ---
    # "buenas noches" queda afuera a propósito: puede ser saludo o despedida
    # según la hora y el tono, así que esa la decide el NLU.
    (r"(?:hola|buenas|buen dia|buenos dias|buenas tardes|que tal|que onda|como andas)",
     "saludo", None),
    (r"(?:chau|adios|nos vemos|hasta luego|hasta manana|me voy|hasta la proxima)",
     "despedida", None),
    (r"(?:contame|conta|decime|deci|tirame|tira|mandate)\s+(?:un\s+)?chiste",
     "chiste", None),
]

_REGLAS_COMPILADAS = [
    (re.compile(_PRE + patron + _POST), accion, cantidad)
    for patron, accion, cantidad in _REGLAS
]


def _normalizar(texto: str) -> str:
    """minúsculas, sin acentos, sin signos, espacios colapsados."""
    texto = (texto or "").lower().strip()
    texto = "".join(
        c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c)
    )
    texto = re.sub(r"[¿?¡!.:;]", " ", texto)
    return " ".join(texto.split())


def route(texto: str) -> Optional[FileAction]:
    """
    Devuelve la acción si la orden encaja **entera** con una regla conocida, o
    None para que la resuelva el NLU.

    El None no es un fallo: es el caso normal para todo lo que tenga parámetros.
    """
    normalizado = _normalizar(texto)
    if not normalizado:
        return None

    for patron, accion, cantidad in _REGLAS_COMPILADAS:
        if patron.fullmatch(normalizado):
            logger.info(f"Router: '{normalizado}' -> {accion}"
                        + (f"/{cantidad}" if cantidad else ""))
            return FileAction(action=accion, ruta_base="", cantidad=cantidad)
    return None

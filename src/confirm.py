"""
Interpretación de confirmaciones de voz (sí/no) para acciones destructivas.

Diseño "fail-safe": ante cualquier ambigüedad (silencio, transcripción vacía,
palabra no reconocida), se interpreta como NO/cancelar. Nunca se asume "sí"
por defecto — eso violaría la capa 5 del plan (confirmación obligatoria).
"""
import unicodedata
from typing import Optional

_YES_WORDS = {
    "si", "sí", "confirmo", "confirmar", "confirmado",
    "adelante", "hazlo", "correcto", "afirmativo", "dale", "va", "ok", "okay",
}
_NO_WORDS = {
    "no", "cancela", "cancelar", "cancelado", "negativo",
    "detente", "para", "aborta", "abortar", "nel", "nope",
}


def _normalize(text: str) -> str:
    """minúsculas, sin acentos, sin signos de puntuación sueltos."""
    text = text.lower().strip()
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )
    for ch in ".,!¡¿?":
        text = text.replace(ch, " ")
    return text


def interpret_confirmation(text: Optional[str]) -> Optional[bool]:
    """
    Retorna True (sí), False (no), o None (ambiguo/no reconocido).
    El NO tiene prioridad si aparecen palabras de ambos grupos (p.ej.
    negaciones tipo "no, mejor no" no deben colarse como afirmativas).
    """
    if not text:
        return None
    words = set(_normalize(text).split())
    if words & _NO_WORDS:
        return False
    if words & _YES_WORDS:
        return True
    return None

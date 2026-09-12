"""
Charla: saludo / chiste / despedida.

No ejecuta nada — respuestas fijas (rotando para no sonar repetitivo).
Es la pieza de menor riesgo del plan de expansión.
"""
import random

from src.schemas import FileAction, ExecutionResult

_SALUDOS = [
    "¡Hola! ¿En qué te ayudo?",
    "Hola, acá estoy.",
    "¡Buenas! Decime qué necesitás.",
]

_CHISTES = [
    "¿Por qué los programadores prefieren el frío? Porque odian los bugs.",
    "¿Cómo se llama el campeón de buceo japonés? Tokofondo.",
    "Un byte le dice a otro: che, ¿estás bit? No, estoy bien.",
]

_DESPEDIDAS = [
    "¡Chau! Cualquier cosa, decí 'oye niri'.",
    "Nos vemos.",
    "Listo, acá quedo por si me necesitás.",
]


def saludo(action: FileAction) -> ExecutionResult:
    return ExecutionResult(text=random.choice(_SALUDOS), public_tts=True)


def chiste(action: FileAction) -> ExecutionResult:
    return ExecutionResult(text=random.choice(_CHISTES), public_tts=True)


def despedida(action: FileAction) -> ExecutionResult:
    return ExecutionResult(text=random.choice(_DESPEDIDAS), public_tts=True)

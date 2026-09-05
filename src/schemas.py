from typing import Literal, Optional
from pydantic import BaseModel, Field

class FileAction(BaseModel):
    action: Literal[
        # archivos
        "crear_carpeta", "crear_archivo", "mover", "eliminar", "listar", "leer",
        # apps, ventanas y rutinas
        "abrir_aplicacion", "enfocar_ventana", "ejecutar_rutina",
        # sistema
        "volumen", "brillo", "captura_pantalla", "wifi", "bluetooth", "energia",
        # musica
        "control_musica", "reproducir_cancion",
        # tiempo
        "hora", "temporizador", "alarma", "recordatorio", "nota",
        # charla
        "saludo", "chiste", "despedida",
        # utilidades
        "calculo", "conversion", "traduccion",
        "ninguna",
    ] = Field(
        description="La acción que el usuario quiere que el asistente realice."
    )
    ruta_base: str = Field(
        default="",
        description="La carpeta raíz donde ocurre la acción, SOLO si el usuario la mencionó "
                     "explícitamente (p.ej. 'Proyectos', 'Clases'). Si el usuario NO dijo en "
                     "qué carpeta, dejar este campo como cadena vacía \"\" — NUNCA adivinar "
                     "ni rellenar con un valor por defecto. El asistente le va a preguntar "
                     "la ubicación al usuario cuando este campo venga vacío."
    )
    nombre: Optional[str] = Field(
        default=None,
        description="El nombre del archivo o carpeta objetivo."
    )
    destino: Optional[str] = Field(
        default=None,
        description="Carpeta de destino (usado mayormente para 'mover'), SOLO si el usuario "
                     "la mencionó explícitamente. Igual que ruta_base: dejar vacío/None si no "
                     "se especificó, nunca adivinar."
    )
    contenido: Optional[str] = Field(
        default=None,
        description="Texto libre asociado a la acción: contenido a escribir en un archivo, "
                     "texto a traducir o a recordar, o la expresión matemática de un cálculo."
    )
    cantidad: Optional[str] = Field(
        default=None,
        description="Cantidad/parámetro corto asociado a la acción: 'sube'/'baja'/'silencio' "
                     "para volumen o brillo, minutos para un temporizador, la hora para una "
                     "alarma ('7:30'), o 'origen a destino' para una conversión de unidades "
                     "('celsius a fahrenheit'). Dejar vacío si no aplica."
    )


class ExecutionResult(BaseModel):
    """Resultado de una ejecución del Executor."""
    text: str = Field(description="Respuesta hablada para el usuario (TTS).")
    needs_confirmation: bool = Field(
        default=False,
        description="True si `text` es una pregunta de confirmación y la acción "
                     "todavía NO se ejecutó (caso eliminar/mover sin confirmar).",
    )

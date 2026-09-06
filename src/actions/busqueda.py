"""
Búsqueda de archivos por voz sobre el índice híbrido (src/indice.py).

Responde "¿dónde dejé el resumen de sistemas operativos?" hablando la ubicación,
sin abrir ni tocar nada: es una acción de solo lectura.

El índice se carga perezosamente y se comparte entre llamadas: son 118 MB de
modelo que no tiene sentido pagar si el usuario nunca busca.
"""
from pathlib import Path
from typing import Optional

from src.schemas import FileAction, ExecutionResult
from src.indice import Indice
from src.paths import speakable_path
from src.logger import get_logger

logger = get_logger("Actions.Busqueda")

MAX_RESULTADOS_HABLADOS = 2

_indice: Optional[Indice] = None


def obtener_indice() -> Indice:
    """Índice compartido por proceso, creado en el primer uso."""
    global _indice
    if _indice is None:
        _indice = Indice()
    return _indice


def _donde(ruta: str, raiz: str) -> str:
    """
    Describe la ubicación como la diría una persona.

    Solo la carpeta que contiene al archivo, no la ruta entera: leer en voz alta
    "en Proyectos, carpeta Asistente, src, actions" es peor que inútil, y con la
    carpeta contenedora el usuario ya sabe dónde ir.
    """
    partes = Path(ruta).parts
    if raiz in partes:
        intermedias = partes[partes.index(raiz) + 1:-1]
        if intermedias:
            return f"en {raiz}, carpeta {intermedias[-1]}"
    return f"en {raiz}"


def buscar(action: FileAction) -> ExecutionResult:
    consulta = (action.contenido or action.nombre or "").strip()
    if not consulta:
        return ExecutionResult(text="¿Qué querés que busque?")

    try:
        indice = obtener_indice()
        resultados = indice.buscar(consulta, limite=MAX_RESULTADOS_HABLADOS + 1)
    except Exception as e:
        logger.error(f"Error buscando '{consulta}': {e}")
        return ExecutionResult(text="Tuve un problema al buscar. Fijate el log.")

    if not resultados:
        return ExecutionResult(text=f"No encontré nada que coincida con {consulta}.")

    literales = [r for r in resultados if r["literal"]]
    logger.info(f"Búsqueda '{consulta}': {len(resultados)} resultados, "
                f"{len(literales)} con coincidencia literal.")

    if not literales:
        # Solo la rama vectorial contestó, y esa siempre devuelve algo. Decirlo
        # como hallazgo sería inventar: se ofrece como aproximación.
        mejor = resultados[0]
        return ExecutionResult(
            text=f"No encontré nada con esas palabras. Lo más parecido que tengo es "
                 f"{mejor['nombre']} {_donde(mejor['ruta'], mejor['raiz'])}."
        )

    principal = literales[0]
    texto = f"Encontré {principal['nombre']} {_donde(principal['ruta'], principal['raiz'])}."
    if len(literales) > 1:
        segundo = literales[1]
        texto += f" También está {segundo['nombre']} {_donde(segundo['ruta'], segundo['raiz'])}."
    return ExecutionResult(text=texto)

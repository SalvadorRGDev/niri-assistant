import sys
import argparse
import tempfile
from pathlib import Path
from src import audit as audit_mod
from src.nlu import NLU
from src.router import route
from src.executor import Executor
from src.tts import TextToSpeech
from src.actions.dispatch import dispatch as dispatch_non_file_action, NON_FILE_ACTIONS
from src.paths import parse_location_speech, DEFAULT_ROOT, speakable_path
from src.logger import get_logger

logger = get_logger("TestPipeline")


def run_test(text: str, *, voz: bool = True) -> int:
    logger.info(f"--- INICIANDO PRUEBA CON TEXTO: '{text}' ---")

    # 1. Intención: primero el router determinista, igual que en src/main_loop.py.
    action = route(text)
    if action is not None:
        logger.info(f"Resuelta por el router (sin LLM): {action}")
    else:
        nlu = NLU()
        action = nlu.parse(text)
        if nlu.last_error:
            logger.error("Prueba incompleta: el NLU no pudo interpretar la orden (%s).", nlu.last_error)
            return 2
        logger.info(f"Acción parseada por el NLU: {action}")

    # 2. Acciones que no son de archivos: van al dispatcher, no al Executor.
    # Sin esto, este script mandaba "qué hora es" al Executor y respondía que
    # faltaba un nombre de archivo, o sea que no verificaba nada de la mitad
    # del asistente. Las irreversibles se quedan en la pregunta de
    # confirmación: acá no hay voz para contestarla, y está bien así.
    if action.action in NON_FILE_ACTIONS:
        resultado = dispatch_non_file_action(action, raw_text=text)
        logger.info(f"Respuesta generada: {resultado.text} "
                    f"(needs_confirmation={resultado.needs_confirmation})")
        if resultado.text and voz:
            return 0 if TextToSpeech().speak(resultado.text, public=resultado.public_tts) else 1
        return 0

    # 3. Resolver ubicación. Este script no tiene micrófono, así que NO
    # reproduce el diálogo de "¿en qué carpeta?" de src/main_loop.py. Ante
    # ubicación ambigua termina sin actuar: una prueba no debe adivinar dónde
    # escribir ni usar una ubicación distinta de la que se pidió.
    base_dir = parse_location_speech(action.ruta_base)
    if base_dir is None:
        if action.action == "ninguna" or (action.action == "listar" and not action.ruta_base):
            base_dir = DEFAULT_ROOT
        else:
            logger.error("La orden necesita una ubicación explícita; el diálogo por voz se prueba con main.py.")
            return 2

    destino_dir = None
    if action.action == "mover" and action.destino:
        destino_dir = parse_location_speech(action.destino) or (base_dir / action.destino)

    # 4. Executor (para eliminar/mover, esto solo arma la pregunta de
    # confirmación — no ejecuta. Para probar el flujo completo con voz hay
    # que correr main.py de verdad.)
    executor = Executor()
    result = executor.execute(action, base_dir=base_dir, raw_text=text, destino_dir=destino_dir)
    logger.info(f"Respuesta generada: {result.text} (needs_confirmation={result.needs_confirmation})")

    # 5. TTS
    if result.text and voz:
        tts = TextToSpeech()
        return 0 if tts.speak(result.text, public=result.public_tts) else 1
    else:
        logger.info("Prueba sin reproducción de voz.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline por texto (puede ejecutar acciones reales).")
    parser.add_argument("texto", nargs="*", default=["hola"])
    parser.add_argument("--sin-voz", action="store_true", help="omite síntesis/reproducción")
    args = parser.parse_args()
    # Las pruebas escritas no forman parte del historial de órdenes de voz.
    with tempfile.TemporaryDirectory(prefix="niri_pipeline_") as tmp:
        audit_mod.AUDIT_LOG_PATH = Path(tmp) / "audit.log"
        raise SystemExit(run_test(" ".join(args.texto) or "hola", voz=not args.sin_voz))

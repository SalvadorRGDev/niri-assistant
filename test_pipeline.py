import sys
from src.nlu import NLU
from src.router import route
from src.executor import Executor
from src.tts import TextToSpeech
from src.actions.dispatch import dispatch as dispatch_non_file_action, NON_FILE_ACTIONS
from src.paths import parse_location_speech, DEFAULT_ROOT, speakable_path
from src.logger import get_logger

logger = get_logger("TestPipeline")


def run_test(text: str):
    logger.info(f"--- INICIANDO PRUEBA CON TEXTO: '{text}' ---")

    # 1. Intención: primero el router determinista, igual que en src/main_loop.py.
    action = route(text)
    if action is not None:
        logger.info(f"Resuelta por el router (sin LLM): {action}")
    else:
        action = NLU().parse(text)
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
        if resultado.text:
            TextToSpeech().speak(resultado.text)
        return

    # 3. Resolver ubicación. Este script no tiene micrófono, así que NO
    # reproduce el diálogo de "¿en qué carpeta?" de src/main_loop.py: si
    # ruta_base no se pudo resolver, usa Proyectos por defecto directamente
    # (avisando en el log) en vez de preguntar por voz.
    base_dir = parse_location_speech(action.ruta_base)
    if base_dir is None:
        base_dir = DEFAULT_ROOT
        logger.info(f"ruta_base vacía/no reconocida ('{action.ruta_base}'); usando {speakable_path(base_dir)} por defecto "
                    "(en main_loop.py real, acá se preguntaría por voz).")

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
    if result.text:
        tts = TextToSpeech()
        tts.speak(result.text)
    else:
        logger.info("No se generó respuesta para hablar.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text_to_test = " ".join(sys.argv[1:])
    else:
        text_to_test = "Crea una carpeta llamada pruebas"

    run_test(text_to_test)

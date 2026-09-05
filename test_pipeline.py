import sys
from src.nlu import NLU
from src.executor import Executor
from src.tts import TextToSpeech
from src.paths import parse_location_speech, DEFAULT_ROOT, speakable_path
from src.logger import get_logger

logger = get_logger("TestPipeline")


def run_test(text: str):
    logger.info(f"--- INICIANDO PRUEBA CON TEXTO: '{text}' ---")

    # 1. NLU
    nlu = NLU()
    action = nlu.parse(text)
    logger.info(f"Acción parseada: {action}")

    # 2. Resolver ubicación. Este script no tiene micrófono, así que NO
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

    # 3. Executor (para eliminar/mover, esto solo arma la pregunta de
    # confirmación — no ejecuta. Para probar el flujo completo con voz hay
    # que correr main.py de verdad.)
    executor = Executor()
    result = executor.execute(action, base_dir=base_dir, raw_text=text, destino_dir=destino_dir)
    logger.info(f"Respuesta generada: {result.text} (needs_confirmation={result.needs_confirmation})")

    # 4. TTS
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

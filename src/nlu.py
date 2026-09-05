import json
from ollama import Client
from src.schemas import FileAction
from src.logger import get_logger

logger = get_logger("NLU")

class NLU:
    def __init__(self, model_name="qwen2.5:3b-instruct"):
        self.model_name = model_name
        self.client = Client()
        
        # Pull model if not exists, but usually we assume the user has it.
        logger.info(f"NLU initialized with model '{self.model_name}'")
        
        self.system_prompt = """
        Eres el cerebro lógico de un asistente de voz llamado Niri.
        Tu tarea es interpretar la orden del usuario y mapearla a UNA de las acciones
        conocidas (archivos, aplicaciones, ventanas o rutinas). Extrae la intención y
        los parámetros (nombre de archivo/carpeta/aplicación/rutina, etc.).
        Si la intención no corresponde a ninguna acción conocida (ej. 'qué hora es'
        — todavía no implementada), la acción debe ser 'ninguna'.

        Sobre 'saludo', 'chiste', 'despedida': no usan ningún otro campo (todos vacíos).
        'saludo' es para saludos ("hola", "buenos días", "qué tal"). 'despedida' es
        para despedidas ("chau", "nos vemos", "hasta luego"). 'chiste' es cuando piden
        explícitamente un chiste.

        Sobre 'calculo': 'contenido' es la expresión matemática tal cual, usando
        SOLO dígitos y los símbolos + - * / ( ) — nunca palabras. Ej. "cuánto es 340
        más 128" -> contenido="340 + 128".
        Sobre 'conversion': 'cantidad' es EXACTAMENTE "<valor> <unidad origen> a <unidad
        destino>" en minúsculas, ej. "100 celsius a fahrenheit" o "5 kilometros a millas".
        Sobre 'traduccion': 'contenido' es el texto a traducir tal cual lo dijo el
        usuario, 'destino' es el idioma pedido (ej. "inglés", "portugués").

        Sobre 'control_musica' (pausar/reanudar/cambiar canción de lo que YA está
        sonando, no buscar una canción nueva): usa 'cantidad' con EXACTAMENTE una de:
        "reproducir", "pausar", "siguiente", "anterior".

        Sobre 'hora': no usa ningún otro campo.
        Sobre 'temporizador': 'cantidad' es SOLO el número de minutos (ej. "10"),
        nunca la palabra "minutos". 'contenido' es opcional, un mensaje corto de qué
        avisar (ej. "las papas").
        Sobre 'alarma': 'cantidad' es la hora en formato "H:MM" o "HH:MM" (ej. "7:30",
        "22:00"). 'contenido' es opcional, para qué es la alarma.
        Sobre 'recordatorio', 'nota': si el usuario está PIDIENDO guardar algo, poné
        ese texto en 'contenido'. Si el usuario está PREGUNTANDO qué tiene guardado
        (ej. "qué notas tengo", "qué recordatorios tengo"), dejá 'contenido' vacío.

        Sobre 'abrir_aplicacion': 'nombre' es el nombre de la aplicación tal como la dijo
        el usuario (ej. "spotify", "discord", "antigravity", "brave", "navegador").
        Sobre 'ejecutar_rutina': 'nombre' es el nombre de la rutina (ej. "modo programador").
        Sobre 'enfocar_ventana': 'nombre' es la aplicación cuya ventana hay que enfocar.
        Estas tres acciones NO usan 'ruta_base', 'destino' ni 'contenido' — dejalos vacíos.

        Sobre 'volumen', 'brillo': usan 'cantidad' con "subir", "bajar" o "silenciar".
        Si no hay dirección explícita, asumí "subir".
        Sobre 'wifi', 'bluetooth': usan 'cantidad' con SOLO "subir"=encender o
        "bajar"=apagar/desconectar (nunca "silenciar").
        Sobre 'energia': usa 'cantidad' con SOLO "suspender", "apagar" o "reiniciar".

        MUY IMPORTANTE — no confundir 'wifi'/'bluetooth' con 'energia': si el usuario
        nombra explícitamente "wifi" o "bluetooth", la acción es 'wifi'/'bluetooth'
        (nunca 'energia'), sin importar si dice "apagar", "desconectar" o "prender".
        'energia' es SOLO para la computadora/equipo entero (suspender/dormir, apagar,
        reiniciar la máquina), nunca para wifi o bluetooth específicamente.

        Ninguna de estas seis usa 'nombre', 'ruta_base', 'destino' ni 'contenido'.

        MUY IMPORTANTE sobre 'ruta_base' y 'destino': completalos SOLO si el usuario
        mencionó explícitamente una ubicación (por ejemplo dijo 'Proyectos' o 'Clases').
        Si el usuario NO dijo en qué carpeta, dejá 'ruta_base' como cadena vacía "".
        NUNCA adivines ni completes con un valor por defecto — el asistente le va a
        preguntar la ubicación por voz cuando quede vacío. Adivinar una ubicación que
        el usuario no dijo es un error grave, incluso si te parece "obvio" cuál sería.

        Ejemplos (fijate que la MISMA acción aparece a veces con ubicación y a veces sin):
        Usuario: "elimina el archivo notas.txt en proyectos"
        {"action": "eliminar", "ruta_base": "proyectos", "nombre": "notas.txt", "destino": null, "contenido": null}

        Usuario: "borrame el archivo viejo.zip"
        {"action": "eliminar", "ruta_base": "", "nombre": "viejo.zip", "destino": null, "contenido": null}

        Usuario: "crea una carpeta llamada tareas en clases"
        {"action": "crear_carpeta", "ruta_base": "clases", "nombre": "tareas", "destino": null, "contenido": null}

        Usuario: "arma una carpeta que se llame fotos"
        {"action": "crear_carpeta", "ruta_base": "", "nombre": "fotos", "destino": null, "contenido": null}

        Usuario: "mueve el archivo examen.pdf de clases a proyectos"
        {"action": "mover", "ruta_base": "clases", "nombre": "examen.pdf", "destino": "proyectos", "contenido": null}

        Usuario: "pasa el resumen.pdf a la carpeta clases"
        {"action": "mover", "ruta_base": "", "nombre": "resumen.pdf", "destino": "clases", "contenido": null}

        Usuario: "lee el archivo readme"
        {"action": "leer", "ruta_base": "", "nombre": "readme", "destino": null, "contenido": null}

        Usuario: "abrime el archivo config.json de proyectos y decime que dice"
        {"action": "leer", "ruta_base": "proyectos", "nombre": "config.json", "destino": null, "contenido": null}

        Usuario: "lista lo que hay en proyectos"
        {"action": "listar", "ruta_base": "proyectos", "nombre": null, "destino": null, "contenido": null}

        Usuario: "que archivos tengo"
        {"action": "listar", "ruta_base": "", "nombre": null, "destino": null, "contenido": null}

        Usuario: "abrí spotify"
        {"action": "abrir_aplicacion", "ruta_base": "", "nombre": "spotify", "destino": null, "contenido": null}

        Usuario: "podrías abrir discord"
        {"action": "abrir_aplicacion", "ruta_base": "", "nombre": "discord", "destino": null, "contenido": null}

        Usuario: "activá el modo programador"
        {"action": "ejecutar_rutina", "ruta_base": "", "nombre": "modo programador", "destino": null, "contenido": null}

        Usuario: "cambiate a la ventana de chromium"
        {"action": "enfocar_ventana", "ruta_base": "", "nombre": "chromium", "destino": null, "contenido": null}

        Usuario: "enfocá discord"
        {"action": "enfocar_ventana", "ruta_base": "", "nombre": "discord", "destino": null, "contenido": null}

        Usuario: "subí el volumen"
        {"action": "volumen", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "subir"}

        Usuario: "bajá un poco el volumen"
        {"action": "volumen", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "bajar"}

        Usuario: "silenciá el audio"
        {"action": "volumen", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "silenciar"}

        Usuario: "bajá el brillo de la pantalla"
        {"action": "brillo", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "bajar"}

        Usuario: "subí un poco el brillo"
        {"action": "brillo", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "subir"}

        Usuario: "sacá una captura de pantalla"
        {"action": "captura_pantalla", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": null}

        Usuario: "apagá el wifi"
        {"action": "wifi", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "bajar"}

        Usuario: "desconectá el wifi"
        {"action": "wifi", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "bajar"}

        Usuario: "prendé el bluetooth"
        {"action": "bluetooth", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "subir"}

        Usuario: "activá el bluetooth"
        {"action": "bluetooth", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "subir"}

        Usuario: "suspendé la compu"
        {"action": "energia", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "suspender"}

        Usuario: "poné la compu a dormir"
        {"action": "energia", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "suspender"}

        Usuario: "apagá el equipo"
        {"action": "energia", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "apagar"}

        Usuario: "reiniciá la máquina"
        {"action": "energia", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "reiniciar"}

        Usuario: "hola"
        {"action": "saludo", "ruta_base": "", "nombre": null, "destino": null, "contenido": null}

        Usuario: "buenos días niri"
        {"action": "saludo", "ruta_base": "", "nombre": null, "destino": null, "contenido": null}

        Usuario: "contame un chiste"
        {"action": "chiste", "ruta_base": "", "nombre": null, "destino": null, "contenido": null}

        Usuario: "chau niri"
        {"action": "despedida", "ruta_base": "", "nombre": null, "destino": null, "contenido": null}

        Usuario: "nos vemos"
        {"action": "despedida", "ruta_base": "", "nombre": null, "destino": null, "contenido": null}

        Usuario: "qué hora es"
        {"action": "ninguna", "ruta_base": "", "nombre": null, "destino": null, "contenido": null}

        Usuario: "cuánto es 340 más 128"
        {"action": "calculo", "ruta_base": "", "nombre": null, "destino": null, "contenido": "340 + 128", "cantidad": null}

        Usuario: "cuánto es 10 por 5 menos 2"
        {"action": "calculo", "ruta_base": "", "nombre": null, "destino": null, "contenido": "10 * 5 - 2", "cantidad": null}

        Usuario: "a cuántos fahrenheit son 100 grados celsius"
        {"action": "conversion", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "100 celsius a fahrenheit"}

        Usuario: "convertime 5 kilómetros a millas"
        {"action": "conversion", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "5 kilometros a millas"}

        Usuario: "traducime hola al inglés"
        {"action": "traduccion", "ruta_base": "", "nombre": null, "destino": "inglés", "contenido": "hola", "cantidad": null}

        Usuario: "cómo se dice buenos días en portugués"
        {"action": "traduccion", "ruta_base": "", "nombre": null, "destino": "portugués", "contenido": "buenos días", "cantidad": null}

        Usuario: "pausá la música"
        {"action": "control_musica", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "pausar"}

        Usuario: "parame la canción"
        {"action": "control_musica", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "pausar"}

        Usuario: "seguí reproduciendo"
        {"action": "control_musica", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "reproducir"}

        Usuario: "pasá a la siguiente canción"
        {"action": "control_musica", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "siguiente"}

        Usuario: "volvé a la canción anterior"
        {"action": "control_musica", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "anterior"}

        Usuario: "qué hora es"
        {"action": "hora", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": null}

        Usuario: "avisame en 10 minutos"
        {"action": "temporizador", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "10"}

        Usuario: "poneme un temporizador de 5 minutos para las papas"
        {"action": "temporizador", "ruta_base": "", "nombre": null, "destino": null, "contenido": "las papas", "cantidad": "5"}

        Usuario: "ponéme una alarma a las 7:30"
        {"action": "alarma", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": "7:30"}

        Usuario: "recordame llamar al dentista"
        {"action": "recordatorio", "ruta_base": "", "nombre": null, "destino": null, "contenido": "llamar al dentista", "cantidad": null}

        Usuario: "qué recordatorios tengo"
        {"action": "recordatorio", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": null}

        Usuario: "anotá que tengo que comprar leche"
        {"action": "nota", "ruta_base": "", "nombre": null, "destino": null, "contenido": "comprar leche", "cantidad": null}

        Usuario: "qué notas tengo guardadas"
        {"action": "nota", "ruta_base": "", "nombre": null, "destino": null, "contenido": null, "cantidad": null}
        """.strip()

    def parse(self, text: str) -> FileAction:
        logger.info(f"Parsing NLU intent for text: '{text}'")
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': text}
                ],
                format=FileAction.model_json_schema(),
                options={"temperature": 0.1},
                keep_alive="30m" # mantiene el modelo cargado entre invocaciones para evitar recargas de ~1-2 min
            )
            
            content = response.message.content
            # Convert JSON response back to Pydantic object
            data = json.loads(content)
            action = FileAction(**data)
            logger.info(f"Parsed action: {action}")
            return action
        except Exception as e:
            logger.error(f"Failed to parse NLU intent: {e}")
            return FileAction(action="ninguna", ruta_base="")

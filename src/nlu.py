import json
import os
import unicodedata
from typing import Optional

from ollama import Client
from src.schemas import FileAction
from src.logger import get_logger

logger = get_logger("NLU")

# Ollama reporta duraciones en nanosegundos; el resto del proyecto razona en ms.
_NS_POR_MS = 1_000_000

# Ventana de contexto explícita, en vez de depender del default del servidor.
#
# Importa porque el modo de falla es silencioso: si el prompt no entra, Ollama lo
# trunca POR EL PRINCIPIO y responde igual —sin error y sin log—, así que la única
# señal sería que el asistente entiende peor sin motivo aparente. Con el default de
# 4096 el turno llegó a consumir 3859 tokens (6% de margen); incluso con el prompt
# ya compactado (~3240) quedaba en 21%, debajo del mínimo de abajo.
#
# Costo medido de subir a 5120: la huella en VRAM pasó de 2127 a 2249 MiB (+122,
# más que la KV sola porque Ollama agranda también el buffer de cómputo), y quedan
# ~1847 MiB de los 4096 para el escritorio.
NUM_CTX = int(os.getenv("NLU_NUM_CTX", "5120"))

# Margen mínimo de ventana libre antes de avisar. 25% deja lugar para una orden
# larga y para crecer el prompt sin quedar al borde del truncado.
MARGEN_MINIMO = 0.25

# Aparatos que, si el usuario los nombra, hacen imposible que la acción sea
# 'energia': "apagá el bluetooth" no apaga la computadora.
#
# Esta corrección es determinista a propósito. El prompt ya advierte el caso en
# mayúsculas Y tiene ejemplos de los dos lados, y qwen2.5:3b se equivoca igual —
# medido en el banco de evaluación. Como la acción equivocada desemboca en
# `systemctl poweroff`, la defensa no puede quedar en manos del modelo. La
# confirmación por voz de src/actions/dispatch.py es la segunda barrera; esta es
# la primera, y evita además tener que contestarle "no" a una pregunta absurda.
_APARATOS_NO_ENERGIA = (("bluetooth", "bluetooth"), ("wifi", "wifi"), ("wi fi", "wifi"))

# Vocabulario canónico de 'cantidad' para wifi y bluetooth. El esquema dice
# "subir"=encender y "bajar"=apagar, pero el modelo devuelve a veces "apagar".
# src/actions/system.py ya tolera esas formas; canonizarlas acá evita que cada
# consumidor futuro tenga que repetir la misma tolerancia. El orden importa:
# "desconect" tiene que evaluarse antes que "conect".
_ENCENDIDO_A_CANTIDAD = (
    ("apag", "bajar"), ("desconect", "bajar"), ("desactiv", "bajar"),
    ("prend", "subir"), ("encend", "subir"), ("activ", "subir"), ("conect", "subir"),
)


def _normalizar(texto: str) -> str:
    """minúsculas, sin acentos, sin guiones (para que "wi-fi" case con "wi fi")."""
    texto = (texto or "").lower().replace("-", " ")
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))


def _canonizar_encendido(cantidad: Optional[str]) -> Optional[str]:
    """
    Lleva la 'cantidad' de wifi/bluetooth al par canónico subir/bajar.

    Si no reconoce la forma, devuelve la original sin tocar: prefiere dejarla
    pasar tal cual —src/actions/system.py la vuelve a normalizar— antes que
    inventar una dirección que el usuario no pidió.
    """
    normalizada = _normalizar(cantidad)
    if normalizada in ("subir", "bajar"):
        return normalizada
    for aguja, canonica in _ENCENDIDO_A_CANTIDAD:
        if aguja in normalizada:
            return canonica
    return cantidad


def corregir_intencion(texto: str, action: FileAction) -> FileAction:
    """
    Corrige errores de clasificación conocidos y verificados del modelo.

    Solo actúa sobre casos donde la acción equivocada tiene consecuencias reales
    y la regla es inequívoca: hoy, que nombrar el wifi o el bluetooth descarta
    'energia'. No es un router genérico ni adivina intenciones; ante cualquier
    otra cosa devuelve la acción tal como vino.
    """
    if action.action == "energia":
        normalizado = _normalizar(texto)
        for aguja, accion_correcta in _APARATOS_NO_ENERGIA:
            if aguja in normalizado:
                logger.warning(
                    f"Corrección determinista: el NLU devolvió 'energia' para una orden que "
                    f"nombra '{aguja}'. Se reinterpreta como '{accion_correcta}' "
                    f"(cantidad={action.cantidad!r})."
                )
                action = action.model_copy(update={"action": accion_correcta})
                break

    if action.action in ("wifi", "bluetooth"):
        canonica = _canonizar_encendido(action.cantidad)
        if canonica not in ("subir", "bajar"):
            # El modelo no dejó una dirección utilizable. Sacarla de lo que el
            # usuario realmente dijo, en vez de dejar que src/actions/system.py
            # caiga en su default "subir" y termine PRENDIENDO lo que se pidió
            # apagar. Si el texto tampoco la tiene, se deja como vino y el
            # handler decide.
            desde_texto = _canonizar_encendido(texto)
            if desde_texto in ("subir", "bajar"):
                canonica = desde_texto
        if canonica != action.cantidad:
            action = action.model_copy(update={"cantidad": canonica})

    return action


def _extract_metrics(response) -> dict:
    """
    Extrae tokens y tiempos de una respuesta de Ollama.

    Sirve para dos cosas distintas: alimentar logs/metrics.jsonl en producción y
    comparar corridas del banco de evaluación. Nunca incluye texto —ni el prompt
    ni la transcripción— porque su destino es un archivo que se puede leer y
    compartir sin exponer lo que dijo el usuario.
    """
    try:
        d = response.model_dump()
    except Exception:  # respuesta con otra forma (cliente distinto, mock, etc.)
        return {}
    return {
        "prompt_tokens": d.get("prompt_eval_count"),
        "output_tokens": d.get("eval_count"),
        "prefill_ms": round((d.get("prompt_eval_duration") or 0) / _NS_POR_MS, 1),
        "decode_ms": round((d.get("eval_duration") or 0) / _NS_POR_MS, 1),
        "load_ms": round((d.get("load_duration") or 0) / _NS_POR_MS, 1),
    }

class NLU:
    def __init__(self, model_name="qwen2.5:3b-instruct"):
        self.model_name = model_name
        self.client = Client()
        # Métricas de la última llamada (tokens y tiempos que reporta Ollama).
        # Las consume src/metrics.py y eval/run.py; se sobrescribe en cada parse().
        self.last_metrics: dict = {}
        # El aviso de margen de contexto se emite una sola vez por proceso: es
        # una condición estructural del prompt, no un evento por turno.
        self._margen_avisado = False
        
        # Pull model if not exists, but usually we assume the user has it.
        logger.info(f"NLU initialized with model '{self.model_name}'")
        
        self.system_prompt = """
        Devolvés SOLO las claves que aplican: omití toda clave vacía o nula.

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

        MUY IMPORTANTE — 'crear_carpeta' vs 'crear_archivo': si el usuario dice
        "archivo" o "documento", la acción es 'crear_archivo'. Si dice "carpeta",
        "directorio" o "folder", es 'crear_carpeta'. Nunca las intercambies: crear
        una carpeta cuando pidieron un archivo deja basura en el disco del usuario.

        Sobre 'calculo': 'contenido' es la expresión matemática tal cual, usando
        SOLO dígitos y los símbolos + - * / ( ) — nunca palabras. Ej. "cuánto es 340
        más 128" -> contenido="340 + 128".
        Sobre 'conversion': 'cantidad' es EXACTAMENTE "<valor> <unidad origen> a <unidad
        destino>" en minúsculas, ej. "100 celsius a fahrenheit" o "5 kilometros a millas".
        'conversion' vs 'calculo': si aparecen UNIDADES (grados, celsius, fahrenheit,
        kilometros, millas, kilos, libras, metros), es 'conversion' aunque la frase
        empiece con "a cuántos" o "cuánto es". 'calculo' es solo aritmética sin unidades.
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

        Sobre 'buscar' (encontrar un archivo del que el usuario NO recuerda el
        nombre exacto: "dónde dejé...", "buscá el archivo de...", "encontrame lo
        de..."): 'contenido' es lo que describe al archivo, sin las muletillas.
        No confundir con 'listar' (mostrar lo que hay en una carpeta) ni con
        'leer' (abrir un archivo cuyo nombre exacto SÍ dijo el usuario).

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

        Sobre 'control_musica' cuando piden "apagar" la música: eso es 'pausar'
        (cantidad="pausar"), nunca 'energia' ni cantidad="apagar".

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
        {"action": "eliminar", "ruta_base": "proyectos", "nombre": "notas.txt"}

        Usuario: "borrame el archivo viejo.zip"
        {"action": "eliminar", "ruta_base": "", "nombre": "viejo.zip"}

        Usuario: "crea una carpeta llamada tareas en clases"
        {"action": "crear_carpeta", "ruta_base": "clases", "nombre": "tareas"}

        Usuario: "arma una carpeta que se llame fotos"
        {"action": "crear_carpeta", "ruta_base": "", "nombre": "fotos"}

        Usuario: "crea un archivo llamado notas.txt en proyectos"
        {"action": "crear_archivo", "ruta_base": "proyectos", "nombre": "notas.txt"}

        Usuario: "creame un archivo que se llame lista.md"
        {"action": "crear_archivo", "ruta_base": "", "nombre": "lista.md"}

        Usuario: "crea un archivo llamado pendientes.txt en clases que diga estudiar para el final"
        {"action": "crear_archivo", "ruta_base": "clases", "nombre": "pendientes.txt", "contenido": "estudiar para el final"}

        Usuario: "mueve el archivo examen.pdf de clases a proyectos"
        {"action": "mover", "ruta_base": "clases", "nombre": "examen.pdf", "destino": "proyectos"}

        Usuario: "pasa el resumen.pdf a la carpeta clases"
        {"action": "mover", "ruta_base": "", "nombre": "resumen.pdf", "destino": "clases"}

        Usuario: "donde deje el resumen de sistemas operativos"
        {"action": "buscar", "ruta_base": "", "contenido": "resumen de sistemas operativos"}

        Usuario: "buscame el archivo del presupuesto"
        {"action": "buscar", "ruta_base": "", "contenido": "presupuesto"}

        Usuario: "lee el archivo readme"
        {"action": "leer", "ruta_base": "", "nombre": "readme"}

        Usuario: "abrime el archivo config.json de proyectos y decime que dice"
        {"action": "leer", "ruta_base": "proyectos", "nombre": "config.json"}

        Usuario: "lista lo que hay en proyectos"
        {"action": "listar", "ruta_base": "proyectos"}

        Usuario: "que archivos tengo"
        {"action": "listar", "ruta_base": ""}

        Usuario: "abrí spotify"
        {"action": "abrir_aplicacion", "nombre": "spotify"}

        Usuario: "podrías abrir discord"
        {"action": "abrir_aplicacion", "nombre": "discord"}

        Usuario: "activá el modo programador"
        {"action": "ejecutar_rutina", "nombre": "modo programador"}

        Usuario: "cambiate a la ventana de chromium"
        {"action": "enfocar_ventana", "nombre": "chromium"}

        Usuario: "enfocá discord"
        {"action": "enfocar_ventana", "nombre": "discord"}

        Usuario: "subí el volumen"
        {"action": "volumen", "cantidad": "subir"}

        Usuario: "bajá un poco el volumen"
        {"action": "volumen", "cantidad": "bajar"}

        Usuario: "silenciá el audio"
        {"action": "volumen", "cantidad": "silenciar"}

        Usuario: "bajá el brillo de la pantalla"
        {"action": "brillo", "cantidad": "bajar"}

        Usuario: "subí un poco el brillo"
        {"action": "brillo", "cantidad": "subir"}

        Usuario: "sacá una captura de pantalla"
        {"action": "captura_pantalla"}

        Usuario: "apagá el wifi"
        {"action": "wifi", "cantidad": "bajar"}

        Usuario: "desconectá el wifi"
        {"action": "wifi", "cantidad": "bajar"}

        Usuario: "prendé el bluetooth"
        {"action": "bluetooth", "cantidad": "subir"}

        Usuario: "activá el bluetooth"
        {"action": "bluetooth", "cantidad": "subir"}

        Usuario: "apagá el bluetooth"
        {"action": "bluetooth", "cantidad": "bajar"}

        Usuario: "suspendé la compu"
        {"action": "energia", "cantidad": "suspender"}

        Usuario: "poné la compu a dormir"
        {"action": "energia", "cantidad": "suspender"}

        Usuario: "apagá el equipo"
        {"action": "energia", "cantidad": "apagar"}

        Usuario: "reiniciá la máquina"
        {"action": "energia", "cantidad": "reiniciar"}

        Usuario: "hola"
        {"action": "saludo"}

        Usuario: "buenos días niri"
        {"action": "saludo"}

        Usuario: "contame un chiste"
        {"action": "chiste"}

        Usuario: "chau niri"
        {"action": "despedida"}

        Usuario: "nos vemos"
        {"action": "despedida"}

        Usuario: "qué hora es"
        {"action": "ninguna"}

        Usuario: "cuánto es 340 más 128"
        {"action": "calculo", "contenido": "340 + 128"}

        Usuario: "cuánto es 10 por 5 menos 2"
        {"action": "calculo", "contenido": "10 * 5 - 2"}

        Usuario: "a cuántos fahrenheit son 100 grados celsius"
        {"action": "conversion", "cantidad": "100 celsius a fahrenheit"}

        Usuario: "convertime 5 kilómetros a millas"
        {"action": "conversion", "cantidad": "5 kilometros a millas"}

        Usuario: "traducime hola al inglés"
        {"action": "traduccion", "destino": "inglés", "contenido": "hola"}

        Usuario: "cómo se dice buenos días en portugués"
        {"action": "traduccion", "destino": "portugués", "contenido": "buenos días"}

        Usuario: "pausá la música"
        {"action": "control_musica", "cantidad": "pausar"}

        Usuario: "parame la canción"
        {"action": "control_musica", "cantidad": "pausar"}

        Usuario: "seguí reproduciendo"
        {"action": "control_musica", "cantidad": "reproducir"}

        Usuario: "pasá a la siguiente canción"
        {"action": "control_musica", "cantidad": "siguiente"}

        Usuario: "volvé a la canción anterior"
        {"action": "control_musica", "cantidad": "anterior"}

        Usuario: "apagá la música"
        {"action": "control_musica", "cantidad": "pausar"}

        Usuario: "qué hora es"
        {"action": "hora"}

        Usuario: "avisame en 10 minutos"
        {"action": "temporizador", "cantidad": "10"}

        Usuario: "poneme un temporizador de 5 minutos para las papas"
        {"action": "temporizador", "contenido": "las papas", "cantidad": "5"}

        Usuario: "ponéme una alarma a las 7:30"
        {"action": "alarma", "cantidad": "7:30"}

        Usuario: "recordame llamar al dentista"
        {"action": "recordatorio", "contenido": "llamar al dentista"}

        Usuario: "qué recordatorios tengo"
        {"action": "recordatorio"}

        Usuario: "anotá que tengo que comprar leche"
        {"action": "nota", "contenido": "comprar leche"}

        Usuario: "qué notas tengo guardadas"
        {"action": "nota"}
        """.strip()

    def _vigilar_margen_de_contexto(self) -> None:
        """
        Avisa si el prompt está por comerse la ventana de contexto.

        Existe porque el modo de falla es silencioso: Ollama trunca el prompt por
        el principio y responde igual, así que la única señal sería que el
        asistente empieza a entender peor sin motivo aparente.
        """
        if self._margen_avisado:
            return
        usados = self.last_metrics.get("prompt_tokens") or 0
        if not usados:
            return
        libre = 1 - (usados / NUM_CTX)
        if libre < MARGEN_MINIMO:
            self._margen_avisado = True
            logger.warning(
                f"Margen de contexto bajo: el prompt usa {usados} de {NUM_CTX} tokens "
                f"({libre:.0%} libre, mínimo {MARGEN_MINIMO:.0%}). Si crece más, Ollama "
                "va a truncarlo en silencio y la comprensión va a empeorar sin error visible."
            )
        else:
            self._margen_avisado = True
            logger.info(f"Contexto: prompt {usados}/{NUM_CTX} tokens ({libre:.0%} libre).")

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
                # temperature=0 (greedy) en vez de 0.1: esto es clasificación
                # contra un enum cerrado, no generación creativa. Con 0.1 medimos
                # que "a cuántos fahrenheit son 100 grados celsius" alternaba
                # entre 'conversion' y 'calculo' en corridas idénticas; con 0 la
                # misma orden da siempre la misma acción, que es lo que hace
                # reproducible al banco de evaluación y auditable al asistente.
                options={"temperature": 0.0, "num_ctx": NUM_CTX},
                keep_alive="30m" # mantiene el modelo cargado entre invocaciones para evitar recargas de ~1-2 min
            )
            
            self.last_metrics = _extract_metrics(response)
            self._vigilar_margen_de_contexto()

            content = response.message.content
            # Convert JSON response back to Pydantic object
            data = json.loads(content)
            action = corregir_intencion(text, FileAction(**data))
            logger.info(f"Parsed action: {action}")
            return action
        except Exception as e:
            logger.error(f"Failed to parse NLU intent: {e}")
            self.last_metrics = {}
            return FileAction(action="ninguna", ruta_base="")

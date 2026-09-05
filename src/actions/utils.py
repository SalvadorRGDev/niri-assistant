"""
Utilidades: cálculo, conversión de unidades, traducción.

'calculo' se evalúa con un parser propio basado en `ast` (whitelist de
operadores aritméticos) — NUNCA con `eval()` sobre texto que viene del
usuario/modelo, mismo principio de seguridad que el resto del proyecto.
'conversion' es un mapa fijo de unidades comunes, determinístico.
'traduccion' es la primera acción que usa el LLM para generar texto libre
en vez de solo clasificar — mismo modelo Ollama que usa el NLU.
"""
import ast
import operator
import re

from ollama import Client

from src.schemas import FileAction, ExecutionResult
from src.logger import get_logger

logger = get_logger("Actions.Utils")

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Expresión no permitida: {ast.dump(node)}")


def calculo(action: FileAction) -> ExecutionResult:
    expr = (action.contenido or "").strip()
    if not expr:
        return ExecutionResult(text="¿Qué cálculo querés que haga?")
    try:
        tree = ast.parse(expr, mode="eval")
        resultado = _safe_eval(tree.body)
        # Enteros sin ".0" cuando el resultado es exacto, para que suene natural al hablarlo.
        if isinstance(resultado, float) and resultado.is_integer():
            resultado = int(resultado)
        logger.info(f"Cálculo: {expr!r} = {resultado}")
        return ExecutionResult(text=f"{expr} es {resultado}.")
    except Exception as e:
        logger.warning(f"No pude evaluar la expresión {expr!r}: {e}")
        return ExecutionResult(text="No pude resolver ese cálculo.")


# (unidad_origen, unidad_destino) -> función de conversión
_CONVERSIONES = {
    ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
    ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
    ("celsius", "kelvin"): lambda v: v + 273.15,
    ("kelvin", "celsius"): lambda v: v - 273.15,
    ("kilometros", "millas"): lambda v: v * 0.621371,
    ("millas", "kilometros"): lambda v: v / 0.621371,
    ("metros", "pies"): lambda v: v * 3.28084,
    ("pies", "metros"): lambda v: v / 3.28084,
    ("kilos", "libras"): lambda v: v * 2.20462,
    ("libras", "kilos"): lambda v: v / 2.20462,
}

_UNIDAD_ALIASES = {
    "celsius": "celsius", "grados celsius": "celsius", "centigrados": "celsius",
    "fahrenheit": "fahrenheit", "grados fahrenheit": "fahrenheit",
    "kelvin": "kelvin",
    "kilometros": "kilometros", "km": "kilometros",
    "millas": "millas",
    "metros": "metros", "m": "metros",
    "pies": "pies",
    "kilos": "kilos", "kilogramos": "kilos", "kg": "kilos",
    "libras": "libras",
}

# "100 celsius a fahrenheit" / "5 kilometros a millas"
_CONVERSION_RE = re.compile(
    r"([\d.,]+)\s*([a-záéíóúñ ]+?)\s+a\s+([a-záéíóúñ ]+)$", re.IGNORECASE
)


def conversion(action: FileAction) -> ExecutionResult:
    texto = (action.cantidad or "").strip().lower()
    match = _CONVERSION_RE.match(texto)
    if not match:
        return ExecutionResult(text="Decime el valor y las unidades, por ejemplo '100 celsius a fahrenheit'.")

    valor_str, origen_raw, destino_raw = match.groups()
    origen = _UNIDAD_ALIASES.get(origen_raw.strip())
    destino = _UNIDAD_ALIASES.get(destino_raw.strip())
    convertir = _CONVERSIONES.get((origen, destino)) if origen and destino else None

    if convertir is None:
        return ExecutionResult(text=f"No sé convertir de '{origen_raw.strip()}' a '{destino_raw.strip()}'.")

    try:
        valor = float(valor_str.replace(",", "."))
        resultado = convertir(valor)
        if resultado == int(resultado):
            resultado = int(resultado)
        else:
            resultado = round(resultado, 2)
        logger.info(f"Conversión: {valor} {origen} -> {resultado} {destino}")
        return ExecutionResult(text=f"{valor_str} {origen_raw.strip()} son {resultado} {destino_raw.strip()}.")
    except Exception as e:
        logger.warning(f"Error en conversión {texto!r}: {e}")
        return ExecutionResult(text="No pude hacer esa conversión.")


_TRANSLATE_SYSTEM_PROMPT = (
    "Traducí el texto del usuario al idioma pedido. Respondé ÚNICAMENTE con la "
    "traducción, sin explicaciones ni comillas."
)


def traduccion(action: FileAction) -> ExecutionResult:
    texto = (action.contenido or "").strip()
    idioma = (action.destino or "").strip()
    if not texto or not idioma:
        return ExecutionResult(text="Decime qué texto traducir y a qué idioma.")

    try:
        client = Client()
        response = client.chat(
            model="qwen2.5:3b-instruct",
            messages=[
                {"role": "system", "content": _TRANSLATE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Traducí a {idioma}: {texto}"},
            ],
            options={"temperature": 0.3},
            keep_alive=0,
        )
        traduccion_texto = response.message.content.strip()
        logger.info(f"Traducción ({idioma}): {texto!r} -> {traduccion_texto!r}")
        return ExecutionResult(text=traduccion_texto)
    except Exception as e:
        logger.error(f"Error traduciendo: {e}")
        return ExecutionResult(text="No pude traducir eso.")

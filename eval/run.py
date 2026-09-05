"""
Banco de evaluación offline del NLU.

Corre las órdenes de eval/casos.jsonl contra el modelo real y reporta cuántas
resuelve bien, con qué latencia y cuántos tokens gasta. No usa micrófono ni
grabaciones: las órdenes son texto escrito a mano, así que se puede correr con
niri.service levantado y sin exponer nada de lo que el usuario dijo de verdad.

Existe porque una optimización puede verse como una mejora limpia y estar
rompiendo la comprensión: acortar el system prompt bajó la latencia un 60% y
cambió 'brillo' por 'volumen' y 'ruta_base' por 'destino'. Sin este banco, esos
dos errores —uno de ellos apunta un 'mover' a otra carpeta— no se ven.

Uso:
    .venv/bin/python eval/run.py                      # corre y compara con el baseline
    .venv/bin/python eval/run.py --guardar-baseline   # fija la corrida como referencia
    .venv/bin/python eval/run.py --comparar eval/resultados/2026-09-05_1200.json
    .venv/bin/python eval/run.py --filtro par_confundible

Código de salida: 1 si hay regresiones contra la referencia, 0 si no.
"""
import argparse
import datetime
import json
import statistics
import sys
import time
import unicodedata
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.nlu import NLU  # noqa: E402

CASOS_PATH = BASE_DIR / "eval" / "casos.jsonl"
RESULTADOS_DIR = BASE_DIR / "eval" / "resultados"
BASELINE_PATH = RESULTADOS_DIR / "baseline.json"

# Campos del FileAction que el banco sabe verificar. 'action' se compara siempre;
# el resto solo cuando el caso lo declara en 'esperado' (así un caso puede exigir
# "ruta_base vacío" sin tener que fijar también nombre, contenido y cantidad).
CAMPOS_SLOT = ("ruta_base", "nombre", "destino", "contenido", "cantidad")

VERDE, ROJO, AMARILLO, GRIS, FIN = "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[0m"


def normalizar(valor) -> str:
    """minúsculas, sin acentos, sin puntuación de borde, espacios colapsados."""
    if valor is None:
        return ""
    texto = str(valor).strip().lower()
    texto = "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))
    return " ".join(texto.strip(" .,;:¡!¿?").split())


def evaluar_caso(esperado: dict, obtenido: dict) -> tuple[bool, bool, list[str]]:
    """Devuelve (acción correcta, slots correctos, lista de diferencias)."""
    accion_ok = normalizar(obtenido.get("action")) == normalizar(esperado["action"])
    diferencias = []
    if not accion_ok:
        diferencias.append(f"action: esperaba {esperado['action']!r}, dio {obtenido.get('action')!r}")

    slots_ok = True
    for campo in CAMPOS_SLOT:
        if campo not in esperado:
            continue
        esp, obt = normalizar(esperado[campo]), normalizar(obtenido.get(campo))
        if esp != obt:
            slots_ok = False
            diferencias.append(f"{campo}: esperaba {esp!r}, dio {obt!r}")
    return accion_ok, slots_ok, diferencias


def cargar_casos(filtro: str | None) -> list[dict]:
    casos = []
    for linea in CASOS_PATH.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("//"):
            continue
        caso = json.loads(linea)
        if filtro and filtro not in caso.get("etiquetas", []):
            continue
        casos.append(caso)
    return casos


def percentil(valores: list[float], p: float) -> float:
    """Percentil por interpolación lineal; statistics.quantiles necesita n>=2."""
    if not valores:
        return 0.0
    orden = sorted(valores)
    if len(orden) == 1:
        return orden[0]
    pos = (len(orden) - 1) * p
    bajo, alto = int(pos), min(int(pos) + 1, len(orden) - 1)
    return orden[bajo] + (orden[alto] - orden[bajo]) * (pos - bajo)


def correr(casos: list[dict], verboso: bool) -> dict:
    nlu = NLU()

    # Dos llamadas de calentamiento: la primera carga los pesos a VRAM, la
    # segunda deja caliente la cache de prefijo del system prompt. Sin esto, los
    # dos primeros casos del banco cargarían con varios segundos que no son
    # suyos y ensuciarían la mediana.
    t0 = time.perf_counter()
    nlu.parse("hola")
    frio_s = time.perf_counter() - t0
    metricas_frio = dict(nlu.last_metrics)
    nlu.parse("hola")

    resultados = []
    for caso in casos:
        t0 = time.perf_counter()
        accion = nlu.parse(caso["texto"])
        ms = (time.perf_counter() - t0) * 1000
        obtenido = accion.model_dump()
        accion_ok, slots_ok, diferencias = evaluar_caso(caso["esperado"], obtenido)

        resultados.append({
            "id": caso["id"],
            "texto": caso["texto"],
            "etiquetas": caso.get("etiquetas", []),
            "accion_ok": accion_ok,
            "slots_ok": slots_ok,
            "ok": accion_ok and slots_ok,
            "diferencias": diferencias,
            "obtenido": {k: v for k, v in obtenido.items() if v not in (None, "")},
            "ms": round(ms, 1),
            "tokens_salida": nlu.last_metrics.get("output_tokens"),
            "tokens_prompt": nlu.last_metrics.get("prompt_tokens"),
        })

        if verboso or not (accion_ok and slots_ok):
            marca = f"{VERDE}OK  {FIN}" if accion_ok and slots_ok else (
                f"{AMARILLO}SLOT{FIN}" if accion_ok else f"{ROJO}MAL {FIN}")
            print(f"  {marca} {caso['id']:34} {ms:6.0f} ms  {caso['texto']!r}")
            for d in diferencias:
                print(f"       {GRIS}{d}{FIN}")

    latencias = [r["ms"] for r in resultados]
    salidas = [r["tokens_salida"] for r in resultados if r["tokens_salida"]]
    return {
        "fecha": datetime.datetime.now().isoformat(timespec="seconds"),
        "modelo": nlu.model_name,
        "total": len(resultados),
        "accion_ok": sum(r["accion_ok"] for r in resultados),
        "slots_ok": sum(r["ok"] for r in resultados),
        "latencia_p50_ms": round(percentil(latencias, 0.50), 1),
        "latencia_p95_ms": round(percentil(latencias, 0.95), 1),
        "tokens_prompt": resultados[0]["tokens_prompt"] if resultados else None,
        "tokens_salida_mediana": round(statistics.median(salidas), 1) if salidas else None,
        "arranque_frio_s": round(frio_s, 2),
        "arranque_frio_detalle": metricas_frio,
        "casos": resultados,
    }


def resumen_por_etiqueta(informe: dict) -> dict:
    por_etiqueta: dict[str, list[bool]] = {}
    for caso in informe["casos"]:
        for etiqueta in caso["etiquetas"]:
            por_etiqueta.setdefault(etiqueta, []).append(caso["ok"])
    return {e: (sum(v), len(v)) for e, v in sorted(por_etiqueta.items())}


def comparar(informe: dict, referencia: dict) -> int:
    previos = {c["id"]: c for c in referencia["casos"]}
    regresiones, arreglos = [], []
    for caso in informe["casos"]:
        anterior = previos.get(caso["id"])
        if anterior is None:
            continue
        if anterior["ok"] and not caso["ok"]:
            regresiones.append(caso)
        elif not anterior["ok"] and caso["ok"]:
            arreglos.append(caso)

    print(f"\n{GRIS}vs {referencia['fecha']} "
          f"({referencia['slots_ok']}/{referencia['total']}, "
          f"p50 {referencia['latencia_p50_ms']:.0f} ms){FIN}")
    delta = informe["latencia_p50_ms"] - referencia["latencia_p50_ms"]
    signo = "+" if delta >= 0 else ""
    print(f"  latencia p50: {informe['latencia_p50_ms']:.0f} ms ({signo}{delta:.0f} ms)")
    for caso in arreglos:
        print(f"  {VERDE}arreglado{FIN} {caso['id']}")
    for caso in regresiones:
        print(f"  {ROJO}REGRESIÓN{FIN} {caso['id']}: {'; '.join(caso['diferencias'])}")
    if not regresiones and not arreglos:
        print(f"  {GRIS}sin cambios de exactitud{FIN}")
    return 1 if regresiones else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Banco de evaluación del NLU de Niri")
    parser.add_argument("--filtro", help="correr solo los casos con esta etiqueta")
    parser.add_argument("--comparar", type=Path, help="archivo de resultados a usar como referencia")
    parser.add_argument("--guardar-baseline", action="store_true",
                        help="además de guardar la corrida, fijarla como referencia")
    parser.add_argument("--verboso", action="store_true", help="listar también los casos que pasan")
    args = parser.parse_args()

    casos = cargar_casos(args.filtro)
    if not casos:
        print("No hay casos que correr (¿filtro sin coincidencias?)", file=sys.stderr)
        return 2

    print(f"Banco de evaluación del NLU — {len(casos)} casos"
          + (f" (filtro: {args.filtro})" if args.filtro else ""))
    informe = correr(casos, args.verboso)

    print(f"\n{'':2}acción correcta      {informe['accion_ok']}/{informe['total']}")
    print(f"{'':2}acción + slots       {informe['slots_ok']}/{informe['total']}")
    print(f"{'':2}latencia p50 / p95   {informe['latencia_p50_ms']:.0f} / {informe['latencia_p95_ms']:.0f} ms")
    print(f"{'':2}tokens prompt        {informe['tokens_prompt']}")
    print(f"{'':2}tokens salida        {informe['tokens_salida_mediana']:.0f} (mediana)")
    print(f"{'':2}arranque en frío     {informe['arranque_frio_s']:.2f} s "
          f"(carga {informe['arranque_frio_detalle'].get('load_ms', 0)/1000:.2f} s)")

    print(f"\n{'':2}por etiqueta:")
    for etiqueta, (ok, total) in resumen_por_etiqueta(informe).items():
        marca = "" if ok == total else f"  {ROJO}<-{FIN}"
        print(f"{'':4}{etiqueta:22} {ok}/{total}{marca}")

    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    sello = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    destino = RESULTADOS_DIR / f"{sello}.json"
    destino.write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{GRIS}resultados: {destino.relative_to(BASE_DIR)}{FIN}")

    if args.guardar_baseline:
        BASELINE_PATH.write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{GRIS}baseline actualizado: {BASELINE_PATH.relative_to(BASE_DIR)}{FIN}")
        return 0

    referencia_path = args.comparar or (BASELINE_PATH if BASELINE_PATH.exists() else None)
    if referencia_path and Path(referencia_path).exists():
        return comparar(informe, json.loads(Path(referencia_path).read_text(encoding="utf-8")))

    print(f"\n{GRIS}Sin referencia previa. Fijá esta corrida con --guardar-baseline.{FIN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

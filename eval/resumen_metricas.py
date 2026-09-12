"""
Resumen de logs/metrics.jsonl: p50 y p95 por etapa del pipeline de voz.

Responde de un vistazo "¿en qué se va el tiempo de un turno?" con datos de uso
real, en vez de mediciones manuales que envejecen. Solo lee: no modifica ni rota
el archivo (de eso se encarga src/metrics.py).

Uso:
    .venv/bin/python eval/resumen_metricas.py
    .venv/bin/python eval/resumen_metricas.py --ultimos 50
    .venv/bin/python eval/resumen_metricas.py --accion eliminar
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.metrics import METRICS_PATH  # noqa: E402

# Orden real del pipeline, para que la tabla se lea como el turno sucede.
ETAPAS = [("vad", "espera + VAD"), ("stt", "transcripción"),
          ("router", "router"), ("nlu", "intención (LLM)"), ("tts", "voz")]


def percentil(valores, p):
    if not valores:
        return 0.0
    orden = sorted(valores)
    if len(orden) == 1:
        return orden[0]
    pos = (len(orden) - 1) * p
    bajo, alto = int(pos), min(int(pos) + 1, len(orden) - 1)
    return orden[bajo] + (orden[alto] - orden[bajo]) * (pos - bajo)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumen de métricas de latencia de Niri")
    parser.add_argument("--ultimos", type=int, help="usar solo los N turnos más recientes")
    parser.add_argument("--accion", help="filtrar por nombre de acción")
    parser.add_argument("--archivo", default=METRICS_PATH, help="ruta de metrics.jsonl")
    args = parser.parse_args()

    ruta = Path(args.archivo)
    if not ruta.exists():
        print(f"Todavía no hay métricas en {ruta}.\n"
              "Se generan solas: cada turno de voz agrega una línea.")
        return 0

    turnos = []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea:
            continue
        try:
            turnos.append(json.loads(linea))
        except json.JSONDecodeError:
            continue  # línea a medio escribir por un corte: se ignora, no se corrige

    if args.accion:
        turnos = [t for t in turnos if t.get("accion") == args.accion]
    if args.ultimos:
        turnos = turnos[-args.ultimos:]

    if not turnos:
        print("No hay turnos que resumir con esos filtros.")
        return 0

    print(f"{len(turnos)} turnos · {turnos[0]['timestamp']} → {turnos[-1]['timestamp']}\n")
    print(f"  {'etapa':16} {'n':>4} {'p50':>9} {'p95':>9}")
    print(f"  {'-' * 40}")

    por_etapa = defaultdict(list)
    for turno in turnos:
        for clave, valor in turno.items():
            if clave.endswith("_ms"):
                por_etapa[clave[:-3]].append(valor)

    for clave, etiqueta in ETAPAS:
        valores = por_etapa.get(clave, [])
        if not valores:
            continue
        print(f"  {etiqueta:16} {len(valores):>4} {percentil(valores, .50):>7.0f} ms {percentil(valores, .95):>7.0f} ms")

    total = por_etapa.get("total", [])
    print(f"  {'-' * 40}")
    print(f"  {'turno completo':16} {len(total):>4} {percentil(total, .50):>7.0f} ms {percentil(total, .95):>7.0f} ms")

    salidas = [t["tokens_salida"] for t in turnos if t.get("tokens_salida")]
    if salidas:
        print(f"\n  tokens de salida del NLU: p50 {percentil(salidas, .50):.0f} · p95 {percentil(salidas, .95):.0f}")

    # 'falso_positivo' histórico no incluía capturas sin voz. Mostrar ambos
    # casos sin cambiar los registros viejos ni convertir sospechas en verdad.
    motivos = Counter()
    for turno in turnos:
        motivo = turno.get("motivo_sin_orden")
        if motivo:
            motivos[motivo] += 1
        elif not turno.get("accion"):
            motivos["historico_sin_accion"] += 1
    if motivos:
        sin_orden = sum(motivos.values())
        print(f"\n  turnos sin acción: {sin_orden}/{len(turnos)} ({sin_orden / len(turnos):.0%})")
        for motivo, cantidad in sorted(motivos.items()):
            print(f"    {motivo:26} {cantidad:>4}")
        print("  No equivale a falsos positivos confirmados ni a una tasa por hora.")

    resueltos = Counter(t.get("origen") for t in turnos if t.get("origen") and t.get("accion"))
    if resueltos:
        con_orden = sum(resueltos.values())
        por_router = resueltos.get("router", 0)
        print(f"\n  resueltos sin LLM: {por_router}/{con_orden} ({por_router / con_orden:.0%})")

    acciones = Counter(t.get("accion") for t in turnos if t.get("accion"))
    if acciones:
        print("\n  acciones más frecuentes:")
        for accion, veces in acciones.most_common(10):
            print(f"    {accion:22} {veces:>4}  ({veces / len(turnos) * 100:.0f}%)")
        vacios = sum(1 for t in turnos if not t.get("accion"))
        if vacios:
            print(f"    {'(sin acción)':22} {vacios:>4}  ({vacios / len(turnos) * 100:.0f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

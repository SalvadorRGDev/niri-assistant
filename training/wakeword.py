"""Entrenamiento local reproducible; nunca sustituye el modelo de producción.

Comandos y límites de las métricas están documentados en README.md.
Las etiquetas deducidas de un directorio son provisionales. El modo independiente
exige revisión y sesiones separadas; --exploratory permite investigar sin afirmar
generalización cuando los datos disponibles no alcanzan.
"""
import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import re
import sys
import time
import wave
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from eval.calidad_audio import tiene_voz  # noqa: E402

SAMPLE_RATE = 16000
CHUNK = 1280
WINDOWS = 16
FEATURES = 96
FORMAT_VERSION = 1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path, obj):
    """Crear exclusivamente: una corrida no puede alterar evidencia anterior."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def audio_read(path):
    with wave.open(str(path), "rb") as handle:
        if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth(),
                handle.getcomptype()) != (SAMPLE_RATE, 1, 2, "NONE"):
            raise ValueError(f"WAV incompatible (se exige PCM mono 16 kHz/16 bit): {path}")
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    if not frames or len(raw) != frames * 2:
        raise ValueError(f"WAV vacío o truncado: {path}")
    return np.frombuffer(raw, dtype="<i2")


def resolve_recording(entry):
    path = (BASE_DIR / entry["path"]).resolve()
    if not path.is_relative_to((BASE_DIR / "recordings").resolve()):
        raise ValueError("El manifiesto solo puede referenciar recordings/.")
    return path


def manifest_create(output):
    sources = (("wakeword_positivos", 1, "prompted_positive"),
               ("falsos_positivos", 0, "automatic_trigger"),
               ("wakeword_negativos_continuos", 0, "continuous_candidate"))
    entries = []
    for folder, label, source in sources:
        for path in sorted((BASE_DIR / "recordings" / folder).glob("*.wav")):
            audio = audio_read(path)
            valid, reason = tiene_voz(audio) if label else (True, "")
            date = re.search(r"(?:^|_)(20\d{6})_", path.stem)
            # Un día completo queda unido aunque cambie el directorio o micrófono.
            # Sin metadatos, inventar sesiones por clip produciría fuga de datos.
            session = f"day-{date.group(1)}" if date else "unknown-session"
            entries.append({
                "path": str(path.relative_to(BASE_DIR)), "sha256": sha256(path),
                "pcm_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                "duration_seconds": len(audio) / SAMPLE_RATE,
                "label": label, "label_reviewed": False,
                "source": source, "session": session, "session_reviewed": False,
                "device": "unknown", "continuous_negative": False,
                "usable": bool(valid), "exclusion_reason": reason,
            })
    if not entries:
        raise ValueError("No hay grabaciones de wake word.")
    doc = {"format_version": FORMAT_VERSION,
           "notes": "Etiquetas provisionales; revisar escuchando localmente. "
                    "Las sesiones por día son una agrupación conservadora inferida.",
           "entries": entries}
    json_write(output, doc)
    return {"manifest": str(output), "clips": len(entries),
            "usable": sum(e["usable"] for e in entries),
            "reviewed_labels": 0,
            "sessions_per_label": sessions_summary([e for e in entries if e["usable"]])}


def manifest_read(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("format_version") != FORMAT_VERSION:
        raise ValueError("Versión de manifiesto incompatible.")
    if not isinstance(doc.get("entries"), list) or not doc["entries"]:
        raise ValueError("El manifiesto no contiene clips.")
    for entry in doc["entries"]:
        if type(entry.get("label")) is not int or entry["label"] not in (0, 1):
            raise ValueError("Etiqueta inválida: debe ser 0 o 1.")
        if not isinstance(entry.get("session"), str) or not entry["session"]:
            raise ValueError("Cada clip requiere una sesión.")
        for field in ("usable", "label_reviewed", "session_reviewed", "continuous_negative"):
            if type(entry.get(field)) is not bool:
                raise ValueError(f"Se exige un booleano real en {field}.")
        audio = audio_read(resolve_recording(entry))
        if sha256(resolve_recording(entry)) != entry["sha256"]:
            raise ValueError("El contenido de un WAV cambió respecto al manifiesto.")
        if hashlib.sha256(audio.tobytes()).hexdigest() != entry["pcm_sha256"]:
            raise ValueError("El hash PCM no corresponde al WAV.")
        if not math.isclose(entry["duration_seconds"], len(audio) / SAMPLE_RATE):
            raise ValueError("La duración del manifiesto no corresponde al WAV.")
        if entry["continuous_negative"] and (entry["label"] != 0 or
                                            not entry["label_reviewed"] or
                                            entry["source"] == "automatic_trigger"):
            raise ValueError("Falsos disparos no son negativos continuos revisados.")
    return doc


def manifest_merge(paths, output):
    """Unir manifiestos revisados sin reetiquetar ni dividir sesiones."""
    entries = {}
    for path in paths:
        for entry in manifest_read(path)["entries"]:
            key = entry["path"]
            if key in entries and entries[key] != entry:
                raise ValueError("Dos manifiestos discrepan sobre el mismo clip; revisar antes de unir.")
            entries[key] = entry
    doc = {"format_version": FORMAT_VERSION,
           "notes": "Unión de manifiestos; conserva etiquetas, revisión y sesiones originales.",
           "entries": [entries[key] for key in sorted(entries)]}
    json_write(output, doc)
    return {"manifest": str(output), "clips": len(entries)}


def sessions_summary(entries):
    return {str(label): len({e["session"] for e in entries if e["label"] == label})
            for label in (0, 1)}


def distinct_entries(entries):
    """Deduplicar PCM y unir sesiones que comparten una copia del mismo audio."""
    parent = {e["session"]: e["session"] for e in entries}

    def root(value):
        while parent[value] != value:
            value = parent[value]
        return value

    by_pcm = {}
    for entry in entries:
        digest = entry["pcm_sha256"]
        if digest in by_pcm:
            previous = by_pcm[digest]
            if previous["label"] != entry["label"]:
                raise ValueError("Un mismo audio tiene etiquetas contradictorias.")
            a, b = sorted((root(previous["session"]), root(entry["session"])))
            parent[b] = a
        else:
            by_pcm[digest] = entry
    # Guardar las sesiones originales permite detectar copias en evaluaciones futuras.
    return [dict(e, group=root(e["session"])) for e in by_pcm.values()]


def require_reviewed(entries):
    if any(not e["label_reviewed"] or not e["session_reviewed"] or
           e.get("device", "unknown") == "unknown" for e in entries):
        raise ValueError("El modo independiente exige etiquetas, sesiones y micrófono "
                         "revisados. Use --exploratory solo para diagnóstico provisional.")


def split_groups(entries, seed=42):
    """Fijar train/validation/test por grupo, con ambas clases en cada partición."""
    groups = sorted({e["group"] for e in entries})
    for label in (0, 1):
        if len({e["group"] for e in entries if e["label"] == label}) < 3:
            raise ValueError("Faltan al menos tres sesiones independientes por clase "
                             "para entrenamiento, validación y prueba.")
    rng = np.random.default_rng(seed)
    held = max(1, round(len(groups) * 0.2))
    for _ in range(1000):
        shuffled = rng.permutation(groups).tolist()
        roles = {g: ("test" if i < held else "validation" if i < 2 * held else "train")
                 for i, g in enumerate(shuffled)}
        split = {role: [e for e in entries if roles[e["group"]] == role]
                 for role in ("train", "validation", "test")}
        if all({e["label"] for e in samples} == {0, 1} for samples in split.values()):
            return split
    raise ValueError("No se pudo construir una división por sesiones con ambas clases; "
                     "hacen falta más sesiones de ambas clases.")


def padded_chunks(audio, continuous=False):
    if not continuous:
        # El stream real continúa después de decir la palabra. El relleno permite
        # ver la última ventana de clips cortos; nunca cuenta como tiempo ambiente.
        audio = np.pad(audio, (SAMPLE_RATE, SAMPLE_RATE))
    for start in range(0, len(audio), CHUNK):
        chunk = audio[start:start + CHUNK]
        if len(chunk) < CHUNK and not continuous:
            chunk = np.pad(chunk, (0, CHUNK - len(chunk)))
        yield chunk


def extract_windows(extractor, entry):
    extractor.reset()
    embeddings = []
    for chunk in padded_chunks(audio_read(resolve_recording(entry)),
                               entry["continuous_negative"]):
        for feature in extractor.process_streaming(chunk.tobytes()):
            embeddings.extend(feature.reshape(-1, FEATURES))
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if len(embeddings) < WINDOWS:
        raise ValueError("El extractor no produjo una ventana completa.")
    return np.asarray([embeddings[i:i + WINDOWS].reshape(-1)
                       for i in range(len(embeddings) - WINDOWS + 1)], dtype=np.float64)


def train_mil(bags, labels, seed=42, l2=0.05, max_iterations=250):
    """Regresión logística MIL: la etiqueta pertenece al clip, no a su silencio.

    Un soft-max de ventanas localiza evidencia de la palabra sin inventar marcas
    temporales. L2 y normalización se ajustan únicamente con entrenamiento.
    """
    sizes = np.asarray([len(b) for b in bags])
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    matrix = np.concatenate(bags)
    mean = matrix.mean(axis=0)
    scale = np.maximum(matrix.std(axis=0), 0.1)
    matrix = (matrix - mean) / scale
    labels = np.asarray(labels, dtype=float)
    counts = Counter(labels)
    clip_weights = np.array([1 / (2 * counts[y]) for y in labels])
    temperature = 0.5

    def objective(params):
        logits = matrix @ params[:-1] + params[-1]
        row_gradient = np.empty(len(matrix))
        loss = 0.5 * l2 * np.dot(params[:-1], params[:-1])
        for start, size, label, weight in zip(starts, sizes, labels, clip_weights):
            segment = logits[start:start + size] / temperature
            normalizer = logsumexp(segment)
            pooled = temperature * (normalizer - np.log(size))
            loss += weight * (np.logaddexp(0, pooled) - label * pooled)
            row_gradient[start:start + size] = (
                weight * (expit(pooled) - label) * np.exp(segment - normalizer))
        gradient = np.concatenate((matrix.T @ row_gradient + l2 * params[:-1],
                                   [row_gradient.sum()]))
        return float(loss), gradient

    rng = np.random.default_rng(seed)
    initial = rng.normal(0, 0.001, matrix.shape[1] + 1)
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      options={"maxiter": max_iterations, "ftol": 1e-10})
    if not result.success:
        raise ValueError(f"El optimizador no convergió: {result.message}")
    weights = result.x[:-1] / scale
    bias = result.x[-1] - mean @ weights
    return weights.astype(np.float32), np.float32(bias), {
        "algorithm": "logistic_multiple_instance_softmax", "l2": l2,
        "temperature": temperature, "seed": seed, "iterations": int(result.nit),
        "training_loss": float(result.fun), "windows": int(len(matrix)),
        "parameters": int(len(result.x)), "normalization_fit": "train_only",
    }


def export_onnx(path, weights, bias):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = helper.make_graph([
        helper.make_node("Flatten", ["embeddings"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "weights", "bias"], ["logit"]),
        helper.make_node("Sigmoid", ["logit"], ["probability"]),
    ], "oye_niri_candidate", [helper.make_tensor_value_info(
        "embeddings", TensorProto.FLOAT, [None, WINDOWS, FEATURES])],
        [helper.make_tensor_value_info("probability", TensorProto.FLOAT, [None, 1])],
        initializer=[numpy_helper.from_array(weights.reshape(-1, 1), "weights"),
                     numpy_helper.from_array(np.asarray([bias], dtype=np.float32), "bias")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                              producer_name="niri-local-training", ir_version=8)
    onnx.checker.check_model(model)
    with Path(path).open("xb") as handle:
        handle.write(model.SerializeToString())


def longest_run(scores, threshold):
    best = current = 0
    for score in scores:
        current = current + 1 if score >= threshold else 0
        best = max(best, current)
    return best


def calibrate(bags, labels, weights, bias):
    scores = [expit(bag @ weights + bias) for bag in bags]
    candidates = []
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        for hits in (2, 3, 4, 5):
            triggered = [longest_run(s, threshold) >= hits for s in scores]
            tp = sum(flag for flag, y in zip(triggered, labels) if y == 1)
            fp = sum(flag for flag, y in zip(triggered, labels) if y == 0)
            # Rechazar falsos en validación tiene prioridad; el test no participa.
            candidates.append((-fp, tp, threshold, hits))
    _, _, threshold, hits = max(candidates)
    return {"threshold": threshold, "trigger_level": hits,
            "selected_on": "validation_only", "grid_candidates": len(candidates)}


def proportion(successes, count):
    if not count:
        return {"count": 0, "detected": 0, "rate": None, "wilson_95": None}
    p, z = successes / count, 1.959963984540054
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return {"count": count, "detected": successes, "rate": p,
            "wilson_95": [max(0.0, center - radius), min(1.0, center + radius)]}


def summarize_records(records):
    positive = [r for r in records if r["label"] == 1]
    negative = [r for r in records if r["label"] == 0]
    continuous = [r for r in negative if r["continuous_negative"] and r["label_reviewed"]]
    seconds = sum(r["duration_seconds"] for r in continuous)
    events = sum(r["events"] for r in continuous)
    return {
        "positive_recall": proportion(sum(r["events"] > 0 for r in positive), len(positive)),
        "negative_clip_activation": proportion(sum(r["events"] > 0 for r in negative), len(negative)),
        "continuous_negative_seconds": seconds,
        "continuous_false_activations": events,
        "false_activations_per_hour": events / (seconds / 3600) if seconds else None,
        # Cero observaciones nunca demuestra tasa cero: cota Poisson unilateral 95%.
        "zero_event_upper_95_per_hour": -math.log(0.05) / (seconds / 3600)
        if seconds and events == 0 else None,
        "sessions": len({r["session"] for r in records}),
        "all_labels_reviewed": all(r["label_reviewed"] for r in records),
        "by_device": {device: {
            "clips": sum(r["device"] == device for r in records),
            "positive_recall": proportion(
                sum(r["events"] > 0 for r in positive if r["device"] == device),
                sum(r["device"] == device for r in positive)),
            "negative_clip_activation": proportion(
                sum(r["events"] > 0 for r in negative if r["device"] == device),
                sum(r["device"] == device for r in negative)),
            "continuous_negative_seconds": sum(r["duration_seconds"] for r in continuous if r["device"] == device),
            "continuous_false_activations": sum(r["events"] for r in continuous if r["device"] == device),
        } for device in sorted({r["device"] for r in records})},
    }


def streaming_evaluate(model, entries, threshold, trigger_level):
    """Ejecutar el detector real, incluidos sus resets después de cada disparo."""
    from src.wake_word import WakeWordDetector, logger

    detector = WakeWordDetector(str(model), threshold=threshold, trigger_level=trigger_level)
    rows = []
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        for entry in entries:
            detector.oww_features.reset()
            detector.oww_model.reset()
            detector._consecutive_hits = 0
            events = sum(bool(detector.feed(chunk)) for chunk in padded_chunks(
                audio_read(resolve_recording(entry)), entry["continuous_negative"]))
            rows.append({key: entry[key] for key in (
                "sha256", "pcm_sha256", "label", "label_reviewed", "session", "device",
                "duration_seconds", "continuous_negative") } | {"events": events})
    finally:
        detector.oww_features.close()
        logger.setLevel(previous_level)
    return {"summary": summarize_records(rows), "clips": rows,
            "threshold": threshold, "trigger_level": trigger_level,
            "model_sha256": sha256(model), "runtime": "WakeWordDetector.feed",
            "clip_padding_seconds_each_side": 1, "continuous_padding_seconds": 0}


def environment(extractor=None):
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": {name: importlib.metadata.version(name) for name in
                           ("numpy", "scipy", "onnx", "onnxruntime", "pyopen-wakeword")},
              "source_sha256": sha256(Path(__file__)), "sample_rate": SAMPLE_RATE,
              "chunk_samples": CHUNK, "input_shape": [1, WINDOWS, FEATURES]}
    if extractor is not None:
        result["feature_models_sha256"] = {"mel": sha256(extractor.mel_path),
                                            "embedding": sha256(extractor.emb_path)}
    return result


def train(args):
    import onnxruntime as ort
    from pyopen_wakeword import OpenWakeWordFeatures
    from src.config import WAKE_WORD_MODEL, WAKE_WORD_THRESHOLD, WAKE_WORD_TRIGGER_LEVEL

    started = time.monotonic()
    manifest = manifest_read(args.manifest)
    usable = [e for e in manifest["entries"] if e["usable"]]
    entries = distinct_entries(usable)
    if {e["label"] for e in entries} != {0, 1}:
        raise ValueError("El entrenamiento exige positivos y negativos utilizables.")
    if args.exploratory:
        split = {"train": entries, "validation": [], "test": []}
    else:
        require_reviewed(usable)
        split = split_groups(entries, args.seed)
    output = Path(args.output).resolve()
    if not output.is_relative_to((BASE_DIR / "data" / "wakeword").resolve()):
        raise ValueError("Las corridas deben guardarse bajo data/wakeword/ (datos privados).")
    output.mkdir(parents=True, exist_ok=False)
    json_write(output / "manifest.json", manifest)
    with (output / "training_source.py").open("xb") as handle:
        handle.write(Path(__file__).read_bytes())
    extractor = OpenWakeWordFeatures.from_builtin()
    try:
        env = environment(extractor)
        bags = {e["pcm_sha256"]: extract_windows(extractor, e) for e in
                split["train"] + split["validation"]}
    finally:
        extractor.close()
    weights, bias, fit = train_mil([bags[e["pcm_sha256"]] for e in split["train"]],
                                   [e["label"] for e in split["train"]],
                                   args.seed, args.l2, args.max_iterations)
    candidate = output / "candidate.onnx"
    export_onnx(candidate, weights, bias)
    # Verificar exportación con ventanas reales sin usar el conjunto de prueba.
    sample = bags[split["train"][0]["pcm_sha256"]][:8].astype(np.float32)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(candidate), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    actual = session.run(None, {"embeddings": sample.reshape(-1, WINDOWS, FEATURES)})[0].ravel()
    expected = expit(sample @ weights + bias)
    if not np.allclose(actual, expected, atol=1e-5, rtol=1e-5):
        raise ValueError("La exportación ONNX no coincide con el clasificador entrenado.")
    settings = ({"threshold": WAKE_WORD_THRESHOLD, "trigger_level": WAKE_WORD_TRIGGER_LEVEL,
                 "selected_on": "production_defaults_no_calibration"} if args.exploratory else
                calibrate([bags[e["pcm_sha256"]] for e in split["validation"]],
                          [e["label"] for e in split["validation"]], weights, bias))
    evaluation = entries if args.exploratory else split["test"]
    report = {
        "format_version": FORMAT_VERSION,
        "mode": "exploratory_training_fit" if args.exploratory else "independent_test",
        "promotion_allowed": False,
        "limitations": ["No se modifica ni promociona el modelo de producción.",
                        "Las métricas por clip no son falsos positivos por hora.",
                        "Los intervalos por clip no modelan dependencia dentro de una sesión.",
                        "Se exige validación en sesiones y micrófonos nuevos antes de adoptar."] +
                       (["Etiquetas provisionales: la carpeta no demuestra qué contiene el audio.",
                         "Evaluación sobre entrenamiento: no mide generalización."] if args.exploratory else []),
        "manifest_sha256": sha256(args.manifest), "environment": env,
        "dataset": {"total": len(manifest["entries"]), "usable": len(usable),
                    "deduplicated": len(entries), "sessions_per_label": sessions_summary(entries),
                    "excluded": len(manifest["entries"]) - len(usable)},
        "split": {role: [{key: e[key] for key in ("pcm_sha256", "session", "group", "label")}
                         for e in samples] for role, samples in split.items()},
        "all_original_sessions": sorted({e["session"] for e in usable}),
        "training": fit, "candidate_settings": settings,
        "onnx_parity_max_absolute_error": float(np.max(np.abs(actual - expected))),
        "candidate": streaming_evaluate(candidate, evaluation, settings["threshold"], settings["trigger_level"]),
        "baseline": streaming_evaluate(WAKE_WORD_MODEL, evaluation, WAKE_WORD_THRESHOLD, WAKE_WORD_TRIGGER_LEVEL),
    }
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    json_write(output / "report.json", report)
    return {"report": str(output / "report.json"), "mode": report["mode"],
            "dataset": report["dataset"], "candidate": report["candidate"]["summary"],
            "baseline": report["baseline"]["summary"], "training": fit,
            "elapsed_seconds": report["elapsed_seconds"], "promotion_allowed": False}


def assert_independent(entries, report):
    known_pcm = {e["pcm_sha256"] for samples in report["split"].values() for e in samples}
    known_sessions = set(report["all_original_sessions"])
    if any(e["pcm_sha256"] in known_pcm or e["session"] in known_sessions for e in entries):
        raise ValueError("La evaluación nueva comparte audio o sesiones con la corrida original.")


def evaluate(args):
    from src.config import WAKE_WORD_MODEL, WAKE_WORD_THRESHOLD, WAKE_WORD_TRIGGER_LEVEL

    report = json.loads(Path(args.run_report).read_text(encoding="utf-8"))
    manifest = manifest_read(args.manifest)
    usable = [e for e in manifest["entries"] if e["usable"]]
    require_reviewed(usable)
    entries = distinct_entries(usable)
    if not entries:
        raise ValueError("La evaluación no contiene clips utilizables.")
    assert_independent(usable, report)
    candidate = Path(args.run_report).parent / "candidate.onnx"
    if sha256(candidate) != report["candidate"]["model_sha256"]:
        raise ValueError("El candidato cambió después de su entrenamiento.")
    settings = report["candidate_settings"]
    result = {"mode": "new_sessions_independent_evaluation", "promotion_allowed": False,
              "run_report_sha256": sha256(args.run_report),
              "manifest_sha256": sha256(args.manifest), "environment": environment(),
              "candidate": streaming_evaluate(candidate, entries, settings["threshold"], settings["trigger_level"]),
              "baseline": streaming_evaluate(WAKE_WORD_MODEL, entries, WAKE_WORD_THRESHOLD, WAKE_WORD_TRIGGER_LEVEL)}
    json_write(args.output, result)
    return {"report": str(args.output), "candidate": result["candidate"]["summary"],
            "baseline": result["baseline"]["summary"], "promotion_allowed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest", help="Inventario fijo con hashes y etiquetas provisionales")
    manifest.add_argument("--output", default="data/wakeword/manifest.json")
    merger = commands.add_parser("merge", help="Unir manifiestos de captura y revisión")
    merger.add_argument("--manifest", action="append", required=True)
    merger.add_argument("--output", required=True)
    trainer = commands.add_parser("train", help="Crear un candidato sin sustituir producción")
    trainer.add_argument("--manifest", required=True)
    trainer.add_argument("--output", required=True)
    trainer.add_argument("--exploratory", action="store_true")
    trainer.add_argument("--seed", type=int, default=42)
    trainer.add_argument("--l2", type=float, default=0.05)
    trainer.add_argument("--max-iterations", type=int, default=250)
    evaluator = commands.add_parser("evaluate", help="Evaluar candidato fijo en sesiones nuevas revisadas")
    evaluator.add_argument("--manifest", required=True)
    evaluator.add_argument("--run-report", required=True)
    evaluator.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "train" and (not math.isfinite(args.l2) or args.l2 <= 0 or args.max_iterations < 1):
            raise ValueError("L2 e iteraciones deben ser positivos.")
        result = (manifest_create(args.output) if args.command == "manifest" else
                  manifest_merge(args.manifest, args.output) if args.command == "merge" else
                  train(args) if args.command == "train" else evaluate(args))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, wave.Error, ImportError) as exc:
        print(f"No se completó: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

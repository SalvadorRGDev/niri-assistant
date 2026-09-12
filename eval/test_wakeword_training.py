"""Contratos de separación, tasas y exportación del entrenamiento local."""
import json
import io
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from training.wakeword import (  # noqa: E402
    assert_independent, distinct_entries, export_onnx, json_write, padded_chunks,
    manifest_merge, proportion, require_reviewed, split_groups, summarize_records, train_mil,
)


def entry(label, session, digest):
    return {"label": label, "session": session, "group": session,
            "pcm_sha256": digest, "label_reviewed": True,
            "session_reviewed": True, "device": "internal", "usable": True}


class DatasetTests(unittest.TestCase):
    def test_session_split_disjoint_reproducible_both_classes(self):
        entries = [entry(label, f"session-{i}", f"{i}-{label}")
                   for i in range(6) for label in (0, 1)]
        split = split_groups(entries, 42)
        self.assertEqual(split, split_groups(entries, 42))
        self.assertEqual(sum(map(len, split.values())), len(entries))
        groups = [{e["group"] for e in part} for part in split.values()]
        for i in range(3):
            self.assertEqual({e["label"] for e in list(split.values())[i]}, {0, 1})
            for j in range(i + 1, 3):
                self.assertFalse(groups[i] & groups[j])

    def test_single_positive_session_cannot_be_independent(self):
        entries = [entry(1, "one-day", f"p{i}") for i in range(30)]
        entries += [entry(0, f"day-{i}", f"n{i}") for i in range(6)]
        with self.assertRaisesRegex(ValueError, "tres sesiones"):
            split_groups(entries)

    def test_pcm_duplicates_unite_sessions_before_split(self):
        entries = [entry(1, "a", "same"), entry(1, "b", "same"),
                   entry(0, "b", "other")]
        distinct = distinct_entries(entries)
        self.assertEqual(len(distinct), 2)
        self.assertEqual({e["group"] for e in distinct}, {"a"})

    def test_contradictory_duplicate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "contradictorias"):
            distinct_entries([entry(1, "a", "same"), entry(0, "b", "same")])

    def test_provisional_review_cannot_be_inferred(self):
        record = entry(1, "a", "x")
        record["label_reviewed"] = False
        with self.assertRaisesRegex(ValueError, "revisados"):
            require_reviewed([record])
        record["label_reviewed"] = True
        record["device"] = "unknown"
        with self.assertRaisesRegex(ValueError, "revisados"):
            require_reviewed([record])

    def test_new_evaluation_rejects_original_audio_or_sessions(self):
        report = {"split": {"train": [entry(1, "a", "same")]},
                  "all_original_sessions": ["a", "duplicate-original-session"]}
        for sample in (entry(1, "new", "same"), entry(1, "a", "new"),
                       entry(0, "duplicate-original-session", "different")):
            with self.assertRaisesRegex(ValueError, "comparte"):
                assert_independent([sample], report)
        assert_independent([entry(1, "new", "different")], report)

    def test_existing_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            json_write(path, {"kept": True})
            with self.assertRaises(FileExistsError):
                json_write(path, {"kept": False})
            self.assertEqual(json.loads(path.read_text()), {"kept": True})

    def test_merge_preserves_review_and_rejects_contradictions(self):
        sample = entry(1, "session", "digest") | {"path": "recordings/clip.wav"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            with patch("training.wakeword.manifest_read", return_value={"entries": [sample]}):
                manifest_merge(["first", "duplicate"], output)
            self.assertEqual(json.loads(output.read_text())["entries"], [sample])
            with patch("training.wakeword.manifest_read", side_effect=[
                {"entries": [sample]}, {"entries": [sample | {"label": 0}]}]):
                with self.assertRaisesRegex(ValueError, "discrepan"):
                    manifest_merge(["first", "contradiction"], Path(directory) / "other.json")


class MetricsTests(unittest.TestCase):
    def row(self, continuous=False, reviewed=True, events=2):
        return {"label": 0, "continuous_negative": continuous,
                "label_reviewed": reviewed, "duration_seconds": 600,
                "events": events, "session": "a", "device": "internal"}

    def test_trigger_selected_clips_never_produce_hourly_rate(self):
        report = summarize_records([self.row()])
        self.assertIsNone(report["false_activations_per_hour"])
        self.assertEqual(report["negative_clip_activation"]["detected"], 1)

    def test_hourly_rate_counts_events_on_reviewed_continuous_duration(self):
        report = summarize_records([self.row(True), self.row(False, events=99),
                                    self.row(True, reviewed=False, events=99)])
        self.assertEqual(report["continuous_negative_seconds"], 600)
        self.assertEqual(report["false_activations_per_hour"], 12)

    def test_zero_events_has_nonzero_confidence_bound(self):
        report = summarize_records([self.row(True, events=0)])
        self.assertEqual(report["false_activations_per_hour"], 0)
        self.assertGreater(report["zero_event_upper_95_per_hour"], 17)
        self.assertIsNone(proportion(0, 0)["rate"])
        self.assertGreater(proportion(0, 30)["wilson_95"][1], 0)

    def test_continuous_audio_has_no_artificial_context_padding(self):
        audio = np.ones(32000, dtype=np.int16)
        self.assertEqual(len(list(padded_chunks(audio, True))), 25)
        self.assertEqual(len(list(padded_chunks(audio, False))), 50)
        final = list(padded_chunks(np.ones(1300, dtype=np.int16), True))
        self.assertEqual(len(final), 2)
        self.assertEqual(len(final[-1]), 20)
        self.assertEqual(np.count_nonzero(final[-1]), 20)


class ModelTests(unittest.TestCase):
    def test_mil_learns_clip_evidence_without_labeling_all_silence_positive(self):
        rng = np.random.default_rng(1)
        bags = [rng.normal(0, 0.1, (12, 4)) for _ in range(8)]
        labels = [0, 1] * 4
        for bag, label in zip(bags, labels):
            if label:
                bag[5:8, 0] += 3
        weights, bias, result = train_mil(bags, labels, seed=3)
        scores = [np.max(bag @ weights + bias) for bag in bags]
        self.assertLess(max(scores[::2]), min(scores[1::2]))
        self.assertTrue(result["iterations"] > 0)
        other_weights, other_bias, _ = train_mil(bags, labels, seed=3)
        np.testing.assert_array_equal(weights, other_weights)
        self.assertEqual(bias, other_bias)

    def test_export_matches_probability_and_runtime_window_contract(self):
        from scipy.special import expit
        from src.wake_word import OnnxClassifier

        rng = np.random.default_rng(2)
        weights = rng.normal(0, 0.01, 16 * 96).astype(np.float32)
        bias = np.float32(-0.2)
        embeddings = rng.normal(0, 1, (19, 96)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.onnx"
            export_onnx(path, weights, bias)
            classifier = OnnxClassifier(path)
            self.assertEqual(classifier.input_windows, 16)
            self.assertEqual(list(classifier.process_streaming(embeddings[:15])), [])
            actual = list(classifier.process_streaming(embeddings[15:]))
            expected = [float(expit(embeddings[i:i + 16].reshape(-1) @ weights + bias))
                        for i in range(4)]
            np.testing.assert_allclose(actual, expected, atol=1e-6)
            classifier.reset()
            self.assertEqual(list(classifier.process_streaming(embeddings[:15])), [])


class CaptureTests(unittest.TestCase):
    def test_collection_keeps_labels_unreviewed_and_can_reuse_session(self):
        from eval.grabar_validacion import run
        from training.wakeword import manifest_read
        from unittest.mock import Mock

        stream = SimpleNamespace(
            _abrir=Mock(return_value=True), start=Mock(side_effect=AssertionError("fallback")),
            read_chunk=lambda size: np.ones(size, dtype=np.int16),
            reset_buffers=Mock(), stop=Mock())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind in ("positive", "negative-continuous"):
                output = root / "data" / "wakeword" / f"{kind}.json"
                args = SimpleNamespace(device=2, microphone="internal", session="same-session",
                                       count=1, seconds=0.1, kind=kind, manifest=str(output))
                with patch("eval.grabar_validacion.BASE_DIR", root), \
                        patch("training.wakeword.BASE_DIR", root), \
                        patch("eval.grabar_validacion.check_service_stopped"), \
                        patch("eval.grabar_validacion.explicit_device", return_value="physical (hw:1,0)"), \
                        patch.dict(sys.modules, {"src.audio": SimpleNamespace(AudioStream=lambda **_: stream)}), \
                        patch("builtins.input", return_value=""), redirect_stdout(io.StringIO()):
                    self.assertEqual(run(args), 0)
                    result = manifest_read(output)["entries"][0]
                self.assertFalse(result["label_reviewed"])
                self.assertFalse(result["continuous_negative"])
                self.assertEqual(result["session"], "same-session")
                self.assertEqual(result["device"], "internal")
                self.assertTrue(result["usable"])
            self.assertEqual(stream._abrir.call_count, 2)
            stream.start.assert_not_called()
            self.assertEqual(stream.stop.call_count, 2)

    def test_unknown_service_state_blocks_capture(self):
        from eval.grabar_validacion import check_service_stopped

        for code, state in ((1, ""), (0, "active"), (0, "activating"), (0, "unknown")):
            with patch("eval.grabar_validacion.subprocess.run", return_value=SimpleNamespace(
                    returncode=code, stdout=state)):
                with self.assertRaisesRegex(ValueError, "inactivo"):
                    check_service_stopped()
        with patch("eval.grabar_validacion.subprocess.run", return_value=SimpleNamespace(
                returncode=0, stdout="inactive\n")):
            check_service_stopped()

    def test_capture_exact_duration_preserves_existing_audio(self):
        from eval.grabar_validacion import capture

        stream = SimpleNamespace(read_chunk=lambda size: np.ones(size, dtype=np.int16))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.wav"
            duration, digest = capture(stream, path, 0.1)
            self.assertEqual(duration, 0.1)
            self.assertEqual(len(digest), 64)
            previous = path.read_bytes()
            with self.assertRaises(FileExistsError):
                capture(stream, path, 1)
            self.assertEqual(path.read_bytes(), previous)

    def test_interrupted_capture_keeps_readable_partial_wav(self):
        from eval.grabar_validacion import capture
        from unittest.mock import Mock

        stream = SimpleNamespace(read_chunk=Mock(side_effect=[
            np.ones(1280, dtype=np.int16), KeyboardInterrupt()]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.wav"
            with self.assertRaises(KeyboardInterrupt):
                capture(stream, path, 1)
            with wave.open(str(path), "rb") as handle:
                self.assertEqual(handle.getnframes(), 1280)
                self.assertEqual(handle.getframerate(), 16000)


if __name__ == "__main__":
    unittest.main()

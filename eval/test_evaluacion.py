"""Regresiones de los bancos: un requisito ausente nunca cuenta como éxito.

Uso: .venv/bin/python eval/test_evaluacion.py
Usa dobles: no carga modelos, abre audio ni conecta con Ollama.
"""
import contextlib
import io
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import src
from eval import run, test_stt
from src.schemas import FileAction


class BancosDeEvaluacion(unittest.TestCase):
    def setUp(self):
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        self.ruta = Path(self.temporal.name)
        self.salida = io.StringIO()
        self.enterContext(contextlib.redirect_stdout(self.salida))
        self.enterContext(contextlib.redirect_stderr(self.salida))

    def configurar_nlu(self, argumentos, nlu):
        self.enterContext(patch.object(sys, "argv", ["eval/run.py", *argumentos]))
        self.enterContext(patch.object(run, "BASE_DIR", self.ruta))
        self.enterContext(patch.object(run, "RESULTADOS_DIR", self.ruta / "resultados"))
        self.enterContext(patch.object(run, "BASELINE_PATH", self.ruta / "baseline.json"))
        self.enterContext(patch.object(run, "NLU", return_value=nlu))
        self.enterContext(patch.object(run, "Client"))
        self.enterContext(patch.object(run, "route", return_value=None))
        self.enterContext(patch.object(run, "cargar_casos", return_value=[{
            "id": "negativo", "texto": "orden sintética desconocida", "etiquetas": [],
            "esperado": {"action": "ninguna"},
        }]))

    def test_backend_caido_no_cuenta_negativo_como_acierto(self):
        nlu = SimpleNamespace(model_name="modelo-local", last_metrics={},
                              parse=lambda texto: FileAction(action="ninguna"))
        self.configurar_nlu(["--guardar-baseline"], nlu)
        self.assertEqual(run.main(), 2)
        self.assertIn("Evaluación incompleta", self.salida.getvalue())
        self.assertFalse((self.ruta / "baseline.json").exists())
        self.assertFalse((self.ruta / "resultados").exists())

    def test_perdida_backend_a_mitad_no_guarda_corrida(self):
        nlu = SimpleNamespace(model_name="modelo-local", last_metrics={})
        llamadas = []

        def parse(texto):
            llamadas.append(texto)
            nlu.last_metrics = {"output_tokens": 2} if len(llamadas) < 3 else {}
            return FileAction(action="ninguna")

        nlu.parse = parse
        self.configurar_nlu([], nlu)
        self.assertEqual(run.main(), 2)
        self.assertEqual(len(llamadas), 3)
        self.assertFalse((self.ruta / "resultados").exists())

    def test_tokens_ausentes_se_reportan_sin_error_de_formato(self):
        nlu = SimpleNamespace(model_name="modelo-local",
                              last_metrics={"output_tokens": None, "prompt_tokens": None},
                              parse=lambda texto: FileAction(action="ninguna"))
        self.configurar_nlu([], nlu)
        self.assertEqual(run.main(), 0)
        self.assertIn("no reportados", self.salida.getvalue())

    def test_modelo_exploratorio_no_puede_sobrescribir_baseline(self):
        self.configurar_nlu(["--modelo", "otro", "--guardar-baseline"], Mock())
        with self.assertRaises(SystemExit) as salida:
            run.main()
        self.assertEqual(salida.exception.code, 2)

    def test_comparacion_inexistente_no_aparenta_exito(self):
        self.configurar_nlu(["--comparar", str(self.ruta / "ausente.json")], Mock())
        self.assertEqual(run.main(), 2)

    def configurar_stt(self, referencia=None):
        self.enterContext(patch.object(sys, "argv", ["eval/test_stt.py"]))
        self.enterContext(patch.object(test_stt, "BASELINE", self.ruta / "baseline.json"))
        self.enterContext(patch.object(test_stt, "RECORDINGS", self.ruta))
        transcribir = self.enterContext(patch.object(test_stt, "transcribir"))
        if referencia is not None:
            (self.ruta / "baseline.json").write_text(json.dumps(referencia), encoding="utf-8")
        return transcribir

    def test_baseline_stt_ausente_es_omitido(self):
        transcribir = self.configurar_stt()
        self.assertEqual(test_stt.main(), 2)
        transcribir.assert_not_called()

    def test_baseline_stt_vacio_es_omitido(self):
        transcribir = self.configurar_stt({"hashes": {}})
        self.assertEqual(test_stt.main(), 2)
        transcribir.assert_not_called()

    def test_grabacion_faltante_no_cambia_el_conjunto(self):
        transcribir = self.configurar_stt({"hashes": {"uno.wav": "a", "dos.wav": "b"}})
        (self.ruta / "uno.wav").touch()
        self.assertEqual(test_stt.main(), 2)
        transcribir.assert_not_called()

    def test_excepcion_stt_no_expone_contenido(self):
        transcribir = self.configurar_stt({"hashes": {"uno.wav": "a"}, "audio_s": 1})
        (self.ruta / "uno.wav").touch()
        transcribir.side_effect = RuntimeError("contenido confidencial de voz")
        self.assertEqual(test_stt.main(), 2)
        self.assertNotIn("contenido confidencial", self.salida.getvalue())

    def test_transcribir_silencia_resultados_y_descartes_y_usa_cache(self):
        salida_logger = io.StringIO()
        logger = logging.Logger("stt-doble", level=logging.DEBUG)
        logger.addHandler(logging.StreamHandler(salida_logger))

        class Motor:
            def __init__(self, model_size):
                self.model_size = model_size

            def transcribe(self, audio):
                logger.warning("fragmento descartado confidencial")
                logger.info("resultado confidencial")
                return "resultado confidencial"

        descargar = Mock(return_value=str(self.ruta / "modelo"))
        modulo = SimpleNamespace(logger=logger, SpeechToText=Motor)
        utilidades = SimpleNamespace(download_model=descargar)
        with patch.dict(sys.modules, {"faster_whisper.utils": utilidades}), \
                patch.object(src, "stt", modulo, create=True), \
                patch.object(test_stt, "cargar_audio", return_value=None):
            hashes, tiempos = test_stt.transcribir([self.ruta / "uno.wav"])
        self.assertEqual(salida_logger.getvalue(), "")
        self.assertFalse(logger.disabled)
        self.assertEqual(len(hashes["uno.wav"]), 16)
        self.assertEqual(len(tiempos), 1)
        self.assertTrue(descargar.call_args.kwargs["local_files_only"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

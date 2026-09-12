"""Regresión TTS: no red, sin reproducción ni modelos reales.

Uso: .venv/bin/python eval/test_privacidad.py
"""
import asyncio
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import tts as tts_mod  # noqa: E402


class PrivacidadTTS(unittest.TestCase):
    def setUp(self):
        self.logger_patch = patch.object(tts_mod, "logger")
        self.logger = self.logger_patch.start()
        self.addCleanup(self.logger_patch.stop)
        with patch.object(tts_mod, "_buscar_voz_piper", return_value=None), \
                patch.object(tts_mod.shutil, "which", side_effect=lambda name: name):
            self.tts = tts_mod.TextToSpeech()
        self.tts._speak_piper = Mock(return_value=False)
        self.tts._speak_offline = Mock(return_value=False)
        self.tts._generate_edge_audio = AsyncMock(return_value=False)
        self.tts._play_file = Mock(return_value=True)

    def test_nube_apagada_por_defecto(self):
        with patch.dict(os.environ, {}, clear=True):
            config = runpy.run_path(str(BASE_DIR / "src" / "config.py"))
        self.assertIs(config["ALLOW_CLOUD_TTS_FALLBACK"], False)

    def test_texto_privado_usa_piper(self):
        self.tts._speak_piper.return_value = True
        self.assertTrue(self.tts.speak("Dato de prueba privado"))
        self.tts._speak_offline.assert_not_called()
        self.tts._generate_edge_audio.assert_not_called()

    def test_fallo_piper_sigue_con_espeak(self):
        self.tts._speak_offline.return_value = True
        self.assertTrue(self.tts.speak("Dato de prueba privado"))
        self.tts._speak_offline.assert_called_once()
        self.tts._generate_edge_audio.assert_not_called()

    def test_privado_nunca_va_a_nube_aunque_flag_habilitado(self):
        with patch.object(tts_mod, "ALLOW_CLOUD_TTS_FALLBACK", True):
            self.assertFalse(self.tts.speak("Contenido privado ficticio"))
        self.tts._generate_edge_audio.assert_not_called()

    def test_publico_sin_opt_in_no_va_a_nube(self):
        with patch.object(tts_mod, "ALLOW_CLOUD_TTS_FALLBACK", False):
            self.assertFalse(self.tts.speak("Listo.", public=True))
        self.tts._generate_edge_audio.assert_not_called()

    def test_publico_con_opt_in_prioriza_piper(self):
        self.tts._speak_piper.return_value = True
        with patch.object(tts_mod, "ALLOW_CLOUD_TTS_FALLBACK", True):
            self.assertTrue(self.tts.speak("Listo.", public=True))
        self.tts._generate_edge_audio.assert_not_called()

    def test_publico_con_opt_in_prioriza_espeak(self):
        self.tts._speak_offline.return_value = True
        with patch.object(tts_mod, "ALLOW_CLOUD_TTS_FALLBACK", True):
            self.assertTrue(self.tts.speak("Listo.", public=True))
        self.tts._generate_edge_audio.assert_not_called()

    def test_publico_con_opt_in_y_fallos_locales_usa_nube(self):
        self.tts._generate_edge_audio.return_value = True
        with patch.object(tts_mod, "ALLOW_CLOUD_TTS_FALLBACK", True):
            self.assertTrue(self.tts.speak("Listo.", public=True))
        self.tts._generate_edge_audio.assert_awaited_once()
        text, audio_path = self.tts._generate_edge_audio.call_args.args
        self.assertEqual(text, "Listo.")
        self.tts._play_file.assert_called_once_with(audio_path)
        self.assertFalse(Path(audio_path).exists())

    def test_fallo_nube_limpia_temporal_y_no_reporta_exito(self):
        with patch.object(tts_mod, "ALLOW_CLOUD_TTS_FALLBACK", True):
            self.assertFalse(self.tts.speak("Listo.", public=True))
        audio_path = self.tts._generate_edge_audio.call_args.args[1]
        self.assertFalse(Path(audio_path).exists())
        self.tts._play_file.assert_not_called()

    def test_fallo_reproduccion_nube_no_reporta_exito(self):
        self.tts._generate_edge_audio.return_value = True
        self.tts._play_file.return_value = False
        with patch.object(tts_mod, "ALLOW_CLOUD_TTS_FALLBACK", True):
            self.assertFalse(self.tts.speak("Listo.", public=True))

    def test_sin_reproductor_no_usa_nube_pero_espeak_funciona(self):
        self.tts.player = None
        self.tts._speak_offline.return_value = True
        with patch.object(tts_mod, "ALLOW_CLOUD_TTS_FALLBACK", True):
            self.assertTrue(self.tts.speak("Listo.", public=True))
        self.tts._generate_edge_audio.assert_not_called()

    def test_valor_truthy_no_autoriza_nube(self):
        with patch.object(tts_mod, "ALLOW_CLOUD_TTS_FALLBACK", True):
            for value in ("false", "true", 1, [], None):
                with self.subTest(public=value), self.assertRaises(TypeError):
                    self.tts.speak("Dato privado ficticio", public=value)
        self.tts._generate_edge_audio.assert_not_called()

    def test_texto_vacio_no_invoca_motores(self):
        for text in ("", "   ", None):
            self.assertFalse(self.tts.speak(text))
        self.tts._speak_piper.assert_not_called()
        self.tts._speak_offline.assert_not_called()
        self.tts._generate_edge_audio.assert_not_called()

    def test_logs_tts_no_duplican_contenido_privado(self):
        secret = "contenido_ficticio_no_debe_aparecer_en_logs"
        self.tts.speak(secret)
        self.assertNotIn(secret, str(self.logger.mock_calls))

    def test_temporales_son_privados_unicos_y_se_limpian(self):
        with tts_mod._audio_temporal(".wav") as first:
            with tts_mod._audio_temporal(".wav") as second:
                self.assertNotEqual(first, second)
                self.assertEqual(os.stat(first).st_mode & 0o777, 0o600)
                self.assertEqual(os.stat(second).st_mode & 0o777, 0o600)
            self.assertFalse(Path(second).exists())
        self.assertFalse(Path(first).exists())

    def test_temporal_se_limpia_tambien_con_excepcion(self):
        with self.assertRaises(RuntimeError):
            with tts_mod._audio_temporal(".wav") as audio_path:
                raise RuntimeError("fallo simulado")
        self.assertFalse(Path(audio_path).exists())

    def test_piper_proceso_reproduccion_fallida_devuelve_false(self):
        paths = []

        def synthesize(text, output):
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\x00\x00" * 160)

        def play(path):
            paths.append(path)
            self.assertTrue(Path(path).exists())
            return False

        self.tts.piper_voz = SimpleNamespace(synthesize_wav=synthesize)
        self.tts._play_file.side_effect = play
        self.assertFalse(self.tts._speak_piper_en_proceso("Texto ficticio"))
        self.assertEqual(len(paths), 1)
        self.assertFalse(Path(paths[0]).exists())

    def test_error_piper_no_publica_texto_de_excepcion(self):
        secret = "contenido_privado_ficticio"
        self.tts.piper_voz = SimpleNamespace(synthesize_wav=Mock(side_effect=RuntimeError(secret)))
        with tempfile.TemporaryDirectory() as tmp, patch.object(tempfile, "tempdir", tmp):
            self.assertFalse(self.tts._speak_piper_en_proceso(secret))
            self.assertEqual(list(Path(tmp).iterdir()), [])
        self.assertNotIn(secret, str(self.logger.mock_calls))

    def test_piper_subproceso_vacio_no_reporta_exito(self):
        self.tts.piper_bin = "/piper-ficticio"
        with patch.object(tts_mod.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
            self.assertFalse(self.tts._speak_piper_subproceso("Dato ficticio"))
        self.tts._play_file.assert_not_called()

    def test_piper_subproceso_archivo_y_texto_son_privados(self):
        self.tts.piper_bin = "/piper-ficticio"
        secret = "Dato ficticio"
        paths = []

        def run(argv, **kwargs):
            self.assertNotIn(secret, argv)
            self.assertEqual(kwargs["input"], secret.encode("utf-8"))
            self.assertNotIn("shell", kwargs)
            path = argv[argv.index("--output_file") + 1]
            paths.append(path)
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            Path(path).write_bytes(b"audio ficticio")
            return SimpleNamespace(returncode=0)

        with patch.object(tts_mod.subprocess, "run", side_effect=run):
            self.assertTrue(self.tts._speak_piper_subproceso(secret))
        self.assertFalse(Path(paths[0]).exists())

    def test_espeak_recibe_texto_por_stdin_no_en_argv(self):
        secret = "--opcion Dato privado ficticio"
        with patch.object(tts_mod.subprocess, "run") as run:
            result = tts_mod.TextToSpeech._speak_offline(self.tts, secret)
        self.assertTrue(result)
        self.assertNotIn(secret, run.call_args.args[0])
        self.assertIn("--stdin", run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["input"], secret.encode("utf-8"))

    def test_espeak_fallido_no_publica_excepcion_con_texto(self):
        secret = "contenido_privado_ficticio"
        with patch.object(tts_mod.subprocess, "run", side_effect=RuntimeError(secret)):
            self.assertFalse(tts_mod.TextToSpeech._speak_offline(self.tts, secret))
        self.assertNotIn(secret, str(self.logger.mock_calls))

    def test_reproductor_fallido_devuelve_false(self):
        with tts_mod._audio_temporal(".wav") as path:
            Path(path).write_bytes(b"audio ficticio")
            with patch.object(tts_mod.subprocess, "run", side_effect=subprocess.TimeoutExpired("mpv", 30)):
                self.assertFalse(tts_mod.TextToSpeech._play_file(self.tts, path))

    def test_carga_piper_fallida_conserva_binario_local(self):
        fake_piper = SimpleNamespace(PiperVoice=SimpleNamespace(load=Mock(side_effect=RuntimeError)))
        with patch.dict(sys.modules, {"piper": fake_piper}), \
                patch.object(tts_mod, "_buscar_voz_piper", return_value=Path("voz-ficticia.onnx")), \
                patch.object(tts_mod, "_buscar_binario_piper", return_value="/piper-ficticio"), \
                patch.object(tts_mod.shutil, "which", return_value="mpv"):
            tts = tts_mod.TextToSpeech()
        self.assertIsNone(tts.piper_voz)
        self.assertEqual(tts.piper_bin, "/piper-ficticio")
        self.assertTrue(tts.has_piper)

    def test_fallo_piper_proceso_prueba_binario(self):
        self.tts.has_piper = True
        self.tts.piper_voz = object()
        self.tts.piper_bin = "/piper-ficticio"
        self.tts._speak_piper_en_proceso = Mock(return_value=False)
        self.tts._speak_piper_subproceso = Mock(return_value=True)
        self.assertTrue(tts_mod.TextToSpeech._speak_piper(self.tts, "Texto ficticio"))
        self.tts._speak_piper_subproceso.assert_called_once()

    def test_edge_timeout_es_controlado_sin_red(self):
        fake_edge = SimpleNamespace(Communicate=Mock(return_value=SimpleNamespace(
            save=AsyncMock(side_effect=TimeoutError))))
        with patch.dict(sys.modules, {"edge_tts": fake_edge}), tts_mod._audio_temporal(".mp3") as path:
            result = asyncio.run(tts_mod.TextToSpeech._generate_edge_audio(self.tts, "Listo.", path))
        self.assertFalse(result)
        self.assertFalse(Path(path).exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)

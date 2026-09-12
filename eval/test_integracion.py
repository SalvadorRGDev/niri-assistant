"""Contratos del ejecutor y del diálogo real, con disco temporal y audio simulado.

No carga modelos ni abre PortAudio; usa MainLoop sin su constructor de hardware.
Las operaciones de archivos sí se ejecutan, exclusivamente en TemporaryDirectory.
Uso: .venv/bin/python eval/test_integracion.py
"""
import json
import logging
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import get_args
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

# Importar sounddevice inicializa PortAudio aunque no se capture. Este banco
# prueba el diálogo, no el controlador: el doble se instala ANTES del import.
_sounddevice_original = sys.modules.get("sounddevice")
sys.modules["sounddevice"] = Mock()
from src import main_loop as loop_mod
if _sounddevice_original is None:
    del sys.modules["sounddevice"]
else:
    sys.modules["sounddevice"] = _sounddevice_original
from src import audit as audit_mod, executor as executor_mod, paths as paths_mod
from src.actions.dispatch import _HANDLERS
from src.confirm import interpret_confirmation
from src.schemas import FileAction


class Contratos(unittest.TestCase):
    def setUp(self):
        self.contexto = ExitStack()
        self.addCleanup(self.contexto.close)
        self.tmp = Path(self.contexto.enter_context(tempfile.TemporaryDirectory()))
        self.proyectos, self.clases = self.tmp / "Proyectos", self.tmp / "Clases"
        self.proyectos.mkdir()
        self.clases.mkdir()
        self.contexto.enter_context(patch.object(executor_mod, "ALLOWED_ROOTS", [self.proyectos, self.clases]))
        self.contexto.enter_context(patch.object(paths_mod, "ROOT_ALIASES", [
            ("proyectos", self.proyectos), ("clases", self.clases)]))
        self.contexto.enter_context(patch.object(loop_mod, "DEFAULT_ROOT", self.proyectos))
        self.contexto.enter_context(patch.object(audit_mod, "AUDIT_LOG_PATH", self.tmp / "audit.log"))
        self.executor = executor_mod.Executor()
        self.loop = loop_mod.MainLoop.__new__(loop_mod.MainLoop)
        self.loop.executor = self.executor
        self.loop.turno = None
        self.loop.audio = Mock()
        self.loop.tts = Mock()
        self.loop.tts.speak.return_value = True
        self.loop._listen_reply = Mock(return_value="sí")

    def ejecutar(self, action, **kwargs):
        return self.executor.execute(action, base_dir=self.proyectos, raw_text="prueba escrita", **kwargs)

    def auditoria(self):
        return [json.loads(l) for l in (self.tmp / "audit.log").read_text().splitlines()]

    def test_raices_protegidas_con_y_sin_confirmacion(self):
        for operation in ("eliminar", "mover"):
            for confirmed in (False, True):
                with self.subTest(operation=operation, confirmed=confirmed):
                    result = self.ejecutar(FileAction(action=operation, nombre="."),
                                           confirmed=confirmed, destino_dir=self.clases)
                    self.assertIn("seguridad", result.text.lower())
                    self.assertTrue(self.proyectos.is_dir())

    def test_raiz_mediante_enlace_tambien_protegida(self):
        (self.proyectos / "enlace").symlink_to(self.clases)
        result = self.ejecutar(FileAction(action="eliminar", nombre="enlace"), confirmed=True)
        self.assertIn("seguridad", result.text)
        self.assertTrue(self.clases.is_dir())

    def test_creacion_nunca_sobrescribe(self):
        target = self.proyectos / "apuntes.txt"
        target.write_text("contenido original")
        for confirmed in (False, True):
            result = self.ejecutar(FileAction(action="crear_archivo", nombre=target.name,
                                               contenido="reemplazo"), confirmed=confirmed)
            self.assertIn("no lo sobrescribo", result.text)
            self.assertEqual(target.read_text(), "contenido original")

    def test_creacion_exclusiva_conserva_enlace(self):
        target = self.proyectos / "original.txt"
        target.write_text("original")
        (self.proyectos / "alias.txt").symlink_to(target)
        self.ejecutar(FileAction(action="crear_archivo", nombre="alias.txt", contenido="otro"))
        self.assertEqual(target.read_text(), "original")

    def test_crea_contenido_utf8_y_audita(self):
        self.ejecutar(FileAction(action="crear_archivo", nombre="nuevo/apuntes.txt", contenido="árbol y canción"))
        self.assertEqual((self.proyectos / "nuevo/apuntes.txt").read_text(), "árbol y canción")
        self.assertEqual(self.auditoria()[-1]["accion"]["action"], "crear_archivo")

    def test_mover_conserva_destino_existente_y_enlace_roto(self):
        source = self.proyectos / "apuntes.txt"
        source.write_text("origen")
        destination = self.clases / source.name
        destination.symlink_to(self.clases / "no_existe")
        result = self.ejecutar(FileAction(action="mover", nombre=source.name),
                               confirmed=True, destino_dir=self.clases)
        self.assertIn("no lo muevo", result.text)
        self.assertEqual(source.read_text(), "origen")
        self.assertTrue(destination.is_symlink())

    def test_traversal_y_base_externa_bloqueados(self):
        target = self.tmp / "externo.txt"
        target.write_text("intacto")
        self.ejecutar(FileAction(action="eliminar", nombre="../externo.txt"), confirmed=True)
        self.assertEqual(target.read_text(), "intacto")
        result = self.executor.execute(FileAction(action="listar", nombre=str(self.proyectos)),
                                       base_dir=self.tmp, raw_text="prueba")
        self.assertIn("seguridad", result.text)

    def test_nombre_absoluto_no_reemplaza_ubicacion(self):
        result = self.ejecutar(FileAction(action="crear_archivo", nombre=str(self.clases / "otro.txt")))
        self.assertIn("seguridad", result.text)
        self.assertFalse((self.clases / "otro.txt").exists())

    def test_ubicacion_con_barras_resuelve_igual_que_hablada(self):
        """El NLU devuelve la ubicación anidada como ruta ("clases/ada") en vez de
        como frase. Antes no coincidía ninguna raíz —era UNA sola palabra— y el
        asistente volvía a preguntar la ubicación aunque ya se la hubieran dicho."""
        (self.clases / "ADA").mkdir()
        equivalentes = {
            "clases/ada": self.clases / "ADA",                    # corrige la mayúscula contra el disco
            "clases/ada/parciales": self.clases / "ADA" / "parciales",
            "~/clases/ada": self.clases / "ADA",                  # prefijos que no aportan segmentos
            f"{self.tmp}/clases/ada": self.clases / "ADA",
            "clases//ada": self.clases / "ADA",
            "clases/": self.clases,
            "clases/la carpeta ada": self.clases / "ADA",
            "clases, carpeta ada": self.clases / "ADA",           # la frase hablada sigue igual
            "dentro de la carpeta ada en clases": self.clases / "ADA",
        }
        for texto, esperado in equivalentes.items():
            with self.subTest(texto=texto):
                self.assertEqual(paths_mod.parse_location_speech(texto), esperado)

    def test_ubicacion_con_barras_no_permite_salir_de_la_raiz(self):
        for texto in ("clases/../fuera", "clases/../../etc/passwd", "clases/./ada",
                      "/etc/passwd", "../../etc", "descargas/cosas", "clases\\ada"):
            with self.subTest(texto=texto):
                self.assertIsNone(paths_mod.parse_location_speech(texto))

    def test_crear_en_subcarpeta_que_dio_el_nlu(self):
        """Camino completo del caso real: 'crea una carpeta llamada parciales en
        clases, dentro de la carpeta ada' -> ruta_base='clases/ada'."""
        (self.clases / "ADA").mkdir()
        self.loop._dispatch_action(
            FileAction(action="crear_carpeta", nombre="parciales", ruta_base="clases/ada"), "prueba")
        self.assertTrue((self.clases / "ADA" / "parciales").is_dir())
        self.assertFalse((self.clases / "parciales").exists())

    def test_confirmaciones_condicionales_no_autorizan(self):
        for text in ("sí pero espera", "por si acaso", "dale mañana", "correcto pero después", "quizás sí"):
            with self.subTest(text=text):
                self.assertIsNot(interpret_confirmation(text), True)
        for text in ("sí", "sí, confirmo", "dale", "Sí, por favor."):
            self.assertIs(interpret_confirmation(text), True)

    def test_ubicacion_incomprensible_cancela_y_audita(self):
        self.loop._listen_reply.return_value = "no sé dónde"
        self.loop._dispatch_action(FileAction(action="crear_archivo", nombre="nuevo.txt"), "prueba")
        self.assertFalse((self.proyectos / "nuevo.txt").exists())
        self.assertTrue(self.auditoria()[-1]["cancelado"])

    def test_silencio_cancela_eliminar_y_audita(self):
        (self.proyectos / "nota.txt").write_text("conservar")
        self.loop._listen_reply.return_value = ""
        self.loop._dispatch_action(FileAction(action="eliminar", nombre="nota.txt", ruta_base="proyectos"), "prueba")
        self.assertTrue((self.proyectos / "nota.txt").exists())
        self.assertTrue(self.auditoria()[-1]["cancelado"])

    def test_confirmacion_inaudible_no_escucha_ni_ejecuta(self):
        (self.proyectos / "nota.txt").write_text("conservar")
        self.loop.tts.speak.return_value = False
        self.loop._dispatch_action(FileAction(action="eliminar", nombre="nota.txt", ruta_base="proyectos"), "prueba")
        self.loop._listen_reply.assert_not_called()
        self.assertTrue((self.proyectos / "nota.txt").exists())
        self.assertTrue(self.auditoria()[-1]["cancelado"])

    def test_eliminar_confirmado_respeta_la_ubicacion_dicha(self):
        for root in (self.proyectos, self.clases):
            (root / "nota.txt").write_text(root.name)
        self.loop._dispatch_action(FileAction(action="eliminar", nombre="nota.txt", ruta_base="clases"), "prueba")
        self.assertTrue((self.proyectos / "nota.txt").exists())
        self.assertFalse((self.clases / "nota.txt").exists())
        self.assertTrue(self.auditoria()[-1]["confirmado"])

    def test_revalida_raiz_cambiada_durante_confirmacion(self):
        source = self.proyectos / "carpeta"
        source.mkdir()
        def respuesta(**kwargs):
            source.rename(self.proyectos / "carpeta_original")
            source.symlink_to(self.clases)
            return "sí"
        self.loop._listen_reply.side_effect = respuesta
        self.loop._dispatch_action(FileAction(action="eliminar", nombre="carpeta", ruta_base="proyectos"), "prueba")
        self.assertTrue(self.clases.is_dir())
        self.assertIn("seguridad", self.auditoria()[-1]["resultado"])

    def test_flujo_mover_confirmado(self):
        (self.proyectos / "nota.txt").write_text("mover intacto")
        self.loop._dispatch_action(FileAction(action="mover", nombre="nota.txt", ruta_base="proyectos", destino="clases"), "prueba")
        self.assertFalse((self.proyectos / "nota.txt").exists())
        self.assertEqual((self.clases / "nota.txt").read_text(), "mover intacto")

    def test_lectura_privada_y_saludo_publico(self):
        (self.proyectos / "nota.txt").write_text("contenido de prueba privado")
        self.loop._dispatch_action(FileAction(action="leer", nombre="nota.txt", ruta_base="proyectos"), "prueba")
        self.assertFalse(self.loop.tts.speak.call_args.kwargs["public"])
        self.loop._dispatch_action(FileAction(action="saludo"), "hola")
        self.assertTrue(self.loop.tts.speak.call_args.kwargs["public"])

    def test_no_hay_acciones_declaradas_sin_implementacion(self):
        supported = set(_HANDLERS) | {"crear_archivo", "crear_carpeta", "mover", "eliminar", "listar", "leer", "ninguna"}
        self.assertEqual(set(get_args(FileAction.model_fields["action"].annotation)), supported)

    def test_sin_audio_y_sin_texto_recolectan_candidatos(self):
        self.loop.guardar_falso_positivo = Mock()
        self.loop.save_utterance = Mock()
        self.loop.stt = Mock()
        self.loop.stt.transcribe.return_value = ""
        self.assertEqual(self.loop._procesar_audio(None), (None, None, "sin_voz"))
        self.assertEqual(self.loop._procesar_audio(np.zeros(16000, dtype=np.int16)),
                         (None, None, "sin_transcripcion"))
        self.assertEqual(self.loop.guardar_falso_positivo.call_count, 2)

    def test_fallo_nlu_no_se_confunde_con_falso_positivo(self):
        self.loop.save_utterance = Mock()
        self.loop.guardar_falso_positivo = Mock()
        self.loop.stt = Mock()
        self.loop.stt.transcribe.return_value = "crea una carpeta de prueba"
        self.loop.nlu = Mock(last_error="ConnectionError")
        self.loop.nlu.parse.return_value = FileAction(action="ninguna")
        result = self.loop._procesar_audio(np.zeros(16000, dtype=np.int16))
        self.assertEqual(result, (None, "nlu", "error_nlu"))
        self.loop.guardar_falso_positivo.assert_not_called()
        self.assertTrue(self.auditoria()[-1]["cancelado"])


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main(verbosity=2)

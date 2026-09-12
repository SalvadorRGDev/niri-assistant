"""Movimiento real en disco temporal y errores simulados de libc, sin hardware.

Uso: .venv/bin/python eval/test_mover_seguro.py
"""
import ctypes
import errno
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import safe_move  # noqa: E402


class MovimientoSeguro(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "origen.txt"
        self.destination = self.root / "destino.txt"
        self.source.write_text("contenido original", encoding="utf-8")

    def assert_origen_intacto(self):
        self.assertEqual(self.source.read_text(encoding="utf-8"), "contenido original")

    def test_archivo_se_mueve_contenido_e_inode_intactos(self):
        inode = self.source.stat().st_ino
        safe_move.move_no_replace(self.source, self.destination)
        self.assertFalse(self.source.exists())
        self.assertEqual(self.destination.read_text(), "contenido original")
        self.assertEqual(self.destination.stat().st_ino, inode)

    def test_directorio_se_mueve_con_su_arbol(self):
        source = self.root / "carpeta"
        source.mkdir()
        (source / "apuntes.txt").write_text("apuntes")
        destination = self.root / "carpeta_movida"
        safe_move.move_no_replace(source, destination)
        self.assertFalse(source.exists())
        self.assertEqual((destination / "apuntes.txt").read_text(), "apuntes")

    def test_archivo_existente_nunca_se_reemplaza(self):
        self.destination.write_text("conservar destino")
        with self.assertRaises(FileExistsError) as raised:
            safe_move.move_no_replace(self.source, self.destination)
        self.assertEqual(raised.exception.errno, errno.EEXIST)
        self.assertEqual(raised.exception.filename, str(self.source))
        self.assertEqual(raised.exception.filename2, str(self.destination))
        self.assert_origen_intacto()
        self.assertEqual(self.destination.read_text(), "conservar destino")

    def test_directorio_existente_no_recibe_el_origen(self):
        self.destination.mkdir()
        with self.assertRaises(FileExistsError):
            safe_move.move_no_replace(self.source, self.destination)
        self.assert_origen_intacto()
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_directorio_existente_no_es_reemplazado_por_otro(self):
        source = self.root / "carpeta"
        source.mkdir()
        (source / "archivo.txt").write_text("conservar origen")
        self.destination.mkdir()
        with self.assertRaises(FileExistsError):
            safe_move.move_no_replace(source, self.destination)
        self.assertEqual((source / "archivo.txt").read_text(), "conservar origen")
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_enlace_roto_en_destino_se_conserva(self):
        missing = self.root / "no_existe"
        self.destination.symlink_to(missing)
        with self.assertRaises(FileExistsError):
            safe_move.move_no_replace(self.source, self.destination)
        self.assert_origen_intacto()
        self.assertTrue(self.destination.is_symlink())
        self.assertEqual(self.destination.readlink(), missing)
        self.assertFalse(missing.exists())

    def test_enlace_como_origen_mueve_el_enlace_no_su_contenido(self):
        alias = self.root / "alias.txt"
        alias.symlink_to(self.source)
        safe_move.move_no_replace(alias, self.destination)
        self.assertFalse(alias.is_symlink())
        self.assertTrue(self.destination.is_symlink())
        self.assertEqual(self.destination.readlink(), self.source)
        self.assert_origen_intacto()

    def test_destino_creado_justo_antes_de_syscall_se_conserva(self):
        syscall = safe_move._load_renameat2()

        def concurrent_writer(*args):
            self.destination.write_text("escritura concurrente")
            return syscall(*args)

        with patch.object(safe_move, "_load_renameat2", return_value=concurrent_writer):
            with self.assertRaises(FileExistsError):
                safe_move.move_no_replace(self.source, self.destination)
        self.assert_origen_intacto()
        self.assertEqual(self.destination.read_text(), "escritura concurrente")

    def test_directorio_creado_justo_antes_de_syscall_se_conserva(self):
        syscall = safe_move._load_renameat2()

        def concurrent_mkdir(*args):
            self.destination.mkdir()
            return syscall(*args)

        with patch.object(safe_move, "_load_renameat2", return_value=concurrent_mkdir):
            with self.assertRaises(FileExistsError):
                safe_move.move_no_replace(self.source, self.destination)
        self.assert_origen_intacto()
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_cross_device_y_otros_errores_no_intentan_respaldo(self):
        for error in (errno.EXDEV, errno.ENOSYS, errno.EOPNOTSUPP, errno.EACCES, errno.EINVAL):
            with self.subTest(errno=error):
                def failed_syscall(*args):
                    ctypes.set_errno(error)
                    return -1
                with patch.object(safe_move, "_load_renameat2", return_value=failed_syscall):
                    with self.assertRaises(OSError) as raised:
                        safe_move.move_no_replace(self.source, self.destination)
                self.assertEqual(raised.exception.errno, error)
                self.assert_origen_intacto()
                self.assertFalse(self.destination.exists())

    def test_libc_sin_renameat2_cancela(self):
        with patch.object(safe_move.ctypes, "CDLL", return_value=SimpleNamespace()):
            with self.assertRaises(OSError) as raised:
                safe_move.move_no_replace(self.source, self.destination)
        self.assertEqual(raised.exception.errno, errno.ENOSYS)
        self.assert_origen_intacto()
        self.assertFalse(self.destination.exists())

    def test_fallo_cargando_libc_cancela(self):
        with patch.object(safe_move.ctypes, "CDLL", side_effect=OSError("fallo simulado")):
            with self.assertRaises(OSError) as raised:
                safe_move.move_no_replace(self.source, self.destination)
        self.assertEqual(raised.exception.errno, errno.ENOSYS)
        self.assert_origen_intacto()
        self.assertFalse(self.destination.exists())

    def test_plataforma_sin_soporte_cancela_sin_llamar_libc(self):
        with patch.object(safe_move.sys, "platform", "otro"), patch.object(safe_move.ctypes, "CDLL") as libc:
            with self.assertRaises(OSError) as raised:
                safe_move.move_no_replace(self.source, self.destination)
        self.assertEqual(raised.exception.errno, errno.ENOSYS)
        libc.assert_not_called()
        self.assert_origen_intacto()

    def test_origen_inexistente_no_crea_destino(self):
        with self.assertRaises(FileNotFoundError):
            safe_move.move_no_replace(self.root / "ausente.txt", self.destination)
        self.assertFalse(self.destination.exists())

    def test_rutas_unicode(self):
        destination = self.root / "canción_árbol.txt"
        safe_move.move_no_replace(self.source, destination)
        self.assertEqual(destination.read_text(), "contenido original")

    def test_nul_no_trunca_origen_ni_destino(self):
        for source, destination in (
            (Path(str(self.source) + "\x00otro"), self.destination),
            (self.source, Path(str(self.destination) + "\x00otro")),
        ):
            with self.subTest(source=source, destination=destination):
                with patch.object(safe_move, "_load_renameat2") as load:
                    with self.assertRaises(OSError) as raised:
                        safe_move.move_no_replace(source, destination)
                self.assertEqual(raised.exception.errno, errno.EINVAL)
                load.assert_not_called()
                self.assert_origen_intacto()
                self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)

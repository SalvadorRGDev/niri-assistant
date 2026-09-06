"""
Índice híbrido de los archivos del usuario: BM25 + vectores, todo local.

Responde a "¿dónde dejé el resumen de sistemas operativos?", que es lo que el
asistente no podía hacer: hasta ahora sabía listar y leer un archivo por nombre
exacto, nada más.

Por qué híbrido y no solo vectores: medido sobre 10 consultas en español, el
codificador solo acierta 9 de 10 en el primer puesto, y el que falla lo hace
entre dos temas de informática que comparten vocabulario — justo donde el
BM25 acierta, porque la palabra literal está. Y al revés: BM25 no encuentra
nada si el usuario no usó ninguna palabra del archivo. Se fusionan con RRF, que
no necesita que los dos puntajes estén en la misma escala.

Frontera de seguridad: solo se indexa lo que hay bajo las raíces de
`src/paths.py` (`~/Proyectos` y `~/Clases`). El índice vive en `data/`, que no
se versiona, y no sale de la máquina.
"""
import os
import sqlite3
import threading
import time
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

from src.config import BASE_DIR, DATA_DIR, LOGS_DIR, MODELS_DIR, RECORDINGS_DIR
from src.paths import ALLOWED_ROOTS
from src.embeddings import Embeddings, DIMENSION
from src.logger import get_logger

logger = get_logger("Indice")

INDICE_PATH = Path(os.getenv("NIRI_INDICE_PATH", str(DATA_DIR / "indice.db")))

# Extensiones cuyo contenido vale la pena leer. Del resto se indexa el nombre,
# que muchas veces es todo lo que el usuario recuerda ("el pdf del parcial").
EXTENSIONES_TEXTO = {
    ".txt", ".md", ".rst", ".org", ".tex", ".csv", ".tsv", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".log", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".sh",
    ".sql", ".html", ".css", ".xml", ".env", ".gitignore",
}

# Carpetas que no son del usuario aunque vivan bajo sus raíces: son artefactos de
# herramientas y meterlas al índice lo llena de ruido que nadie va a buscar.
CARPETAS_IGNORADAS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "target", "build", "dist", ".next", ".cache",
    ".idea", ".vscode", "site-packages",
}

# Directorios de datos de runtime del propio asistente. Viven bajo ~/Proyectos,
# así que sin esta exclusión el índice se los traga: en la primera corrida entraron
# los 84 nombres de `recordings/`, el estado de `data/` y —peor— el contenido de
# `logs/audit.log`, que son las transcripciones de todo lo que el usuario dijo
# frente al micrófono. Nada de eso es un documento que alguien vaya a buscar, y
# copiarlo a otro archivo va contra el criterio de §3.1.4. El código fuente del
# asistente sí se indexa: es un proyecto del usuario como cualquier otro.
DIRECTORIOS_EXCLUIDOS = tuple(
    str(d.resolve())
    for d in (RECORDINGS_DIR, LOGS_DIR, DATA_DIR, MODELS_DIR,
              BASE_DIR / ".venv", BASE_DIR / "eval" / "resultados")
)


def _esta_excluido(carpeta: Path) -> bool:
    """True si la carpeta ES uno de los directorios excluidos o está adentro."""
    ruta = str(carpeta.resolve())
    return any(ruta == d or ruta.startswith(d + os.sep) for d in DIRECTORIOS_EXCLUIDOS)

MAX_BYTES_LEIDOS = 4096      # fragmento por archivo: alcanza para saber de qué trata
MAX_TAMANO_ARCHIVO = 5 * 1024 * 1024
K_CANDIDATOS = 30            # por cada rama antes de fusionar
RRF_K = 60                   # constante estándar de Reciprocal Rank Fusion

# Palabras que no aportan a una búsqueda por palabras y sí ensucian el BM25.
_VACIAS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "a",
    "en", "y", "o", "que", "donde", "esta", "estan", "mi", "mis", "me", "lo",
    "para", "por", "con", "sobre", "archivo", "archivos", "carpeta", "busca",
    "buscar", "buscame", "encontra", "encontrar", "encontrame", "deje", "dejé",
    "tengo", "hay", "cual", "cuales", "es", "son",
}


def _normalizar(texto: str) -> str:
    texto = (texto or "").lower()
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))


def _terminos(consulta: str) -> List[str]:
    """Palabras útiles de la consulta, sin acentos ni relleno."""
    limpio = "".join(c if c.isalnum() or c.isspace() else " " for c in _normalizar(consulta))
    return [p for p in limpio.split() if len(p) > 2 and p not in _VACIAS]


class Indice:
    """Índice híbrido sobre SQLite. Todas las operaciones son locales y síncronas."""

    def __init__(self, ruta: Path = INDICE_PATH, embeddings: Optional[Embeddings] = None):
        self.ruta = ruta
        self.embeddings = embeddings or Embeddings()
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + lock: el actualizador corre en un hilo de
        # fondo y la búsqueda en el hilo de voz, y comparten instancia para no
        # tener dos copias del modelo de 118 MB en RAM. WAL permite leer
        # mientras se escribe; el lock serializa el uso de la conexión, que es
        # lo que sqlite3 no garantiza por sí solo.
        self.con = sqlite3.connect(str(self.ruta), check_same_thread=False)
        self._lock = threading.Lock()
        self.con.execute("PRAGMA journal_mode=WAL")
        self._crear_esquema()

    def _crear_esquema(self) -> None:
        self.con.executescript("""
            CREATE TABLE IF NOT EXISTS documentos (
                id        INTEGER PRIMARY KEY,
                ruta      TEXT UNIQUE NOT NULL,
                nombre    TEXT NOT NULL,
                raiz      TEXT NOT NULL,
                mtime     REAL NOT NULL,
                tamano    INTEGER NOT NULL,
                fragmento TEXT NOT NULL DEFAULT ''
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS documentos_fts USING fts5(nombre, fragmento);
            CREATE TABLE IF NOT EXISTS vectores (
                id INTEGER PRIMARY KEY REFERENCES documentos(id) ON DELETE CASCADE,
                v  BLOB NOT NULL
            );
        """)
        self.con.commit()

    # --- indexado -------------------------------------------------------

    def _recorrer(self) -> Iterable[Tuple[Path, Path]]:
        """Genera (raíz, archivo) para todo lo indexable bajo la whitelist."""
        for raiz in ALLOWED_ROOTS:
            if not raiz.is_dir():
                continue
            for carpeta, subcarpetas, archivos in os.walk(raiz):
                subcarpetas[:] = [s for s in subcarpetas
                                  if s not in CARPETAS_IGNORADAS and not s.startswith(".")]
                if _esta_excluido(Path(carpeta)):
                    subcarpetas[:] = []
                    continue
                for nombre in archivos:
                    if nombre.startswith("."):
                        continue
                    yield raiz, Path(carpeta) / nombre

    @staticmethod
    def _fragmento(archivo: Path) -> str:
        """Primeros bytes del archivo si es texto; cadena vacía si no."""
        if archivo.suffix.lower() not in EXTENSIONES_TEXTO:
            return ""
        try:
            with open(archivo, "r", encoding="utf-8", errors="ignore") as f:
                return " ".join(f.read(MAX_BYTES_LEIDOS).split())
        except OSError:
            return ""

    @staticmethod
    def _texto_para_vector(nombre: str, ruta_relativa: str, fragmento: str) -> str:
        """
        Qué se codifica de cada documento.

        Incluye la ruta relativa además del nombre porque la carpeta suele llevar
        el tema ("Clases/Sistemas Operativos/tp2.pdf"), que es exactamente lo que
        el usuario recuerda cuando no se acuerda del nombre del archivo.
        """
        return f"{ruta_relativa} {nombre} {fragmento}".strip()[:1000]

    def actualizar(self, verboso: bool = False) -> dict:
        """
        Sincroniza el índice con el disco. Solo toca lo que cambió: compara
        `mtime` y tamaño, así que una segunda corrida sobre un árbol quieto no
        vuelve a leer ni a codificar nada.
        """
        with self._lock:
            return self._actualizar(verboso)

    def _actualizar(self, verboso: bool) -> dict:
        inicio = time.perf_counter()
        existentes = {r: (i, m, t) for i, r, m, t in
                      self.con.execute("SELECT id, ruta, mtime, tamano FROM documentos")}
        vistos, pendientes = set(), []

        for raiz, archivo in self._recorrer():
            ruta = str(archivo)
            vistos.add(ruta)
            try:
                st = archivo.stat()
            except OSError:
                continue
            if st.st_size > MAX_TAMANO_ARCHIVO:
                continue
            previo = existentes.get(ruta)
            if previo and abs(previo[1] - st.st_mtime) < 1e-6 and previo[2] == st.st_size:
                continue
            pendientes.append((raiz, archivo, st))

        nuevos = modificados = 0
        for raiz, archivo, st in pendientes:
            ruta = str(archivo)
            fragmento = self._fragmento(archivo)
            relativa = str(archivo.relative_to(raiz))
            cur = self.con.execute(
                "INSERT INTO documentos (ruta, nombre, raiz, mtime, tamano, fragmento) "
                "VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(ruta) DO UPDATE SET mtime=excluded.mtime, tamano=excluded.tamano, "
                "fragmento=excluded.fragmento RETURNING id",
                (ruta, archivo.name, raiz.name, st.st_mtime, st.st_size, fragmento),
            )
            doc_id = cur.fetchone()[0]
            if ruta in existentes:
                modificados += 1
            else:
                nuevos += 1
            self.con.execute("DELETE FROM documentos_fts WHERE rowid = ?", (doc_id,))
            self.con.execute(
                "INSERT INTO documentos_fts (rowid, nombre, fragmento) VALUES (?,?,?)",
                (doc_id, _normalizar(f"{relativa} {archivo.name}"), _normalizar(fragmento)),
            )

        borrados = 0
        for ruta, (doc_id, _, _) in existentes.items():
            if ruta not in vistos:
                self.con.execute("DELETE FROM documentos WHERE id = ?", (doc_id,))
                self.con.execute("DELETE FROM documentos_fts WHERE rowid = ?", (doc_id,))
                self.con.execute("DELETE FROM vectores WHERE id = ?", (doc_id,))
                borrados += 1
        self.con.commit()

        vectorizados = self._vectorizar_pendientes()
        total = self.con.execute("SELECT COUNT(*) FROM documentos").fetchone()[0]
        resumen = {"total": total, "nuevos": nuevos, "modificados": modificados,
                   "borrados": borrados, "vectorizados": vectorizados,
                   "segundos": round(time.perf_counter() - inicio, 2)}
        if verboso or nuevos or modificados or borrados:
            logger.info(f"Índice actualizado: {resumen}")
        return resumen

    def _vectorizar_pendientes(self) -> int:
        """Codifica los documentos que todavía no tienen vector."""
        filas = self.con.execute(
            "SELECT d.id, d.nombre, d.ruta, d.raiz, d.fragmento FROM documentos d "
            "LEFT JOIN vectores v ON v.id = d.id WHERE v.id IS NULL"
        ).fetchall()
        if not filas:
            return 0

        textos = []
        for _, nombre, ruta, raiz, fragmento in filas:
            relativa = ruta.split(f"/{raiz}/", 1)[-1] if f"/{raiz}/" in ruta else nombre
            textos.append(self._texto_para_vector(nombre, f"{raiz}/{relativa}", fragmento))

        matriz = self.embeddings.codificar(textos)
        if matriz is None:
            return 0
        self.con.executemany(
            "INSERT OR REPLACE INTO vectores (id, v) VALUES (?, ?)",
            [(fila[0], vector.astype(np.float32).tobytes()) for fila, vector in zip(filas, matriz)],
        )
        self.con.commit()
        return len(filas)

    # --- búsqueda -------------------------------------------------------

    def _buscar_bm25(self, consulta: str, k: int) -> List[int]:
        terminos = _terminos(consulta)
        if not terminos:
            return []
        # OR y no AND: una orden hablada trae palabras de más, y exigirlas todas
        # deja la búsqueda vacía justo cuando el usuario fue más específico.
        expresion = " OR ".join(f'"{t}"' for t in terminos)
        try:
            filas = self.con.execute(
                "SELECT rowid FROM documentos_fts WHERE documentos_fts MATCH ? "
                "ORDER BY rank LIMIT ?", (expresion, k)).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning(f"Consulta FTS inválida ({expresion!r}): {e}")
            return []
        return [f[0] for f in filas]

    def _buscar_vectores(self, consulta: str, k: int) -> List[int]:
        vector = self.embeddings.codificar_consulta(consulta)
        if vector is None:
            return []
        filas = self.con.execute("SELECT id, v FROM vectores").fetchall()
        if not filas:
            return []
        ids = np.array([f[0] for f in filas])
        matriz = np.frombuffer(b"".join(f[1] for f in filas), dtype=np.float32).reshape(len(filas), DIMENSION)
        similitudes = matriz @ vector
        mejores = np.argsort(-similitudes)[:k]
        return [int(ids[i]) for i in mejores]

    def buscar(self, consulta: str, limite: int = 5) -> List[dict]:
        """
        Busca por palabras y por significado, y fusiona con RRF.

        RRF suma 1/(60 + puesto) de cada lista: solo mira el ORDEN, así que no
        hace falta que el puntaje BM25 y la similitud coseno vivan en la misma
        escala — que no viven.

        Cada resultado trae `literal`: si ninguna palabra de la consulta aparece
        en el documento, el resultado viene solo de la rama vectorial y hay que
        presentarlo como aproximación, no como hallazgo. Se probó cortar por
        similitud y no sirve: sobre este corpus el margen contra la mediana es
        +0.036 cuando el archivo existe y +0.033 cuando no, o sea que ningún
        umbral los separa.
        """
        with self._lock:
            return self._buscar(consulta, limite)

    def _buscar(self, consulta: str, limite: int) -> List[dict]:
        puntajes: dict[int, float] = {}
        por_palabras = self._buscar_bm25(consulta, K_CANDIDATOS)
        for lista in (por_palabras, self._buscar_vectores(consulta, K_CANDIDATOS)):
            for puesto, doc_id in enumerate(lista):
                puntajes[doc_id] = puntajes.get(doc_id, 0.0) + 1.0 / (RRF_K + puesto + 1)

        if not puntajes:
            return []
        literales = set(por_palabras)
        mejores = sorted(puntajes.items(), key=lambda x: -x[1])[:limite]
        marcadores = ",".join("?" * len(mejores))
        filas = {
            f[0]: f for f in self.con.execute(
                f"SELECT id, ruta, nombre, raiz FROM documentos WHERE id IN ({marcadores})",
                [d for d, _ in mejores])
        }
        # `literal` dice si el documento apareció por una palabra que el usuario
        # dijo de verdad. Importa para no mentir al hablar: la rama vectorial
        # SIEMPRE devuelve sus 30 más parecidos, tenga o no sentido, así que sin
        # esta marca el asistente contestaría con cualquier cosa ante una
        # búsqueda de algo que no existe.
        return [{"ruta": filas[d][1], "nombre": filas[d][2], "raiz": filas[d][3],
                 "puntaje": round(p, 5), "literal": d in literales}
                for d, p in mejores if d in filas]

    def estadisticas(self) -> dict:
        return {
            "documentos": self.con.execute("SELECT COUNT(*) FROM documentos").fetchone()[0],
            "vectorizados": self.con.execute("SELECT COUNT(*) FROM vectores").fetchone()[0],
            "con_texto": self.con.execute(
                "SELECT COUNT(*) FROM documentos WHERE fragmento != ''").fetchone()[0],
        }

    def cerrar(self) -> None:
        self.con.close()

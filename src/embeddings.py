"""
Codificador de texto a vectores, 100% local y en CPU.

Usa `multilingual-e5-small` cuantizado a int8 (118 MB) con onnxruntime. Corre en
CPU a propósito: la GPU de 4 GB ya tiene al NLU ocupando ~2.2 GB y el escritorio
necesita el resto, así que la búsqueda no puede competir por VRAM. Medido en este
equipo: 935 textos/s por lote y 3 ms una consulta suelta, sin necesitar AVX-512.

Sobre los prefijos: e5 fue entrenado con "query: " para lo que se busca y
"passage: " para lo que se indexa, y sin ellos la calidad cae. Por eso los pone
esta clase y no quien la llama.

Nota de lectura: las similitudes de e5 viven todas cerca de 0.8, incluso entre
textos que no tienen nada que ver. El número absoluto no dice nada; lo que sirve
es el orden. Medido sobre 10 consultas en español: recall@1 = 9/10, recall@3 = 10/10.
"""
import os
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from src.config import MODELS_DIR
from src.logger import get_logger

logger = get_logger("Embeddings")

MODELO_DIR = Path(os.getenv("NIRI_EMBEDDINGS_DIR", str(MODELS_DIR / "e5-small")))
DIMENSION = 384
MAX_TOKENS = 128          # los nombres y fragmentos son cortos; truncar acota el costo
LOTE = 32


class Embeddings:
    """
    Envoltorio del codificador. La carga es perezosa: el modelo son 118 MB de
    RAM que no tiene sentido pagar si el usuario nunca busca nada.
    """

    def __init__(self, modelo_dir: Path = MODELO_DIR):
        self.modelo_dir = modelo_dir
        self._sesion = None
        self._tokenizer = None
        self._usa_token_type = False

    @property
    def disponible(self) -> bool:
        return (self.modelo_dir / "model.onnx").exists() and \
               (self.modelo_dir / "tokenizer.json").exists()

    def _cargar(self) -> bool:
        if self._sesion is not None:
            return True
        if not self.disponible:
            logger.warning(
                f"Falta el modelo de embeddings en {self.modelo_dir}. La búsqueda va a "
                "funcionar solo por palabras (BM25), sin significado."
            )
            return False
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
            self._sesion = ort.InferenceSession(str(self.modelo_dir / "model.onnx"),
                                                 providers=["CPUExecutionProvider"])
            self._tokenizer = Tokenizer.from_file(str(self.modelo_dir / "tokenizer.json"))
            self._tokenizer.enable_truncation(MAX_TOKENS)
            self._tokenizer.enable_padding(length=None)
            self._usa_token_type = "token_type_ids" in [e.name for e in self._sesion.get_inputs()]
            logger.info(f"Codificador cargado desde {self.modelo_dir.name}.")
            return True
        except Exception as e:
            logger.error(f"No pude cargar el codificador de embeddings: {e}")
            self._sesion = None
            return False

    def _codificar_lote(self, textos: List[str]) -> np.ndarray:
        codificados = self._tokenizer.encode_batch(textos)
        ids = np.array([c.ids for c in codificados], dtype=np.int64)
        mascara = np.array([c.attention_mask for c in codificados], dtype=np.int64)
        entradas = {"input_ids": ids, "attention_mask": mascara}
        if self._usa_token_type:
            entradas["token_type_ids"] = np.zeros_like(ids)
        estados = self._sesion.run(None, entradas)[0]

        # Mean pooling sobre los tokens reales (sin el relleno) y normalización L2,
        # que es lo que e5 espera para comparar con producto interno.
        m = mascara[..., None].astype(np.float32)
        vectores = (estados * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
        normas = np.linalg.norm(vectores, axis=1, keepdims=True)
        return (vectores / np.maximum(normas, 1e-9)).astype(np.float32)

    def codificar(self, textos: Iterable[str], prefijo: str = "passage") -> Optional[np.ndarray]:
        """
        Devuelve una matriz (n, 384) normalizada, o None si el modelo no está.

        `prefijo` es "passage" para lo que se indexa y "query" para lo que se busca.
        """
        textos = [f"{prefijo}: {t}" for t in textos]
        if not textos or not self._cargar():
            return None
        partes = [self._codificar_lote(textos[i:i + LOTE]) for i in range(0, len(textos), LOTE)]
        return np.vstack(partes)

    def codificar_consulta(self, texto: str) -> Optional[np.ndarray]:
        """Atajo para una sola consulta: devuelve un vector (384,) o None."""
        matriz = self.codificar([texto], prefijo="query")
        return matriz[0] if matriz is not None else None

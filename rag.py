"""
rag.py — Búsqueda semántica sobre numpy (sin ChromaDB).

Carga embeddings.npy + metadata.json y calcula similitud coseno
para recuperar los k fragmentos más relevantes a una consulta.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import VECTORSTORE_DIR, EMBEDDINGS_FILE, METADATA_FILE, EMBED_MODEL

TOP_K_DEFAULT = 3


class RecuperadorRAG:

    def __init__(self, top_k: int = TOP_K_DEFAULT):
        self.top_k = top_k

        if not EMBEDDINGS_FILE.exists() or not METADATA_FILE.exists():
            raise RuntimeError(
                "Vectorstore no encontrado. Ejecuta primero: python ingest.py"
            )

        self._modelo    = SentenceTransformer(EMBED_MODEL)
        self._embeddings = np.load(str(EMBEDDINGS_FILE))          # (N, dim) float32
        self._meta       = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

        # Pre-normalizar para que la similitud coseno sea solo un producto punto
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        self._embeddings_norm = self._embeddings / norms

    # ── API pública ────────────────────────────────────────────────────────────

    def recuperar(self, consulta: str, k: int | None = None) -> list[dict]:
        k = k or self.top_k
        q = self._modelo.encode([consulta])[0].astype(np.float32)
        q /= max(np.linalg.norm(q), 1e-9)

        similitudes = self._embeddings_norm @ q          # producto punto = coseno
        indices     = np.argsort(similitudes)[::-1][:k]  # top-k descendente

        return [
            {
                "texto":      self._meta[i]["texto"],
                "fuente":     self._meta[i]["fuente"],
                "similitud":  round(float(similitudes[i]), 4),
            }
            for i in indices
        ]

    def construir_contexto(self, consulta: str, k: int | None = None) -> tuple[str, list[str]]:
        chunks = self.recuperar(consulta, k)

        if not chunks:
            return "No se encontró información relevante.", []

        bloques, fuentes = [], []
        for i, c in enumerate(chunks, 1):
            bloques.append(
                f"[Fragmento {i} | Fuente: {c['fuente']} | Relevancia: {c['similitud']:.2%}]\n{c['texto']}"
            )
            fuentes.append(c["fuente"])

        return "\n\n".join(bloques), list(dict.fromkeys(fuentes))

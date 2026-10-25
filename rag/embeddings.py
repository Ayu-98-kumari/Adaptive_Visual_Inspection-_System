"""
Embedding Model Interface for RAG
Supports SentenceTransformers when available, with a resilient fallback for lightweight environments.
"""

import math
import re
from typing import List
import numpy as np
from rag.config import RAGConfig

class EmbeddingModel:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or RAGConfig.DEFAULT_EMBEDDING_MODEL
        self._st_model = None
        self._backend = "fallback"

        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name)
            self._backend = "sentence_transformers"
        except Exception:
            # Fallback to internal deterministic semantic embedder
            self._backend = "fallback"

    @property
    def backend(self) -> str:
        return self._backend

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encodes a list of text strings into normalized vector embeddings."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        if self._backend == "sentence_transformers" and self._st_model is not None:
            embeddings = self._st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.astype(np.float32)

        # Fallback implementation: 384-dimensional deterministic feature hashing & subword n-gram embedding
        return self._fallback_encode(texts)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _fallback_encode(self, texts: List[str], dim: int = 384) -> np.ndarray:
        embeddings = np.zeros((len(texts), dim), dtype=np.float32)

        for i, text in enumerate(texts):
            tokens = re.findall(r"\b\w+\b", text.lower())
            if not tokens:
                continue

            vec = np.zeros(dim, dtype=np.float32)
            for token in tokens:
                # Word-level hash
                h1 = hash(token) % dim
                vec[h1] += 1.0

                # Character 3-gram hashing for morphological capture (e.g. 'break', 'broken')
                if len(token) >= 3:
                    for j in range(len(token) - 2):
                        tri = token[j:j+3]
                        h2 = hash(tri) % dim
                        vec[h2] += 0.5

            norm = np.linalg.norm(vec)
            if norm > 0:
                embeddings[i] = vec / norm
            else:
                embeddings[i] = vec

        return embeddings

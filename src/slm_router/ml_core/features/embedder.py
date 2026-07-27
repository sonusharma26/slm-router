"""Sentence embedding with a deterministic offline fallback."""

from __future__ import annotations

import hashlib
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defensive imports
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer as _ST  # type: ignore

    _ST_AVAILABLE = True
except Exception:  # noqa: BLE001
    _ST_AVAILABLE = False
    _ST = None  # type: ignore

_MODEL_CACHE: dict[str, object] = {}

_FALLBACK_DIM = 384  # matches all-MiniLM-L6-v2


def _hash_embed(text: str, dim: int = _FALLBACK_DIM) -> np.ndarray:
    """Deterministic hashing-based embedding — no network, no torch required.

    Produces a unit-norm vector by hashing consecutive 64-bit chunks of the
    SHA-512 digest of the UTF-8 encoded text.
    """
    digest = hashlib.sha512(text.encode()).digest()  # 64 bytes
    # Tile digest until we have enough bytes
    repeats = (dim * 4 // len(digest)) + 2
    raw = (digest * repeats)[:dim]
    vec = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    vec = vec - vec.mean()
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class Embedder:
    """Wraps sentence-transformers with a deterministic fallback.

    When ``sentence-transformers`` / ``torch`` are unavailable the encoder
    falls back to a hashing-based embedder that returns unit-norm float32
    vectors of the same dimension.  Output is reproducible for identical
    inputs so unit tests pass without any ML dependencies.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Ignored by the fallback.
    use_fallback:
        Use the hashing fallback (default). Set false only when a locally available
        sentence-transformers model is explicitly intended.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_fallback: bool = True,
    ) -> None:
        self.model_name = model_name
        self._use_fallback = use_fallback or not _ST_AVAILABLE
        self._model: object | None = None

        if self._use_fallback:
            logger.warning("sentence-transformers not available; using hashing fallback embedder.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lazily load the sentence-transformer model (cached globally)."""
        if self._use_fallback or self._model is not None:
            return
        if self.model_name not in _MODEL_CACHE:
            logger.info("Loading SentenceTransformer model %s …", self.model_name)
            _MODEL_CACHE[self.model_name] = _ST(self.model_name)  # type: ignore[misc]
        self._model = _MODEL_CACHE[self.model_name]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        if self._use_fallback:
            return _FALLBACK_DIM
        self._load()
        # sentence-transformers exposes get_sentence_embedding_dimension()
        return self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]

    def encode(self, texts: list[str] | str) -> np.ndarray:
        """Encode one or more texts into a float32 embedding matrix.

        Parameters
        ----------
        texts:
            A single string or a list of strings.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), dim)`` or ``(dim,)`` for a single string.
        """
        scalar = isinstance(texts, str)
        if scalar:
            texts = [texts]

        if self._use_fallback:
            vecs = np.stack([_hash_embed(t) for t in texts])
        else:
            self._load()
            vecs = np.asarray(
                self._model.encode(texts, show_progress_bar=False),  # type: ignore[union-attr]
                dtype=np.float32,
            )

        return vecs[0] if scalar else vecs

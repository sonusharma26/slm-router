"""Request text features for learned routing.

``ngram_features`` is a dependency-free lexical encoder (hashed word unigrams and
bigrams with sublinear term frequency, plus a length signal). It is stronger than
the 64-dimensional ``hash-text-v1`` features but is still lexical, not semantic.
``SentenceEmbedder`` optionally uses a locally installed sentence-transformers
model. Feature vectors are always L2-normalized so cosine similarity is a dot product.
"""
from __future__ import annotations

import hashlib
import math
import re
from itertools import pairwise

NGRAM_FEATURE_VERSION = "hash-ngram-v2-256"
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _bucket(token: str, dimensions: int) -> tuple[int, float]:
    key = hashlib.blake2b(token.encode(), digest_size=8).digest()
    return int.from_bytes(key[:4], "big") % dimensions, (1. if key[4] & 1 else -1.)


def ngram_features(text: str, dimensions: int = 256) -> tuple[float, ...]:
    """Hashed unigram+bigram counts (1+log tf), a log-length channel, L2-normalized."""
    if dimensions < 8:
        raise ValueError("at least eight dimensions required")
    words = _WORD.findall(text.lower())
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    for left, right in pairwise(words):
        gram = left + " " + right
        counts[gram] = counts.get(gram, 0) + 1
    values = [0.] * dimensions
    hashed = dimensions - 1  # the final channel carries length
    for token, count in counts.items():
        index, sign = _bucket(token, hashed)
        values[index] += sign * (1. + math.log(count))
    norm = math.sqrt(sum(x*x for x in values[:hashed])) or 1.
    values = [x / norm for x in values[:hashed]]
    # Length is a weak difficulty signal; keep it small relative to lexical content.
    values.append(min(1., math.log1p(len(words)) / 10.) * .5)
    total = math.sqrt(sum(x*x for x in values)) or 1.
    return tuple(x / total for x in values)


class SentenceEmbedder:
    """Optional semantic features from a locally installed sentence-transformers model."""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install sentence-transformers to use semantic embeddings") from exc
        self.model_name = model_name
        self.feature_version = f"st:{model_name}"
        self._model = SentenceTransformer(model_name)

    def __call__(self, texts: list[str], batch_size: int = 64) -> list[tuple[float, ...]]:
        vectors = self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                     show_progress_bar=False)
        return [tuple(float(x) for x in row) for row in vectors]

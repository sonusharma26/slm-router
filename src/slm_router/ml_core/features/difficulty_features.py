"""Difficulty feature extraction from query text and embeddings."""
from __future__ import annotations

import re
import math
import logging
from typing import TYPE_CHECKING

import numpy as np

from slm_router.ml_core.features.embedder import Embedder
from slm_router.types import Query

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defensive imports
# ---------------------------------------------------------------------------
try:
    from sklearn.cluster import KMeans as _KMeans  # type: ignore
    _SKLEARN_AVAILABLE = True
except Exception:  # noqa: BLE001
    _KMeans = None  # type: ignore
    _SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Prototype sentences for "reasoning depth" similarity
# ---------------------------------------------------------------------------
_REASONING_PROTOTYPES = [
    "prove that the following statement is true",
    "derive step by step the solution",
    "explain why this phenomenon occurs",
    "what is the mathematical justification",
    "solve this multi-step problem",
]

# Regex cues that suggest reasoning depth
_REASONING_RE = re.compile(
    r"\b(step[\s-]by[\s-]step|prove|proof|derive|why|because|explain|"
    r"justify|show that|demonstrate|reasoning|logic|theorem|lemma|corollary)\b",
    re.IGNORECASE,
)
_NUMERIC_OP_RE = re.compile(r"[\+\-\*/=\^√∫∑∏<>≤≥≠]|\b\d+\b")


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _top_k_cosine(query_emb: np.ndarray, bank: np.ndarray, k: int = 5) -> np.ndarray:
    """Return top-k cosine similarities between query_emb and rows in bank."""
    norms = np.linalg.norm(bank, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    normed = bank / safe
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    sims = normed @ q_norm
    k = min(k, len(sims))
    return np.partition(sims, -k)[-k:]


class DifficultyFeaturizer:
    """Converts a :class:`~slm_router.types.Query` into a numeric feature vector.

    Parameters
    ----------
    embedder:
        Pre-instantiated :class:`Embedder`.
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._anchor_bank: np.ndarray | None = None  # shape (k, dim)
        self._proto_embs: np.ndarray | None = None  # reasoning prototypes

    # ------------------------------------------------------------------
    # Anchor / prototype helpers
    # ------------------------------------------------------------------

    def _get_proto_embs(self) -> np.ndarray:
        if self._proto_embs is None:
            self._proto_embs = self.embedder.encode(_REASONING_PROTOTYPES)
        return self._proto_embs

    def fit_anchors(self, corpus_embs: np.ndarray, k: int = 64) -> None:
        """Fit KMeans anchor bank offline from corpus embeddings.

        If sklearn is unavailable, randomly samples *k* rows as anchors.

        Parameters
        ----------
        corpus_embs:
            Shape ``(N, dim)`` embedding matrix.
        k:
            Number of cluster centroids / anchor points.
        """
        k = min(k, len(corpus_embs))
        if _SKLEARN_AVAILABLE and _KMeans is not None:
            km = _KMeans(n_clusters=k, n_init="auto", random_state=0)
            km.fit(corpus_embs)
            self._anchor_bank = km.cluster_centers_.astype(np.float32)
        else:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(corpus_embs), size=k, replace=False)
            self._anchor_bank = corpus_embs[idx].astype(np.float32)
        logger.info("Anchor bank fitted: %d anchors.", k)

    # ------------------------------------------------------------------
    # Individual feature components
    # ------------------------------------------------------------------

    def embedding_entropy(
        self,
        q_emb: np.ndarray,
        anchor_bank: np.ndarray | None = None,
    ) -> float:
        """Entropy over softmax similarities to anchor points.

        High entropy ≈ query is equidistant from all anchors ≈ hard/rare.

        Returns
        -------
        float in [0, 1] (normalised by log(k))
        """
        bank = anchor_bank if anchor_bank is not None else self._anchor_bank
        if bank is None or len(bank) == 0:
            return 0.5  # uninformative prior

        sims = _top_k_cosine(q_emb, bank, k=len(bank))
        # Softmax
        shifted = sims - sims.max()
        exp = np.exp(shifted)
        probs = exp / (exp.sum() + 1e-12)
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        max_entropy = math.log(len(bank))
        return float(np.clip(entropy / (max_entropy + 1e-9), 0.0, 1.0))

    def semantic_complexity(self, text: str, q_emb: np.ndarray) -> float:
        """Lexical complexity score in [0, 1].

        Combines:
        - Token count (log-scaled)
        - Mean sentence length
        - Type-token ratio (vocabulary richness)
        - Rare-word ratio (words > 8 chars as proxy)
        """
        tokens = re.findall(r"\b\w+\b", text.lower())
        n_tokens = len(tokens)
        if n_tokens == 0:
            return 0.0

        sentences = re.split(r"[.!?]+", text.strip())
        sentences = [s for s in sentences if s.strip()]
        mean_sent_len = (
            np.mean([len(re.findall(r"\b\w+\b", s)) for s in sentences])
            if sentences
            else n_tokens
        )

        type_token_ratio = len(set(tokens)) / n_tokens
        rare_ratio = sum(1 for t in tokens if len(t) > 8) / n_tokens

        # Log-scale token count, saturate at 512
        token_score = min(math.log1p(n_tokens) / math.log1p(512), 1.0)
        sent_score = min(mean_sent_len / 40.0, 1.0)

        score = 0.3 * token_score + 0.2 * sent_score + 0.25 * type_token_ratio + 0.25 * rare_ratio
        return float(np.clip(score, 0.0, 1.0))

    def retrieval_ambiguity(
        self,
        q_emb: np.ndarray,
        ctx_embs: np.ndarray | None,
        k: int = 5,
    ) -> float:
        """1 - mean top-k cosine similarity to retrieved context chunks.

        High ambiguity ≈ retrieved docs are semantically distant from query.
        Returns 0.5 when no context is provided.
        """
        if ctx_embs is None or len(ctx_embs) == 0:
            return 0.5
        top_sims = _top_k_cosine(q_emb, ctx_embs, k=k)
        return float(np.clip(1.0 - top_sims.mean(), 0.0, 1.0))

    def reasoning_depth(self, text: str, q_emb: np.ndarray) -> float:
        """Estimate reasoning depth in [0, 1].

        Combines:
        - Presence of reasoning cue words/phrases
        - Density of numeric / mathematical operators
        - Cosine similarity to reasoning prototype sentences
        """
        # Cue word score
        cue_matches = len(_REASONING_RE.findall(text))
        word_count = max(len(text.split()), 1)
        cue_score = min(cue_matches / max(word_count * 0.1, 1), 1.0)

        # Numeric/operator density
        op_count = len(_NUMERIC_OP_RE.findall(text))
        op_score = min(op_count / max(word_count, 1), 1.0)

        # Prototype similarity
        proto_embs = self._get_proto_embs()
        sim_scores = np.array([_cosine_sim(q_emb, p) for p in proto_embs])
        proto_score = float(np.clip(sim_scores.max(), 0.0, 1.0))

        score = 0.35 * cue_score + 0.25 * op_score + 0.40 * proto_score
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Cheap lexical features (supplement embeddings)
    # ------------------------------------------------------------------

    def _lexical_features(self, text: str) -> np.ndarray:
        """Return a small array of cheap lexical features."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        n = len(tokens)
        has_question = float("?" in text)
        has_code = float(bool(re.search(r"```|`[^`]+`|def |class |import ", text)))
        has_math = float(bool(re.search(r"\$.*?\$|\\[a-z]+\{", text)))
        avg_word_len = (
            np.mean([len(t) for t in tokens]) if tokens else 0.0
        )
        return np.array(
            [
                min(n / 200.0, 1.0),   # normalised length
                has_question,
                has_code,
                has_math,
                min(avg_word_len / 12.0, 1.0),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Main vectorize method
    # ------------------------------------------------------------------

    def vectorize(self, query: Query) -> np.ndarray:
        """Encode a :class:`~slm_router.types.Query` into a feature vector.

        Returns
        -------
        np.ndarray
            Shape ``(9,)`` float32 array:
            [embedding_entropy, semantic_complexity, retrieval_ambiguity,
             reasoning_depth, norm_len, has_question, has_code, has_math,
             avg_word_len_norm]
        """
        text = query.text
        q_emb = self.embedder.encode(text)

        ctx_embs: np.ndarray | None = None
        if hasattr(query, "context_chunks") and query.context_chunks:
            ctx_embs = self.embedder.encode(query.context_chunks)

        f_entropy = self.embedding_entropy(q_emb)
        f_complexity = self.semantic_complexity(text, q_emb)
        f_ambiguity = self.retrieval_ambiguity(q_emb, ctx_embs)
        f_reasoning = self.reasoning_depth(text, q_emb)
        f_lexical = self._lexical_features(text)

        feats = np.concatenate(
            [
                np.array(
                    [f_entropy, f_complexity, f_ambiguity, f_reasoning],
                    dtype=np.float32,
                ),
                f_lexical,
            ]
        )
        return feats  # shape (9,)

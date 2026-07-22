"""Judge reliability metrics: human agreement, self-consistency, position bias."""
from __future__ import annotations

import json
import re

try:
    from scipy import stats as _scipy_stats  # type: ignore[import]
    _HAS_SCIPY = True
except ImportError:
    _scipy_stats = None  # type: ignore[assignment]
    _HAS_SCIPY = False

_BINARIZE_THRESHOLD = 0.6


def _binarize(scores: list[float], threshold: float = _BINARIZE_THRESHOLD) -> list[int]:
    """Convert continuous scores to binary labels using *threshold*."""
    return [1 if s >= threshold else 0 for s in scores]


def judge_human_agreement(
    judge_scores: list[float],
    human_scores: list[float],
) -> dict[str, float]:
    """Measure agreement between LLM judge scores and human ratings.

    Args:
        judge_scores: Normalised judge scores in [0, 1].
        human_scores: Normalised human scores in [0, 1].

    Returns:
        Dict with keys:
            - cohen_kappa: Cohen's kappa on binarised scores (threshold=0.6).
            - spearman: Spearman rank correlation.
            - pct_agree: Percentage agreement on binarised labels.

    Raises:
        ImportError: If scipy is not installed.
        ValueError: If lists have different lengths or are empty.
    """
    if not _HAS_SCIPY:
        raise ImportError(
            "scipy is required for judge_human_agreement. "
            "Install it with: pip install scipy"
        )

    if len(judge_scores) != len(human_scores):
        raise ValueError(
            f"judge_scores and human_scores must have equal length, "
            f"got {len(judge_scores)} vs {len(human_scores)}"
        )
    if not judge_scores:
        raise ValueError("Score lists must not be empty")

    # Binarise for kappa and percent agreement
    judge_bin = _binarize(judge_scores)
    human_bin = _binarize(human_scores)

    # Percent agreement
    agreements = sum(j == h for j, h in zip(judge_bin, human_bin))
    pct_agree = agreements / len(judge_bin)

    # Cohen's kappa
    try:
        from sklearn.metrics import cohen_kappa_score  # type: ignore[import]
        kappa = float(cohen_kappa_score(human_bin, judge_bin))
    except ImportError:
        # Manual Cohen's kappa implementation
        n = len(judge_bin)
        po = pct_agree
        # Expected agreement
        p_judge_pos = sum(judge_bin) / n
        p_human_pos = sum(human_bin) / n
        pe = p_judge_pos * p_human_pos + (1 - p_judge_pos) * (1 - p_human_pos)
        kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

    # Spearman correlation
    spearman_result = _scipy_stats.spearmanr(judge_scores, human_scores)
    spearman = float(spearman_result.correlation)  # type: ignore[attr-defined]

    return {
        "cohen_kappa": kappa,
        "spearman": spearman,
        "pct_agree": pct_agree,
    }


def self_consistency(scores_list: list[list[float]]) -> float:
    """Measure self-consistency of a judge across repeated runs.

    For each query position, compute the std-dev of scores across runs.
    Return the mean std-dev (lower = more consistent).

    Args:
        scores_list: List of score lists; each inner list is one judge run.
            All inner lists must have the same length.

    Returns:
        Mean standard deviation across positions. 0.0 if fewer than 2 runs.
    """
    if len(scores_list) < 2:
        return 0.0

    n_items = len(scores_list[0])
    if n_items == 0:
        return 0.0

    try:
        import statistics

        stds: list[float] = []
        for i in range(n_items):
            col = [run[i] for run in scores_list if i < len(run)]
            if len(col) >= 2:
                stds.append(statistics.stdev(col))
            else:
                stds.append(0.0)
        return sum(stds) / len(stds) if stds else 0.0
    except Exception:
        return 0.0


_PREF_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
_PREF_LETTER_RE = re.compile(r"\b(A|B)\b")


def _parse_preference(text: str) -> str | None:
    """Extract the judge's preferred option ("A" or "B") from *text*.

    Tries to parse a JSON object with a "preferred"/"winner"/"choice" key
    first, then falls back to scanning for a standalone "A" or "B" token.

    Returns:
        "A", "B", or None if no preference could be determined.
    """
    stripped = text.strip()

    # Try direct JSON parse, then embedded JSON object.
    candidates: list[dict] = []
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            candidates.append(parsed)
    except (json.JSONDecodeError, ValueError):
        pass

    if not candidates:
        match = _PREF_JSON_RE.search(stripped)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, dict):
                    candidates.append(parsed)
            except (json.JSONDecodeError, ValueError):
                pass

    for parsed in candidates:
        for key in ("preferred", "winner", "choice", "preference"):
            value = parsed.get(key)
            if isinstance(value, str):
                value = value.strip().upper()
                if value in ("A", "B"):
                    return value

    # Fallback: look for a standalone "A" or "B" token in the raw text.
    letter_match = _PREF_LETTER_RE.search(stripped.upper())
    if letter_match:
        return letter_match.group(1)

    return None


def position_bias_check(
    responses_ab: list[str],
    responses_ba: list[str],
) -> dict[str, object]:
    """Detect position bias by comparing A-B vs B-A judge preference rates.

    The same pair of underlying responses is judged twice: once with the
    first response shown first ("AB" ordering) and once with it shown
    second ("BA" ordering). In each judge response, "A" denotes whichever
    response was presented first and "B" denotes whichever was presented
    second (i.e. the labels are positional, not tied to a fixed response
    identity). If the judge's *underlying* preferred response flips
    between orderings, that flip is attributable to position rather than
    content, which is the signature of position bias.

    Args:
        responses_ab: Judge responses when the first response is shown first.
        responses_ba: Judge responses when the first response is shown second
            (i.e. the second response is shown first). Must be paired
            index-for-index with responses_ab (same underlying response pair).

    Returns:
        Dict with keys:
            - prefer_first_ab: fraction of AB-ordering judgments that preferred
              the first-shown option (None if no judgments were parseable).
            - prefer_first_ba: fraction of BA-ordering judgments that preferred
              the first-shown option (None if no judgments were parseable).
            - flip_rate: fraction of paired judgments where the underlying
              preferred response flipped between orderings (the position-bias
              score, in [0, 1]; None if no valid pairs).
            - n_pairs: number of items with a parseable preference in both
              orderings (used to compute flip_rate).
            - n_flipped: number of pairs where the preference flipped.
            - position_bias_detected: bool, True if flip_rate exceeds 0.3.
            - note: explanation string.

    Raises:
        ValueError: If responses_ab and responses_ba have different lengths.
    """
    if len(responses_ab) != len(responses_ba):
        raise ValueError(
            f"responses_ab and responses_ba must have equal length, "
            f"got {len(responses_ab)} vs {len(responses_ba)}"
        )

    prefs_ab = [_parse_preference(r) for r in responses_ab]
    prefs_ba = [_parse_preference(r) for r in responses_ba]

    valid_ab = [p for p in prefs_ab if p is not None]
    valid_ba = [p for p in prefs_ba if p is not None]

    prefer_first_ab = (
        sum(1 for p in valid_ab if p == "A") / len(valid_ab) if valid_ab else None
    )
    prefer_first_ba = (
        sum(1 for p in valid_ba if p == "A") / len(valid_ba) if valid_ba else None
    )

    # Map each ordering's positional label back to the underlying response
    # identity: in AB order, "A" = response 1, "B" = response 2; in BA
    # order, "A" = response 2, "B" = response 1.
    _flip = {"A": "B", "B": "A"}
    n_flipped = 0
    n_pairs = 0
    for pref_ab, pref_ba in zip(prefs_ab, prefs_ba):
        if pref_ab is None or pref_ba is None:
            continue
        n_pairs += 1
        actual_pref_ba = _flip[pref_ba]
        if pref_ab != actual_pref_ba:
            n_flipped += 1

    flip_rate = n_flipped / n_pairs if n_pairs else None
    position_bias_detected = flip_rate is not None and flip_rate > 0.3

    return {
        "prefer_first_ab": prefer_first_ab,
        "prefer_first_ba": prefer_first_ba,
        "flip_rate": flip_rate,
        "n_pairs": n_pairs,
        "n_flipped": n_flipped,
        "position_bias_detected": position_bias_detected,
        "note": (
            "flip_rate is the position-bias score in [0, 1]: the fraction of "
            "paired judgments whose preferred underlying response changed "
            "when its position was swapped. position_bias_detected flags "
            "flip_rate > 0.3."
        ),
    }

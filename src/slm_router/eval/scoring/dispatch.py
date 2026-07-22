from __future__ import annotations

from slm_router.eval.datasets.schema import EvalItem, TaskType
from slm_router.eval.scoring.base import Scorer, ScoreResult


def get_scorer(task_type: TaskType) -> Scorer | None:
    """Return the appropriate Scorer instance for the given task type.

    Returns:
        A Scorer instance, or None for OPEN_ENDED (requires an LLM judge).

    Raises:
        ValueError: If the task_type is unrecognised.
    """
    if task_type == TaskType.MCQ:
        from slm_router.eval.scoring.exact_match import ExactMatchScorer
        return ExactMatchScorer()

    if task_type == TaskType.MULTIHOP_QA:
        from slm_router.eval.scoring.exact_match import TokenF1Scorer
        return TokenF1Scorer()

    if task_type == TaskType.MATH:
        from slm_router.eval.scoring.numeric import NumericScorer
        return NumericScorer()

    if task_type == TaskType.CODE:
        from slm_router.eval.scoring.code_exec import CodeExecScorer
        return CodeExecScorer()

    if task_type == TaskType.OPEN_ENDED:
        # Requires an async LLM judge; caller must use LLMJudgeScorer directly.
        return None

    raise ValueError(f"Unknown task_type: {task_type!r}")


def score_response(
    item: EvalItem,
    model_output: str,
    **kwargs: object,
) -> ScoreResult:
    """Score a model output for the given EvalItem.

    Routes to the correct scorer based on item.task_type.

    Args:
        item: The EvalItem (contains task_type and gold_answer).
        model_output: Raw string output from the model.
        **kwargs: Ignored; reserved for future extensibility.

    Returns:
        ScoreResult with score in [0.0, 1.0].

    Raises:
        ValueError: If task_type is OPEN_ENDED (needs LLM judge, not synchronous).
    """
    scorer = get_scorer(item.task_type)

    if scorer is None:
        return ScoreResult(
            score=0.0,
            error=(
                "OPEN_ENDED tasks require an LLM judge. "
                "Use LLMJudgeScorer.score() (async) instead."
            ),
        )

    return scorer.score(item, model_output)

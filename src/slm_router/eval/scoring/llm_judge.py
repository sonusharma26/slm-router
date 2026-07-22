from __future__ import annotations

import json
import re
from typing import Any, TYPE_CHECKING

from slm_router.eval.scoring.base import ScoreResult

if TYPE_CHECKING:
    from slm_router.eval.datasets.schema import EvalItem

_SYSTEM_PROMPT = (
    "You are an expert evaluator. Given a question and model response, "
    "rate the response on a scale of 1-5. "
    'Respond ONLY with valid JSON: {"score": <1-5>, "reasoning": "..."}'
)

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_judge_response(text: str) -> dict[str, Any] | None:
    """Try to parse a JSON object from judge response text."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find embedded JSON object
    match = _JSON_RE.search(text)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    return None


class LLMJudgeScorer:
    """Async LLM-as-judge scorer for open-ended responses.

    Uses an external LLM client to rate responses on a 1-5 scale,
    which is normalised to [0.0, 1.0].

    Args:
        client: An async LLM client with a `.complete(model, messages, **kwargs)` method.
        judge_model: Model ID to use as the judge.
        temperature: Sampling temperature (default 0.0 for determinism).
    """

    def __init__(
        self,
        client: Any,
        judge_model: str,
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.judge_model = judge_model
        self.temperature = temperature

    async def score(self, item: "EvalItem", model_output: str) -> ScoreResult:
        """Score *model_output* for *item* using an LLM judge.

        Retries once on JSON parse failure.

        Returns:
            ScoreResult with score in [0.0, 1.0] (raw 1-5 divided by 5).
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{item.query}\n\n"
                    f"Model Response:\n{model_output}"
                ),
            },
        ]

        raw_text: str | None = None
        parsed: dict[str, Any] | None = None

        for attempt in range(2):  # Try twice
            try:
                response = await self.client.complete(
                    self.judge_model,
                    messages,
                    temperature=self.temperature,
                    max_tokens=256,
                )
                # Support both attribute and dict access for different client shapes
                if hasattr(response, "content"):
                    raw_text = str(response.content)
                elif isinstance(response, dict):
                    raw_text = str(response.get("content", ""))
                else:
                    raw_text = str(response)

                parsed = _parse_judge_response(raw_text)
                if parsed is not None:
                    break
            except Exception as exc:
                if attempt == 1:
                    return ScoreResult(
                        score=0.0,
                        error=f"LLM judge call failed: {exc}",
                    )

        if parsed is None:
            return ScoreResult(
                score=0.0,
                error=f"Could not parse judge JSON after 2 attempts. Raw: {raw_text!r}",
            )

        raw_score = parsed.get("score")
        reasoning = str(parsed.get("reasoning", ""))

        try:
            numeric = float(raw_score)
        except (TypeError, ValueError):
            return ScoreResult(
                score=0.0,
                error=f"Judge returned non-numeric score: {raw_score!r}",
            )

        # Clamp to [1, 5] then normalise to [0, 1]
        numeric = max(1.0, min(5.0, numeric))
        normalised = (numeric - 1.0) / 4.0  # maps 1->0.0, 5->1.0

        return ScoreResult(
            score=normalised,
            sub_scores={
                "raw_score": numeric,
                "reasoning": reasoning,
            },
        )

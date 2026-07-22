from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from typing import TYPE_CHECKING

from slm_router.eval.scoring.base import ScoreResult, Scorer

if TYPE_CHECKING:
    from slm_router.eval.datasets.schema import EvalItem

# ---------------------------------------------------------------------------
# Code extraction helpers
# ---------------------------------------------------------------------------

# Matches ```python ... ``` or ``` ... ```
_CODE_BLOCK_RE = re.compile(
    r"```(?:python)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def _extract_code_block(text: str) -> str | None:
    """Extract the first fenced code block from model output.

    Returns the raw code string, or None if no code block found.
    """
    match = _CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1)
    # Fallback: return everything if it looks like bare code
    stripped = text.strip()
    if stripped.startswith("def ") or stripped.startswith("class "):
        return stripped
    return None


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class CodeExecScorer(Scorer):
    """HumanEval scorer: execute generated code in a subprocess (pass@1).

    Assembles:
        reference['prompt'] + extracted_code + "\\n" + reference['test']
        + "\\ncheck(" + reference['entry_point'] + ")"

    Runs the assembled script via a child Python process with a 10-second timeout.
    Never exec()s code in-process.  Cross-platform / Windows-friendly.
    """

    TIMEOUT: float = 10.0

    def score(self, item: "EvalItem", model_output: str) -> ScoreResult:
        # Validate reference
        if item.reference is None:
            return ScoreResult(
                score=0.0,
                correct=False,
                error="EvalItem has no reference; cannot execute code",
            )

        prompt: str = item.reference.get("prompt", "")
        test: str = item.reference.get("test", "")
        entry_point: str = item.reference.get("entry_point", "")

        if not test or not entry_point:
            return ScoreResult(
                score=0.0,
                correct=False,
                error="reference missing 'test' or 'entry_point' fields",
            )

        # Extract code from model output
        extracted_code = _extract_code_block(model_output)
        if extracted_code is None:
            return ScoreResult(
                score=0.0,
                correct=False,
                error="No code block found in model output",
            )

        # Assemble the full script
        script = (
            f"{prompt}\n"
            f"{extracted_code}\n"
            f"{test}\n"
            f"check({entry_point})\n"
        )

        # Write to a temporary file and run it
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8",
            ) as tmp_file:
                tmp_file.write(script)
                tmp_path = tmp_file.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                timeout=self.TIMEOUT,
                capture_output=True,
                text=True,
            )

            passed = result.returncode == 0
            return ScoreResult(
                score=1.0 if passed else 0.0,
                correct=passed,
                sub_scores={
                    "stdout": result.stdout[:2000],  # Truncate long output
                    "stderr": result.stderr[:2000],
                    "returncode": result.returncode,
                },
            )

        except subprocess.TimeoutExpired:
            return ScoreResult(
                score=0.0,
                correct=False,
                sub_scores={"stdout": "", "stderr": ""},
                error=f"Code execution timed out after {self.TIMEOUT}s",
            )
        except Exception as exc:
            return ScoreResult(
                score=0.0,
                correct=False,
                error=f"Unexpected error during code execution: {exc}",
            )
        finally:
            # Clean up temp file
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

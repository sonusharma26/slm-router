"""Local answer agreement between two completed calls.

``answer_agreement`` is the built-in agreement function for ``agree_escalate`` plans. It reduces each
response to a final answer and compares the normalized forms (``routing.cascade.normalize_answer``),
the same comparison the offline cascade calibration applies to stored extracted answers. It is a
sanity check on comparable answers (numbers, choices, short strings), not a correctness oracle:
free-form responses rarely agree, so the cascade then escalates.
"""
from __future__ import annotations

import re

from inference_control.routing.cascade import normalize_answer

_ANSWER_LINE = re.compile(r"^[\s>#*_`-]*(?:the\s+)?(?:final\s+answer|answer)\s*(?:is\s*)?[:：]\s*(.+?)\s*$",
                          re.IGNORECASE | re.MULTILINE)
_MARKUP = " \t*_`"


def _boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` with balanced braces, or None."""
    marker = "\\boxed{"
    start = text.rfind(marker)
    while start >= 0:
        depth, index = 1, start + len(marker)
        while index < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[index], 0)
            index += 1
        if depth == 0:
            return text[start + len(marker):index - 1]
        start = text.rfind(marker, 0, start)
    return None


def extract_final_answer(text) -> str | None:
    """``\\boxed{...}``, else the last ``final answer:`` / ``answer:`` line, else the last non-empty line."""
    if not isinstance(text, str):
        return None
    boxed = _boxed(text)
    if boxed is not None and boxed.strip():
        return boxed.strip()
    matches = _ANSWER_LINE.findall(text)
    if matches:
        answer = matches[-1].strip(_MARKUP)
        if answer:
            return answer
    lines = [line.strip(_MARKUP) for line in text.splitlines() if line.strip(_MARKUP)]
    return lines[-1] if lines else None


def message_text(result) -> str | None:
    """Text content of a RuntimeResult message (plain string or text parts); None for tool-only output."""
    message = getattr(result, "message", None) or {}
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = [p.get("text") for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)]
        return "\n".join(parts) if parts else None
    return None


def result_answer(result) -> str | None:
    """Normalized comparable answer of one result, or None when there is nothing to compare."""
    return normalize_answer(extract_final_answer(message_text(result)))


def answer_agreement(first, checker) -> bool:
    """True when both results carry a comparable answer and the normalized answers are equal."""
    a, b = result_answer(first), result_answer(checker)
    return a is not None and a == b

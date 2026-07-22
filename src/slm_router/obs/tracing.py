"""Vendor-neutral tracing seam.

Core modules depend only on the ``Tracer`` protocol via ``get_tracer()``; the
concrete OTel/Langfuse wiring is isolated here and disabled by default
(``SLM_OBS_ENABLED=false``), so the platform runs with tracing fully off and
observability never sits on the critical path.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Protocol


class Tracer(Protocol):
    def span(self, name: str, **attrs: Any) -> Any: ...


class NoopTracer:
    """Null tracer used when observability is disabled."""

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[None]:
        yield None


class _OTelTracer:
    """OpenTelemetry-backed tracer (exports OTLP to Langfuse/Phoenix when configured)."""

    def __init__(self) -> None:
        from opentelemetry import trace  # defensive: only imported when enabled

        self._tracer = trace.get_tracer("slm_router")

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Any]:
        with self._tracer.start_as_current_span(name) as sp:
            for k, v in attrs.items():
                try:
                    sp.set_attribute(k, v)
                except Exception:
                    pass
            yield sp


_TRACER: Tracer | None = None


def get_tracer(enabled: bool | None = None) -> Tracer:
    """Return a process-wide tracer. Falls back to NoopTracer on any error."""
    global _TRACER
    if _TRACER is not None:
        return _TRACER
    if not enabled:
        _TRACER = NoopTracer()
        return _TRACER
    try:
        _TRACER = _OTelTracer()
    except Exception:
        _TRACER = NoopTracer()
    return _TRACER

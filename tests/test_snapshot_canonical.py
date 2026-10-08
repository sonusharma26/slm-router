"""Cached snapshot serialization must keep ledger bytes (and so hashes) unchanged."""
import json
from datetime import datetime, timezone

import pytest

from inference_control.benchmarks.routers import SLMRouter
from inference_control.benchmarks.runner import split_data
from inference_control.benchmarks.synthetic import make_dataset
from inference_control.ledger import SQLiteLedger
from inference_control.util import JsonNative, canonical, primitive


def _fitted_map():
    data = make_dataset(300, 7)
    fit, rows = split_data(data, {"train", "calibration"})
    router = SLMRouter(data.endpoints, k=64, min_samples=8, at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    router.fit(fit, rows)
    return router.planner.capability_map


def test_snapshot_bytes_match_uncached_json_dump():
    cmap = _fitted_map()
    cached = cmap.export()
    plain = dict(cached, rows=[cmap.rows[k].model_dump(mode="json") for k in sorted(cmap.rows)])
    assert all(type(row) is JsonNative for row in cached["rows"])
    assert canonical({"data": cached}) == canonical({"data": plain})
    # A second export reuses the cached dumps and stays identical.
    assert canonical(cmap.export()) == canonical(plain)


def test_cached_row_dump_tracks_replaced_rows():
    cmap = _fitted_map()
    sample_id = sorted(cmap.rows)[0]
    before = cmap.export()["rows"][0]
    cmap.rows[sample_id] = cmap.rows[sample_id].model_copy(update={"disputed": True, "trusted": False})
    cmap._version = cmap._index = None
    after = cmap.export()["rows"][0]
    assert before["disputed"] is False and after["disputed"] is True


def test_primitive_fast_paths_preserve_semantics():
    assert primitive({2: (1, 2.5), "a": [None, True]}) == {"2": [1, 2.5], "a": [None, True]}
    assert json.dumps(primitive({"b": 1, "a": {"d": 2, "c": 3}})) == '{"a": {"c": 3, "d": 2}, "b": 1}'
    with pytest.raises(ValueError):
        primitive({"x": [float("inf")]})


def test_filtered_events_and_last_sequence():
    ledger = SQLiteLedger(":memory:")
    for index, kind in enumerate(("capability_map", "execution", "latency_receipt", "capability_map")):
        ledger.append(kind, {"i": index}, actor="test", idempotency_key=f"k{index}")
    picked = ledger.events(event_types=("execution", "latency_receipt"))
    assert [event.event_type for event in picked] == ["execution", "latency_receipt"]
    assert ledger.events(after=2, event_types=("execution", "latency_receipt"))[0].sequence == 3
    assert ledger.last_sequence() == 4
    assert ledger.verify()

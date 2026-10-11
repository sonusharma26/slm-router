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


def test_spliced_encoding_matches_plain_json():
    rows = [JsonNative({"b": [1, 2.5, None], "a": {"z": True, "y": "é"}}), JsonNative({"k": 0.1})]
    tree = {"version": "v", "data": {"rows": rows, "n": 2, "list": [[1, "x"], {"q": 1}]}}
    plain = {"version": "v", "data": {"rows": [dict(r) for r in rows], "n": 2, "list": [[1, "x"], {"q": 1}]}}
    expected = json.dumps(plain, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert canonical(tree) == expected
    assert canonical(tree) == expected  # second call uses cached row encodings


def test_event_payload_is_lazy_and_equal_to_stored_json():
    ledger = SQLiteLedger(":memory:")
    event = ledger.append("snapshot", {"rows": [{"b": 1, "a": 2}]}, actor="test", idempotency_key="s")
    assert "_payload" not in event.__dict__
    assert event.payload == {"rows": [{"a": 2, "b": 1}]}
    assert event == ledger.get(event.event_id) and hash(event) == hash(ledger.get(event.event_id))
    assert ledger.verify()

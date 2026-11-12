import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))
from commons_prepare import select


def test_select_applies_threshold_and_drops_duplicate_sha1():
    rel = [{"file": "a", "score": "0.9"}, {"file": "b", "score": "0.95"}, {"file": "c", "score": "0.5"},
           {"file": "d", "score": "0.99"}]
    meta = {"a": {"sha1": "x"}, "b": {"sha1": "x"}, "c": {"sha1": "y"}}  # d has no metadata
    assert [r["file"] for r, _ in select(rel, meta, 0.85)] == ["a"]

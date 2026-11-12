import sys
from pathlib import Path

import torch  # noqa: F401  (Windows: torch's DLLs must load before pandas, which clean_dataset imports)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))
from clean_dataset import classify, dedupe


def row(i, sha, ph, w=500, h=500, size=100, meta_n=3):
    return dict(id=i, sha=sha, ph=ph, w=w, h=h, size=size, meta_n=meta_n)


def test_dedupe_keeps_best_quality_exact_and_near():
    kept, dups = dedupe([row("a", "s1", 0b0), row("b", "s1", 0b0, w=900, h=900), row("c", "s2", 0b11, size=50), row("d", "s3", (1 << 48) - 1)])
    assert [r["id"] for r in kept] == ["b", "d"]  # b is the larger of the exact pair; c is near a/b; d is far
    assert sorted((d[0]["id"], d[1]["id"], d[2]) for d in dups) == [("a", "b", "exact"), ("c", "b", "near")]


def test_confirm_blocks_phash_collisions_of_different_content():
    a, b = row("a", "s1", 0), row("b", "s2", 0)  # identical pHash, different pictures
    kept, dups = dedupe([a, b], confirm=lambda x, y: False)
    assert len(kept) == 2 and dups == []
    kept, dups = dedupe([a, b], confirm=lambda x, y: True)
    assert len(kept) == 1 and len(dups) == 1


def test_classify_order_of_checks():
    base = dict(format="jpeg", loadable=True, short_side=500, has_meta=True, licence="CC BY 2.0", class_meta="oil filter", class_folder="oil filter", image_id="x")
    ids = {"x": 1}
    assert classify(base, ids) == ("ok", "")
    assert classify({**base, "licence": "CC BY-SA 4.0"}, ids)[0] == "excluded"
    assert classify({**base, "licence": ""}, ids) == ("needs_review", "licence_missing")
    assert classify({**base, "short_side": 100}, ids) == ("rejected", "rejected_low_resolution")
    assert classify({**base, "format": "gif"}, ids)[0] == "rejected"

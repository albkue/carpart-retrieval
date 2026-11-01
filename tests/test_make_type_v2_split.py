import random
import sys
from pathlib import Path

import torch  # noqa: F401  (Windows: torch's DLLs must load before pandas, which make_type_split imports)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))
from make_type_v2_split import (
    HELDOUT,
    HITL_MAP,
    SYNTH_MAP,
    cap_side,
    check_leakage,
    split_photos,
)


def _r(i, src, label="door", group=""):
    return {"image": f"{src}{i}.jpg", "part_id": label, "src": src, "group": group}


def test_cap_side_drops_hitl_first_but_keeps_a_third():
    items = [_r(i, "kaggle50") for i in range(5)] + [_r(i, "hitl", group=f"g{i}") for i in range(50)]
    out = cap_side(items, 20, random.Random(0))
    assert len(out) == 20 and sum(r["src"] == "kaggle50" for r in out) == 5
    web = [_r(i, "kaggle50") for i in range(200)]
    out = cap_side(web + [_r(i, "hitl", group=f"g{i}") for i in range(50)], 100, random.Random(0))
    assert len(out) == 100 and sum(r["src"] == "hitl" for r in out) == 33  # web photos would otherwise fill the cap
    assert len(cap_side(web[:10], 100, random.Random(0))) == 10


def test_split_photos_fraction_and_determinism():
    g = [f"p{i}" for i in range(100)] * 3
    a, b = split_photos(g, random.Random(1), 0.2), split_photos(g, random.Random(1), 0.2)
    assert a == b and len(a) == 20


def test_leakage_checks_catch_each_failure():
    ok = [ok for _, ok in check_leakage([_r(1, "hitl", group="a")], [_r(2, "hitl", group="b")], [_r(3, "kaggle50")], HELDOUT)]
    assert all(ok)
    same_photo = check_leakage([_r(1, "hitl", group="a")], [_r(2, "hitl", group="a")], [], HELDOUT)
    assert not same_photo[1][1]
    held_in_train = check_leakage([_r(1, "kaggle50", "camshaft")], [], [], HELDOUT)
    assert not held_in_train[2][1]
    synth_eval = check_leakage([], [_r(1, "synthetic")], [], HELDOUT)
    assert not synth_eval[3][1]


def test_label_maps_keep_mirror_heldout_and_drop_unpaired():
    assert HITL_MAP["mirror"] == "side mirror" and "side mirror" in HELDOUT
    assert "mirror" not in SYNTH_MAP and "air_filter" not in SYNTH_MAP and "turbocharger" not in SYNTH_MAP

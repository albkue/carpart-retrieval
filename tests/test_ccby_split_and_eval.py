"""Split integrity and metric sanity for the Sunday experiment scripts (no GPU, no model download)."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load(rel):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


split = load("scripts/data/make_ccby_split.py")
ev = load("scripts/eval_checkpoint.py")


def test_groups_land_in_exactly_one_partition():
    ids = [f"g{i}" for i in range(100)]
    train, valid, far = split.split_groups(ids, 0.15, 5, 42)
    assert not (train & valid) and not (train & far) and not (valid & far)
    assert len(train | valid | far) == 100 and len(far) == 5 and len(valid) == 14


def test_content_check_separates_small_objects_on_white():
    from PIL import Image, ImageDraw
    def img(shape):
        im = Image.new("RGB", (640, 640), "white")
        d = ImageDraw.Draw(im)
        (d.ellipse if shape == "round" else d.rectangle)((300, 300, 340, 340), fill="black")
        return im
    a, b = split.thumb(img("round")), split.thumb(img("square"))
    assert split.same_content(a, split.thumb(img("round"))) and not split.same_content(a, b)


def test_perfect_retrieval_scores_one_and_wrong_scores_zero():
    gallery_parts = ["a", "b", "c"]
    sim = np.eye(3)
    rows = ev.score(sim, gallery_parts, ["a", "b", "c"])
    assert rows["recall@1"].mean() == 1.0 and rows["mAP"].mean() == 1.0
    bad = ev.score(1 - np.eye(3), gallery_parts, ["a", "b", "c"], ks=(1,))
    assert bad["recall@1"].mean() == 0.0

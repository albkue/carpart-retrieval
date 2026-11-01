#!/usr/bin/env python3
"""Tuning split for A5, carved from the TRAINING images only, so hyperparameters never see the test queries.

The 859 seen-type queries in splits_type_v1_eval.json are the test set; tuning temperature, learning rates,
the WiSE-FT alpha or the abstention threshold on them would leak. Per seen type, 10% of its training images
(at least 4) become a tuning eval set (half gallery, half queries); the rest is the tuning train set.
The original split files are not modified. Held-out types stay out of tuning entirely.

Writes E:\\car_search_data\\eda\\type_v1\\tune\\{splits_type_v1_tune_train.json, splits_type_v1_tune_eval.json}.
Usage: python scripts/data/make_type_tune_split.py [--version v2]   (default v1; reads eda/type_<version>)
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SPLIT = Path(r"E:\car_search_data\eda\type_v1")


def carve(train_rows: list, frac=0.1, min_eval=4, seed=42):
    """-> (tune_train, tune_gallery, tune_queries), disjoint, per type."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in train_rows:
        by[r["part_id"]].append(r["image"])
    tr, ga, qu = [], [], []
    for t in sorted(by):
        imgs = sorted(by[t])
        rng.shuffle(imgs)
        n = max(min_eval, round(len(imgs) * frac))
        ev, rest = imgs[:n], imgs[n:]
        half = len(ev) // 2
        qu += [{"image": i, "part_id": t} for i in ev[:half]]
        ga += [{"image": i, "part_id": t} for i in ev[half:]]
        tr += [{"image": i, "part_id": t} for i in rest]
    return tr, ga, qu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    v = ap.parse_args().version
    split = SPLIT.with_name(f"type_{v}")
    src = json.loads((split / f"splits_type_{v}_train.json").read_text(encoding="utf-8"))
    tr, ga, qu = carve(src["catalog"], seed=src.get("seed", 42))
    out = split / "tune"
    out.mkdir(exist_ok=True)
    note = f"tuning split carved from splits_type_{v}_train.json (10% per type); never use for final results"
    (out / f"splits_type_{v}_tune_train.json").write_text(json.dumps(
        {"provisional": True, "seed": 42, "note": note, "catalog": tr, "heldout_part_ids": src.get("heldout_part_ids", [])}, indent=1), encoding="utf-8")
    (out / f"splits_type_{v}_tune_eval.json").write_text(json.dumps(
        {"provisional": True, "seed": 42, "note": note, "catalog": ga, "queries": {"valid": qu}}, indent=1), encoding="utf-8")
    print(f"tune train {len(tr)}, tune gallery {len(ga)}, tune queries {len(qu)} -> {out}")


if __name__ == "__main__":
    main()

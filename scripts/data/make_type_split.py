#!/usr/bin/env python3
"""Type-level train/eval split of the car-part images (Kaggle 50 + Kaggle 40 + Commons) for the type experiment.

Labels are part TYPES, not part numbers: this answers "does fine-tuning teach CLIP car parts?", not RQ2.
Inputs are the cleaned manifests (clean_kaggle50.py for both Kaggle versions, clean_dataset.py for Commons).
- Commons rows reviewed as untrustworthy are dropped (water pump, fuel pump, brake disc: see combine_clean.py).
- Near duplicates across all sources (pHash <= 6 confirmed by a content thumbnail) are grouped: a group with one
  label keeps one image (Kaggle 50 first), a group with conflicting labels is dropped.
- A few whole types are held out of training (heldout_part_ids) to measure generalisation to unseen types.
- Per seen type: 80% train, 20% eval; eval splits into gallery and queries (at most --max-queries per type).
Writes E:\\car_search_data\\eda\\type_v1\\{splits_type_v1_train.json, splits_type_v1_eval.json, report.md}.
Eval query keys: "valid" (seen types) and "heldout" (unseen types, gallery includes their images).

Usage: python scripts/data/make_type_split.py
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_ccby_split import UnionFind, same_content, thumb

E = Path(r"E:\car_search_data")
SOURCES = [("kaggle50", E / "dataset_kaggle50/metadata/kaggle50_clean.csv", "label"),
           ("kaggle40", E / "dataset_kaggle40/metadata/kaggle40_clean.csv", "label"),
           ("commons", E / "dataset_clean/metadata/clean_images.csv", "class")]
COMMONS_DROP = {"water pump", "fuel pump", "brake disc"}  # label review in combine_clean.py
COMMONS_ALIAS = {"clutch disc": "clutch plate"}  # same part, Kaggle's name
OUT = E / "eda/type_v1"


def near_groups(ph: np.ndarray, ths: list, max_dist=6) -> UnionFind:
    uf = UnionFind()
    for i in range(len(ph) - 1):
        d = np.unpackbits((ph[i + 1:] ^ ph[i]).view(np.uint8)).reshape(-1, 64).sum(1)
        for off in np.nonzero(d <= max_dist)[0]:
            j = i + 1 + int(off)
            if same_content(ths[i], ths[j]):
                uf.union(i, j)
    return uf


def split_type(imgs: list, rng: random.Random, eval_frac=0.2, max_queries=20):
    """-> (train, gallery, queries) for one seen type."""
    imgs = sorted(imgs)
    rng.shuffle(imgs)
    n_eval = max(4, round(len(imgs) * eval_frac))
    ev, tr = imgs[:n_eval], imgs[n_eval:]
    nq = min(max_queries, len(ev) // 2)
    return tr, ev[nq:], ev[:nq]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--heldout", type=int, default=5, help="number of whole types kept out of training")
    ap.add_argument("--min-images", type=int, default=30, help="types with fewer clean images are dropped")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rows = []
    for src, path, col in SOURCES:
        df = pd.read_csv(path, dtype=str).fillna("")
        for r in df.to_dict("records"):  # not itertuples: "class" is a Python keyword
            lab = r[col]
            if src == "commons":
                if lab in COMMONS_DROP:
                    continue
                lab = COMMONS_ALIAS.get(lab, lab)
            rows.append({"src": src, "image": Path(r["clean_path"]).as_posix(), "label": lab, "ph": int(r["phash"], 16)})
    print(f"loaded {len(rows)} rows: {dict(Counter(r['src'] for r in rows))}", flush=True)
    ths = [thumb(Image.open(r["image"]).convert("RGB")) for r in rows]
    uf = near_groups(np.array([r["ph"] for r in rows], dtype=np.uint64), ths)
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[uf.find(i)].append(i)
    order = {"kaggle50": 0, "kaggle40": 1, "commons": 2}
    kept, conflicts, dups = [], 0, 0
    for members in groups.values():
        if len({rows[i]["label"] for i in members}) > 1:
            conflicts += len(members)
            continue
        best = min(members, key=lambda i: (order[rows[i]["src"]], rows[i]["image"]))
        kept.append(rows[best])
        dups += len(members) - 1
    by_type = defaultdict(list)
    for r in kept:
        by_type[r["label"]].append(r["image"])
    small = sorted(t for t, v in by_type.items() if len(v) < a.min_images)
    for t in small:
        del by_type[t]
    rng = random.Random(a.seed)
    types = sorted(by_type)
    heldout = sorted(rng.sample(types, a.heldout)) if a.heldout else []
    train, gallery, q_seen, q_held = [], [], [], []
    for t in types:
        if t in heldout:
            imgs = sorted(by_type[t])
            rng.shuffle(imgs)
            nq = min(20, len(imgs) // 2)
            q_held += [{"image": i, "part_id": t} for i in imgs[:nq]]
            gallery += [{"image": i, "part_id": t} for i in imgs[nq:]]
            continue
        tr, ga, qu = split_type(by_type[t], rng)
        train += [{"image": i, "part_id": t} for i in tr]
        gallery += [{"image": i, "part_id": t} for i in ga]
        q_seen += [{"image": i, "part_id": t} for i in qu]
    note = "type labels (not part numbers); Kaggle gpiosenka car parts 50+40 (Apache 2.0) and Wikimedia Commons (CC0/PD/CC BY)"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "splits_type_v1_train.json").write_text(json.dumps(
        {"provisional": True, "seed": a.seed, "note": note, "catalog": train, "heldout_part_ids": heldout}, indent=1), encoding="utf-8")
    (OUT / "splits_type_v1_eval.json").write_text(json.dumps(
        {"provisional": True, "seed": a.seed, "note": note, "catalog": gallery, "queries": {"valid": q_seen, "heldout": q_held}},
        indent=1), encoding="utf-8")
    tc = Counter(e["part_id"] for e in train)
    lines = ["# Type split v1\n", f"- rows loaded {len(rows)} {dict(Counter(r['src'] for r in rows))}",
             f"- near-duplicate copies removed {dups}; images dropped for conflicting labels {conflicts}",
             f"- types dropped (< {a.min_images} images): {small}",
             f"- types {len(types)} (seen {len(types) - len(heldout)}, held out {len(heldout)}: {heldout})",
             f"- train {len(train)} images; train per type min {min(tc.values())}, max {max(tc.values())}",
             f"- eval gallery {len(gallery)}; queries seen {len(q_seen)}, held-out {len(q_held)}",
             f"- kept by source {dict(Counter(r['src'] for r in kept))}"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

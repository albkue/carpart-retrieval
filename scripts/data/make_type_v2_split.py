#!/usr/bin/env python3
"""Merged type-level split v2: Kaggle 50/40 + Wikimedia Commons + HITL real crops + Objaverse renders, one dataset.

Same idea as make_type_split.py (v1), with the HITL crops and synthetic renders from final_dataset added.
- Labels: the v1 lower-case type names; HITL and synthetic names are mapped by HITL_MAP / SYNTH_MAP. HITL mirror
  becomes the held-out type "side mirror". Synthetic types without a real counterpart (turbocharger) and the
  held-out air filter are dropped.
- Held-out HITL crops (side mirror) are used only from photos on the eval side, so no photo straddles the split.
- Held-out types are fixed (not sampled) so v2 stays comparable with v1: air filter, camshaft, fuel injector,
  side mirror, transmission. They never reach training and synthetic images never reach evaluation.
- Near duplicates across all real sources are grouped (pHash <= 6 plus thumbnail check); one image per group stays.
- HITL crops are split by source photo (group_id), never by crop: a photo is wholly train or wholly eval.
  Kaggle and Commons images are split per type, 80/20, as in v1.
- Per type, a side is capped at --cap-train / --cap-eval: the HITL excess goes first, but HITL keeps a third of the cap
  when it has that many, so a type with many web photos still gets real HITL crops in its eval.
- Every row carries "src"; "real" sources are hitl and commons (the capture-condition test), kaggle is web photos.
Writes E:\\car_search_data\\eda\\type_v2\\{splits_type_v2_train.json, splits_type_v2_eval.json, report.md}.

Usage: python scripts/data/make_type_v2_split.py
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_ccby_split import thumb  # noqa: E402
from make_type_split import COMMONS_ALIAS, COMMONS_DROP, SOURCES, near_groups  # noqa: E402

E = Path(r"E:\car_search_data")
FD = E / "car_part_retrieval_data" / "final_dataset"
OUT = E / "eda/type_v2"
HELDOUT = ["air filter", "camshaft", "fuel injector", "side mirror", "transmission"]
HITL_MAP = {"windshield": "windshield", "bumper": "bumper", "door": "door", "fender": "fender", "grille": "grille",
            "hood": "hood", "trunk": "trunk", "wheel": "wheel", "headlight": "headlight", "tail_light": "taillight",
            "mirror": "side mirror"}
SYNTH_MAP = {**{k: v for k, v in HITL_MAP.items() if k != "mirror"}, "alternator": "alternator", "brake_disc": "brake disc",
             "muffler_exhaust": "exhaust muffler", "piston": "piston", "radiator": "radiator",
             "shock_absorber": "shock absorber", "spark_plug": "spark plug"}  # air_filter, turbocharger dropped
PRIORITY = {"kaggle50": 0, "kaggle40": 1, "commons": 2, "hitl": 3}
REAL = {"hitl", "commons"}


def cap_side(items: list, cap: int, rng: random.Random) -> list:
    """Fit a side to `cap`: HITL excess goes first, but HITL keeps at least a third of the cap when it has that many."""
    hitl = [x for x in items if x["src"] == "hitl"]
    rest = [x for x in items if x["src"] != "hitl"]
    rng.shuffle(hitl)
    rng.shuffle(rest)
    keep_rest = rest[:max(cap - min(len(hitl), cap // 3), cap - len(hitl))]
    return keep_rest + hitl[:max(0, cap - len(keep_rest))]


def split_photos(groups: list, rng: random.Random, eval_frac: float) -> set:
    """Photo ids that go to the eval side."""
    g = sorted(set(groups))
    rng.shuffle(g)
    return set(g[:round(len(g) * eval_frac)])


def check_leakage(train, gallery, queries, heldout):
    """-> list of (name, ok) leakage checks over the split rows."""
    paths = [{r["image"] for r in x} for x in (train, gallery, queries)]
    photo = lambda rows: {r["group"] for r in rows if r.get("group")}  # noqa: E731
    return [("no image in two of train/gallery/queries", not (paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2])),
            ("no HITL photo in train and eval", not (photo(train) & (photo(gallery) | photo(queries)))),
            ("held-out types absent from train", not ({r["part_id"] for r in train} & set(heldout))),
            ("no synthetic image in eval", not any(r["src"] == "synthetic" for r in gallery + queries))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-images", type=int, default=30)
    ap.add_argument("--max-queries", type=int, default=20)
    ap.add_argument("--cap-train", type=int, default=300)
    ap.add_argument("--cap-eval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows = []
    for src, path, col in SOURCES:
        for r in pd.read_csv(path, dtype=str).fillna("").to_dict("records"):
            lab = r[col]
            if src == "commons":
                if lab in COMMONS_DROP:
                    continue
                lab = COMMONS_ALIAS.get(lab, lab)
            rows.append({"src": src, "image": Path(r["clean_path"]).as_posix(), "part_id": lab, "group": "", "ph": int(r["phash"], 16)})
    meta = pd.read_csv(FD / "metadata/images.csv", dtype=str).fillna("")
    synth, dropped_labels = [], Counter()
    for r in meta.to_dict("records"):
        kind = r["source"]
        lab = (HITL_MAP if kind == "hitl" else SYNTH_MAP).get(r["part_type"])
        if lab is None:
            dropped_labels[f"{kind}:{r['part_type']}"] += 1
            continue
        row = {"src": "hitl" if kind == "hitl" else "synthetic", "image": (FD / r["cleaned_image_path"]).as_posix(),
               "part_id": lab, "group": r["group_id"] if kind == "hitl" else ""}
        if kind == "hitl":
            rows.append(row)
        else:
            synth.append(row)
    print(f"loaded {len(rows)} real rows {dict(Counter(r['src'] for r in rows))}, {len(synth)} synthetic; label-dropped {dict(dropped_labels)}", flush=True)

    ths = []
    for r in rows:
        rgb = Image.open(r["image"]).convert("RGB")
        if "ph" not in r:
            r["ph"] = int(str(imagehash.phash(rgb)), 16)
        ths.append(thumb(rgb))
    uf = near_groups(np.array([r["ph"] for r in rows], dtype=np.uint64), ths)
    clusters = defaultdict(list)
    for i in range(len(rows)):
        clusters[uf.find(i)].append(i)
    kept, conflicts, dups = [], 0, 0
    for members in clusters.values():
        if len({rows[i]["part_id"] for i in members}) > 1:
            conflicts += len(members)
            continue
        kept.append(rows[min(members, key=lambda i: (PRIORITY[rows[i]["src"]], rows[i]["image"]))])
        dups += len(members) - 1

    by_type = defaultdict(list)
    for r in kept:
        by_type[r["part_id"]].append(r)
    small = sorted(t for t, v in by_type.items() if len(v) < a.min_images)
    for t in small:
        del by_type[t]
    missing = [t for t in HELDOUT if t not in by_type]
    if missing:
        sys.exit(f"held-out types missing from the merged data: {missing}")
    eval_photos = split_photos([r["group"] for r in kept if r["group"]], rng, 0.2)

    train, gallery, q_seen, q_held = [], [], [], []
    for t in sorted(by_type):
        items = sorted(by_type[t], key=lambda r: r["image"])
        if t in HELDOUT:
            # a held-out HITL crop may only come from an eval-side photo, or its photo would straddle train and eval
            pool = cap_side([r for r in items if r["src"] != "hitl" or r["group"] in eval_photos], a.cap_eval, rng)
            rng.shuffle(pool)
            nq = min(a.max_queries, len(pool) // 2)
            q_held += pool[:nq]
            gallery += pool[nq:]
            continue
        tr, ev = [], []
        web = [r for r in items if r["src"] != "hitl"]
        rng.shuffle(web)
        n_ev = max(2, round(len(web) * 0.2)) if web else 0
        ev += web[:n_ev]
        tr += web[n_ev:]
        for r in items:
            if r["src"] == "hitl":
                (ev if r["group"] in eval_photos else tr).append(r)
        tr, ev = cap_side(tr, a.cap_train, rng), cap_side(ev, a.cap_eval, rng)
        rng.shuffle(ev)
        nq = min(a.max_queries, len(ev) // 2)
        q_seen += ev[:nq]
        gallery += ev[nq:]
        train += tr
    seen_types = {r["part_id"] for r in train}
    train += [r for r in synth if r["part_id"] in seen_types]

    checks = check_leakage(train, gallery, q_seen + q_held, HELDOUT)
    if not all(ok for _, ok in checks):
        sys.exit(f"leakage check failed: {checks}")

    def clean(rs):
        return [{"image": r["image"], "part_id": r["part_id"], "src": r["src"]} for r in rs]

    note = ("type labels (not part numbers); Kaggle gpiosenka 50+40 (Apache 2.0), Wikimedia Commons (CC0/PD/CC BY), "
            "HITL part crops (CC0), Objaverse renders (training only, CC BY)")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "splits_type_v2_train.json").write_text(json.dumps(
        {"provisional": True, "seed": a.seed, "note": note, "catalog": clean(train), "heldout_part_ids": HELDOUT}, indent=1), encoding="utf-8")
    (OUT / "splits_type_v2_eval.json").write_text(json.dumps(
        {"provisional": True, "seed": a.seed, "note": note, "catalog": clean(gallery),
         "queries": {"valid": clean(q_seen), "heldout": clean(q_held)}}, indent=1), encoding="utf-8")
    tc = Counter(r["part_id"] for r in train)
    lines = ["# Type split v2 (merged)\n",
             f"- real rows loaded {len(rows)} {dict(Counter(r['src'] for r in rows))}; synthetic {len(synth)}",
             f"- labels dropped {dict(dropped_labels)}",
             f"- near-duplicate copies removed {dups}; images dropped for conflicting labels {conflicts}",
             f"- types dropped (< {a.min_images} images): {small}",
             f"- types {len(by_type)} (seen {len(by_type) - len(HELDOUT)}, held out {len(HELDOUT)}: {HELDOUT})",
             f"- train {len(train)} images {dict(Counter(r['src'] for r in train))}; per type min {min(tc.values())}, max {max(tc.values())}",
             f"- eval gallery {len(gallery)} {dict(Counter(r['src'] for r in gallery))}",
             f"- queries seen {len(q_seen)} {dict(Counter(r['src'] for r in q_seen))}; held-out {len(q_held)} {dict(Counter(r['src'] for r in q_held))}",
             "", "## Leakage checks"] + [f"- {'PASS' if ok else 'FAIL'}: {n}" for n, ok in checks]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate the type split before training: paths exist and decode, labels are sane, nothing leaks.

Checks: every image exists and fully decodes (Pillow load, not just verify); labels are non-empty; held-out types
are absent from training; no image path is in both train and eval; every query type has gallery images;
counts match the expected numbers (if given); pHash near-duplicates across train and eval (from the EDA's
images.csv, confirmed there by a content check) are zero. Exits non-zero on any failure.

Usage: python scripts/data/validate_type_split.py [--expect 7228,1660,859,100]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

SPLIT = Path(r"E:\car_search_data\eda\type_v1")


def check_split(train: dict, ev: dict) -> list[str]:
    """Structural checks only (no file IO). Returns a list of problems; empty means valid."""
    problems = []
    tr, ga = train["catalog"], ev["catalog"]
    qs, qh = ev["queries"].get("valid", []), ev["queries"].get("heldout", [])
    held = set(train.get("heldout_part_ids", []))
    for name, rows in (("train", tr), ("gallery", ga), ("query_seen", qs), ("query_heldout", qh)):
        bad = [r for r in rows if not r.get("part_id") or not r.get("image")]
        if bad:
            problems.append(f"{name}: {len(bad)} rows without image or label")
    if held & {r["part_id"] for r in tr}:
        problems.append(f"held-out types in train: {sorted(held & {r['part_id'] for r in tr})}")
    if held != {r["part_id"] for r in qh}:
        problems.append("held-out query types differ from heldout_part_ids")
    tr_paths = {r["image"] for r in tr}
    ev_paths = [r["image"] for r in ga + qs + qh]
    if tr_paths & set(ev_paths):
        problems.append(f"{len(tr_paths & set(ev_paths))} images in both train and eval")
    dup = [p for p, n in Counter(ev_paths).items() if n > 1]
    if dup:
        problems.append(f"{len(dup)} eval images used twice")
    gallery_types = {r["part_id"] for r in ga}
    missing = {r["part_id"] for r in qs + qh} - gallery_types
    if missing:
        problems.append(f"query types without gallery images: {sorted(missing)}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--expect", default="7228,1660,859,100", help="train,gallery,query_seen,query_heldout")
    a = ap.parse_args()
    train = json.loads((SPLIT / "splits_type_v1_train.json").read_text(encoding="utf-8"))
    ev = json.loads((SPLIT / "splits_type_v1_eval.json").read_text(encoding="utf-8"))
    problems = check_split(train, ev)
    counts = [len(train["catalog"]), len(ev["catalog"]), len(ev["queries"]["valid"]), len(ev["queries"]["heldout"])]
    if a.expect and counts != [int(x) for x in a.expect.split(",")]:
        problems.append(f"counts {counts} != expected {a.expect}")
    unreadable = []
    rows = train["catalog"] + ev["catalog"] + ev["queries"]["valid"] + ev["queries"]["heldout"]
    for r in rows:
        try:
            with Image.open(r["image"]) as im:
                im.load()
        except Exception as e:  # missing file, truncated or corrupt
            unreadable.append((r["image"], type(e).__name__))
    if unreadable:
        problems.append(f"{len(unreadable)} unreadable images, e.g. {unreadable[:3]}")
    leak_csv = SPLIT / "eda" / "leakage_pairs.csv"
    if leak_csv.exists():
        n_leak = max(0, len(leak_csv.read_text(encoding="utf-8").strip().splitlines()) - 1)
        if n_leak:
            problems.append(f"{n_leak} confirmed train/eval near-duplicate pairs (eda/leakage_pairs.csv)")
    else:
        problems.append("eda/leakage_pairs.csv missing: run scripts/data/eda_type_split.py")
    types = {r["part_id"] for r in rows}
    print(f"images checked {len(rows)}; types {len(types)} (train {len({r['part_id'] for r in train['catalog']})}, "
          f"held out {len(train.get('heldout_part_ids', []))}); counts {counts}")
    for p in problems:
        print("PROBLEM:", p)
    print("VALID" if not problems else "INVALID")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()

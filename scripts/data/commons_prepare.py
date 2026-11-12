#!/usr/bin/env python3
"""Turn the filtered Commons images into distractors for the Sunday eval.

Reads relevance.csv (commons_filter.py), keeps score >= threshold, drops duplicate sha1, resizes to
--max-side, and writes (1) prepared_commons/images/<query>/N.jpg, (2) credits.csv with licence, creator and
source page for every kept image (CC BY needs credit in the thesis), (3) an eval manifest that is the CC BY
eval manifest plus the Commons images as extra gallery entries, each its own part_id. They have no part
numbers, so they can only be wrong answers (distractors), never queries or training parts.

Usage: python scripts/data/commons_prepare.py --threshold 0.85
"""
import argparse
import csv
import json
import os
from pathlib import Path

from PIL import Image

ROOT = Path(os.environ.get("CARPARTS_COMMONS_DIR", r"E:\car_search_data\raw\commons"))
EVAL = Path(r"E:\car_search_data\eda\ccby_v0\prepared\splits_ccby_v0_eval.json")


def select(rel_rows, meta_by_file, threshold):
    """Rows at or above threshold, one per sha1 (first, i.e. lowest-score-first order is irrelevant)."""
    seen, out = set(), []
    for r in rel_rows:
        m = meta_by_file.get(r["file"])
        if m is None or float(r["score"]) < threshold or m["sha1"] in seen:
            continue
        seen.add(m["sha1"])
        out.append((r, m))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--eval", type=Path, default=EVAL)
    a = ap.parse_args()
    import hashlib
    metas = [json.loads(l) for l in (ROOT / "metadata.jsonl").open(encoding="utf-8")]
    by_file = {}
    for m in metas:
        ext = ".png" if m["download_url"].lower().endswith(".png") else ".jpg"
        by_file[str(ROOT / "images" / (hashlib.sha1(m["page_url"].encode()).hexdigest()[:16] + ext))] = m
    rel = list(csv.DictReader((ROOT / "relevance.csv").open(encoding="utf-8")))
    kept = select(rel, by_file, a.threshold)
    out_dir = ROOT.parent.parent / "eda" / "commons_v0" / "images"
    gallery, counts = [], {}
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir.parent / "credits.csv").open("w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["file", "query", "score", "title", "creator", "licence", "licence_url", "source"])
        for r, m in kept:
            n = counts[m["query"]] = counts.get(m["query"], 0) + 1
            dst = out_dir / m["query"].replace(" ", "_") / f"{n}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            im = Image.open(r["file"]).convert("RGB")
            im.thumbnail((a.max_side, a.max_side))
            im.save(dst, quality=90)
            pid = f"commons-{m['query'].replace(' ', '_')}-{n}"
            gallery.append({"image": dst.as_posix(), "part_id": pid, "category_hint": m["query"]})
            w.writerow([dst.as_posix(), m["query"], r["score"], m["title"], m["creator"], m["licence"], m["licence_url"], m["page_url"]])
    man = json.loads(a.eval.read_text(encoding="utf-8"))
    man["catalog"] = man["catalog"] + gallery
    man["note"] += f"; plus {len(gallery)} Commons distractors (open licence, no part numbers, threshold {a.threshold})"
    dst = a.eval.with_name(a.eval.stem + "_plus_commons.json")
    dst.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    print(f"kept {len(gallery)} of {len(rel)}; per query {counts}\ncredits -> {out_dir.parent / 'credits.csv'}\nmanifest -> {dst}")


if __name__ == "__main__":
    main()

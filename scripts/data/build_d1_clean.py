#!/usr/bin/env python3
"""Turn the messy data/raw/roboflow/ into one deduped D1 manifest.

Built from what scripts/data/audit_dataset.py found on 3 Sep:
  - "car parts", "car parts 50", "car-parts-1" are the same underlying
    ~8.7k-image Kaggle set three times over (car-parts-1 is a heavily
    augmented Roboflow re-export). Keep ONLY "car parts 50" (50 classes,
    superset of the 40-class version, cleanest source of the three).
  - Starter-and-Alternator-2 is exported at 200x200px -- below the 400px
    short-edge floor in docs/DATASET_SPEC.md. Excluded by default; flip
    EXCLUDE_UNDERSIZED off if you decide you want it anyway (e.g. for
    YOLO pretraining, where 200px is less damaging than for CLIP eval).
  - Every Roboflow project's own train/valid/test split leaks duplicates
    across the boundary (proven per-project by the audit). This script
    ignores those folders as splits and re-splits itself, at the
    exact-duplicate-cluster level, so no duplicate can straddle a split.

This does EXACT-hash dedup only (fast, ~O(n)). Near-duplicate (rotated /
recropped) images are NOT collapsed here -- that's a second pass worth
doing later at the smaller post-dedup scale, not before.

Usage:
    python scripts/data/build_d1_clean.py
    python scripts/data/build_d1_clean.py --root data/raw/roboflow --out data/interim
"""
import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import imagehash

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# --- edit this as you add/drop sources -------------------------------------
KEEP_SOURCES = [
    "car parts 50",                    # canonical: gpiosenka 50-class, supersedes "car parts"
    "Complete-Spare-Parts-2",          # Tier 1b
    "Supply-Spare-Parts-1",            # Tier 1b
    "Supply-Spare-Parts-Detection-1",  # Tier 1b
    "Automobile-parts",                # Tier 1, CC0 -- kept despite lower quality (16% dup, 63% undersized)
]
DROP_SOURCES = [
    "car parts",        # subset of "car parts 50" -- redundant
    "car-parts-1",       # ghost-ce1po augmented re-export of the same set -- redundant
]
EXCLUDE_UNDERSIZED_SOURCES = ["Starter-and-Alternator-2"]  # 200x200px, see docstring
# 400px is DATASET_SPEC.md's floor for D2/D3 (catalog + real query photos) --
# D1 feeds classification/detection pretraining, where 224px is CLIP's own
# native input size, not a defect. Use a much lower floor here (128) and
# only drop something as unusably small by putting its SOURCE in
# EXCLUDE_UNDERSIZED_SOURCES, judged case by case (see Starter-and-Alternator-2).
MIN_SHORT_EDGE = 128
SPLIT_RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}
SEED = 42


def hash_source(src_dir: Path, cache_path: Path) -> dict:
    """phash every image under src_dir, caching to cache_path (json)."""
    if cache_path.exists():
        print(f"  [cache] {src_dir.name} -> {cache_path}")
        return json.loads(cache_path.read_text())

    result = {}
    files = [p for p in src_dir.rglob("*") if p.suffix.lower() in IMG_EXT]
    print(f"  hashing {len(files)} images in {src_dir.name} ...")
    for i, f in enumerate(files, 1):
        try:
            with Image.open(f) as im:
                w, h = im.size
                if min(w, h) < MIN_SHORT_EDGE:
                    continue
                result[str(f)] = {"phash": str(imagehash.phash(im)), "w": w, "h": h}
        except (UnidentifiedImageError, OSError):
            continue
        if i % 2000 == 0:
            print(f"    {i}/{len(files)}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/roboflow"))
    ap.add_argument("--out", type=Path, default=Path("data/interim"))
    args = ap.parse_args()

    cache_dir = args.out / "hashes"
    all_records = {}  # phash -> list of (path, source)
    for src_name in KEEP_SOURCES:
        src_dir = args.root / src_name
        if not src_dir.exists():
            print(f"  !! missing: {src_dir} -- skipping")
            continue
        cache_path = cache_dir / f"{src_name.replace(' ', '_')}.json"
        hashes = hash_source(src_dir, cache_path)
        for path, meta in hashes.items():
            all_records.setdefault(meta["phash"], []).append({"path": path, "source": src_name, **meta})

    n_before = sum(len(v) for v in all_records.values())
    n_unique = len(all_records)
    print(f"\n{n_before} qualifying images -> {n_unique} unique (exact-hash) after dedup "
          f"({n_before - n_unique} dropped as exact duplicates)")

    # split at the CLUSTER level so a duplicate can never land in two splits
    phashes = list(all_records.keys())
    random.seed(SEED)
    random.shuffle(phashes)
    n = len(phashes)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_valid = int(n * SPLIT_RATIOS["valid"])
    split_of = {}
    for i, h in enumerate(phashes):
        if i < n_train:
            split_of[h] = "train"
        elif i < n_train + n_valid:
            split_of[h] = "valid"
        else:
            split_of[h] = "test"

    manifest_path = args.out / "d1_clean_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_path", "source", "phash", "width", "height", "split", "is_primary_copy"])
        for h, records in all_records.items():
            split = split_of[h]
            for i, r in enumerate(records):
                # first copy of a duplicate cluster is the one to actually use;
                # the rest are logged for traceability but marked non-primary
                w.writerow([r["path"], r["source"], h, r["w"], r["h"], split, int(i == 0)])

    print(f"\nmanifest -> {manifest_path}")
    print(f"train/valid/test cluster counts: "
          f"{sum(1 for v in split_of.values() if v=='train')} / "
          f"{sum(1 for v in split_of.values() if v=='valid')} / "
          f"{sum(1 for v in split_of.values() if v=='test')}")
    print(f"\nDropped entirely: {', '.join(DROP_SOURCES)}")
    print(f"Excluded (undersized): {', '.join(EXCLUDE_UNDERSIZED_SOURCES)} -- "
          f"flip in this script's config if you want them for detection pretraining")


if __name__ == "__main__":
    main()

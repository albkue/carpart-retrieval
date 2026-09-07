#!/usr/bin/env python3
"""EDA / audit pass over data/raw before anything gets trained on.

Answers three questions per source folder:
  1. How many images, what sizes, any corrupt files?
  2. How much of this is exact/near-duplicate of something else (phash)?
  3. What's the class balance (if the folder is class-per-subdir or has a CSV)?

Cross-source duplication matters more than in-source: data-sources.md already
flagged that gpiosenka/car-parts-40-classes, ghost-ce1po/car-parts-kdymg, and
Roboflow's augmented re-export of the same set are likely the same ~8,700
base images three times over. This script is what proves (or disproves) that
with numbers instead of a guess.

Usage:
    python scripts/audit_dataset.py data/raw/roboflow --out data/interim/audit
    python scripts/audit_dataset.py data/raw/roboflow --sample 800   # quick pass
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import imagehash

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
HAMMING_DUP_THRESHOLD = 6  # <=6 bits different on a 64-bit phash ~= near-duplicate


def iter_images(root: Path, sample: int | None):
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT]
    if sample and len(files) > sample:
        import random
        random.seed(42)
        files = random.sample(files, sample)
    return files


def top_level_source(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if rel.parts else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="folder to audit (scanned recursively)")
    ap.add_argument("--out", type=Path, default=Path("data/interim/audit"))
    ap.add_argument("--sample", type=int, default=None,
                     help="cap total images scanned (random sample) for a quick pass")
    ap.add_argument("--min-short-edge", type=int, default=400,
                     help="flag images below this on the short edge (per DATASET_SPEC.md)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    files = iter_images(args.root, args.sample)
    print(f"scanning {len(files)} image files under {args.root} ...")

    per_source = defaultdict(lambda: {"n": 0, "corrupt": 0, "undersized": 0})
    hashes: dict[str, list[Path]] = defaultdict(list)  # exact phash -> files
    dims = []
    corrupt_files = []
    undersized_files = []

    for i, f in enumerate(files, 1):
        src = top_level_source(f, args.root)
        per_source[src]["n"] += 1
        try:
            with Image.open(f) as im:
                im.verify()
            with Image.open(f) as im:  # re-open, verify() invalidates the handle
                w, h = im.size
                phash = str(imagehash.phash(im))
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            per_source[src]["corrupt"] += 1
            corrupt_files.append(str(f))
            continue

        dims.append((w, h))
        if min(w, h) < args.min_short_edge:
            per_source[src]["undersized"] += 1
            undersized_files.append(str(f))
        hashes[phash].append(f)

        if i % 2000 == 0:
            print(f"  {i}/{len(files)}")

    # --- exact-hash duplicate groups (identical phash) ----------------------
    exact_dup_groups = {h: fs for h, fs in hashes.items() if len(fs) > 1}
    n_exact_dupes = sum(len(fs) - 1 for fs in exact_dup_groups.values())

    # --- cross-source overlap: which sources share exact-hash images --------
    cross_source_overlap = defaultdict(int)
    for fs in exact_dup_groups.values():
        sources = {top_level_source(f, args.root) for f in fs}
        if len(sources) > 1:
            key = " + ".join(sorted(sources))
            cross_source_overlap[key] += 1

    # --- near-duplicate pass (hamming distance) — only if corpus is small ---
    # O(n^2) hamming compare is fine up to a few thousand hashes; skip above that.
    near_dup_pairs = 0
    uniq_hashes = list(hashes.keys())
    if len(uniq_hashes) <= 6000:
        hobjs = [imagehash.hex_to_hash(h) for h in uniq_hashes]
        for i in range(len(hobjs)):
            for j in range(i + 1, len(hobjs)):
                if hobjs[i] - hobjs[j] <= HAMMING_DUP_THRESHOLD:
                    near_dup_pairs += 1
    else:
        near_dup_pairs = None  # too large for O(n^2) here; use exact-hash signal only

    # --- write report ---------------------------------------------------------
    report = {
        "root": str(args.root),
        "n_scanned": len(files),
        "n_corrupt": len(corrupt_files),
        "n_undersized": len(undersized_files),
        "per_source": dict(per_source),
        "n_unique_phash": len(hashes),
        "n_exact_duplicate_images": n_exact_dupes,
        "cross_source_exact_overlap": dict(cross_source_overlap),
        "near_duplicate_pairs_within_hamming_6": near_dup_pairs,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))

    with (args.out / "corrupt_files.txt").open("w") as f:
        f.write("\n".join(corrupt_files))
    with (args.out / "undersized_files.txt").open("w") as f:
        f.write("\n".join(undersized_files))
    with (args.out / "exact_duplicate_groups.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phash", "n_copies", "files"])
        for h, fs in exact_dup_groups.items():
            w.writerow([h, len(fs), " | ".join(str(x) for x in fs)])

    print("\n=== SUMMARY ===")
    print(f"scanned:            {report['n_scanned']}")
    print(f"corrupt:             {report['n_corrupt']}")
    print(f"undersized (<{args.min_short_edge}px): {report['n_undersized']}")
    print(f"unique phash:        {report['n_unique_phash']}")
    print(f"exact-dup images:    {report['n_exact_duplicate_images']}  "
          f"({report['n_exact_duplicate_images']/max(1,report['n_scanned']):.1%} of scanned)")
    if cross_source_overlap:
        print("cross-source overlap (same image, different folders):")
        for k, v in sorted(cross_source_overlap.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
    if near_dup_pairs is not None:
        print(f"near-dup pairs (hamming<={HAMMING_DUP_THRESHOLD}): {near_dup_pairs}")
    print(f"\nfull report -> {args.out}/report.json")


if __name__ == "__main__":
    main()

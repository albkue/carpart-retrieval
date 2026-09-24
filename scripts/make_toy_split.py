#!/usr/bin/env python3
"""Build a toy split manifest to prove the experiment loop end to end.

Picks N parts with at least two images from data/raw/catalog_images/<part>/,
holds one image per part out as the query, and indexes the rest. The queries
are catalogue shots, not shop photographs, so the numbers this produces say the
plumbing works and nothing about the embedding — the manifest is marked
``toy: true`` and so is every metrics.json made from it.

Usage:
    python scripts/make_toy_split.py
    python scripts/make_toy_split.py --parts 50 --seed 42 --out data/processed/splits_toy.json
"""
import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def make_toy_split(root: Path, parts: int, seed: int) -> dict:
    folders = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        if len(images) >= 2:
            folders[d.name] = images
    if len(folders) < parts:
        raise SystemExit(f"Only {len(folders)} parts with 2+ images under {root}")

    rng = random.Random(seed)
    catalog, queries = [], []
    for part in sorted(rng.sample(sorted(folders), parts)):
        images = folders[part]
        query = rng.choice(images)
        queries.append({"image": query.relative_to(REPO).as_posix(), "part_id": part})
        catalog += [{"image": img.relative_to(REPO).as_posix(), "part_id": part} for img in images if img != query]
    return {"toy": True, "seed": seed, "catalog": catalog, "queries": {"valid": queries}}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO / "data/raw/catalog_images")
    parser.add_argument("--parts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=REPO / "data/processed/splits_toy.json")
    args = parser.parse_args()
    split = make_toy_split(args.root.resolve(), args.parts, args.seed)
    args.out.write_text(json.dumps(split, indent=1) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(split['catalog'])} catalogue images, {len(split['queries']['valid'])} queries")


if __name__ == "__main__":
    main()

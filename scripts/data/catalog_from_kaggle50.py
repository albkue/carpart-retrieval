#!/usr/bin/env python3
"""Turn the Kaggle gpiosenka/car-parts-40-classes export into D2 distractor rows.

This set labels part TYPE (e.g. "IGNITION COIL"), never a part number, so
unlike catalog_from_tier1b.py each image becomes its own synthetic,
unverified part entry -- there is no folder-name part number to group on.

Category comes from data/processed/taxonomy.json's yolo_classes lists (the
Kaggle labels line up with those 1:1), so the mapping stays in one place
instead of being retyped here.

Usage:
    python scripts/data/catalog_from_kaggle50.py "data/raw/roboflow/car parts 50" \
        --out data/raw/catalog_images/batch_kaggle50_distractors.csv \
        --collected-by <your-name>
"""
import argparse
import csv
import json
import re
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def slugify(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", label.strip()).strip("-").upper()


def load_label_to_category(taxonomy_path: Path) -> dict:
    entries = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    mapping = {}
    for entry in entries:
        for yolo_class in entry["yolo_classes"]:
            mapping[yolo_class.replace("_", " ").upper()] = entry["label"]
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_root", type=Path, help="'car parts 50' folder (contains train/test/valid)")
    ap.add_argument("--out", type=Path, default=Path("data/raw/catalog_images/batch_kaggle50_distractors.csv"))
    ap.add_argument("--images-out", type=Path, default=Path("data/raw/catalog_images"))
    ap.add_argument("--source-site", default="kaggle_gpiosenka_car-parts-40-classes")
    ap.add_argument("--taxonomy", type=Path, default=Path("data/processed/taxonomy.json"))
    ap.add_argument("--collected-by", required=True)
    args = ap.parse_args()

    label_to_category = load_label_to_category(args.taxonomy)

    images_by_label: dict[str, list[Path]] = {}
    for class_dir in args.src_root.rglob("*"):
        if not class_dir.is_dir():
            continue
        images = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in IMG_EXT)
        if images:
            images_by_label.setdefault(class_dir.name.strip().upper(), []).extend(images)

    rows = []
    n_images = 0
    unmapped = set()
    for label, images in sorted(images_by_label.items()):
        category = label_to_category.get(label)
        if category is None:
            unmapped.add(label)
        slug = slugify(label)
        for i, img in enumerate(images, start=1):
            part_number_raw = f"KAGGLE50-{slug}-{i:04d}"
            dest_dir = args.images_out / part_number_raw
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest_dir / f"1{img.suffix.lower()}")
            n_images += 1
            rows.append({
                "part_number_raw": part_number_raw,
                "brand": "",
                "category": category or "",
                "name": f"{label.title()} (Kaggle 50-type car parts, unverified)",
                "oem_number": "",
                "source_url": "https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes",
                "source_site": args.source_site,
                "collected_by": args.collected_by,
                "is_distractor": "true",
                "verified": "false",
                "notes": "auto-imported, type label only (no part number) -- never promote to ground truth"
                         + ("" if category else " | no taxonomy category match, fill by hand"),
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "part_number_raw", "brand", "category", "name", "oem_number",
            "source_url", "source_site", "collected_by", "is_distractor",
            "verified", "notes",
        ])
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} rows (one per image), {n_images} images -> {args.out}")
    if unmapped:
        print(f"no taxonomy match for: {', '.join(sorted(unmapped))}")
    print("Next: python scripts/validate_catalog.py")


if __name__ == "__main__":
    main()

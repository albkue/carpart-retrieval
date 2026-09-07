#!/usr/bin/env python3
"""Turn a Roboflow 'folder' export (class name = part number) into D2 rows.

Tier 1b projects (Supply Spare Parts family) export one folder per class,
where the class name IS the part number, e.g.:
    tier1b/supply-spare-parts/14670-KWB-600/img1.jpg
    tier1b/supply-spare-parts/14670-KWB-600/img2.jpg

This script:
  1. Copies each image to data/raw/catalog_images/<part_slug>/<n>.jpg
  2. Appends one row per PART (not per image) to a catalog CSV in the same
     9-column shape as catalog_TEMPLATE.csv, so it merges with junior batches.

It does NOT set `verified`. Spot-check a sample of part numbers against the
source listing before you set verified=true in the DB -- that's still a
you-task per DATASET_SPEC.md.

Usage:
    python scripts/catalog_from_tier1b.py data/raw/catalog_images/tier1b \
        --out data/raw/catalog_images/catalog_tier1b.csv \
        --source-site roboflow_tier1b --collected-by <your-name>
"""
import argparse
import csv
import re
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def slugify(part_number: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", part_number.strip()).strip("-").upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_root", type=Path, help="folder containing one subdir per Roboflow project")
    ap.add_argument("--out", type=Path, default=Path("data/raw/catalog_images/catalog_tier1b.csv"))
    ap.add_argument("--images-out", type=Path, default=Path("data/raw/catalog_images"))
    ap.add_argument("--source-site", default="roboflow_tier1b")
    ap.add_argument("--collected-by", required=True)
    args = ap.parse_args()

    rows = []
    n_images = 0
    for project_dir in sorted(p for p in args.src_root.iterdir() if p.is_dir()):
        # Roboflow "folder" exports put class dirs straight under the project;
        # a plain train/valid/test export puts a split dir in between. Find
        # class dirs at either depth by looking for the leaves that hold images,
        # then merge same-named class dirs across splits so a part isn't split
        # across two catalog rows or two images clobbering the same dest slot.
        images_by_part: dict[str, list[Path]] = {}
        for class_dir in project_dir.rglob("*"):
            if not class_dir.is_dir():
                continue
            images = [f for f in class_dir.iterdir() if f.suffix.lower() in IMG_EXT]
            if images:
                images_by_part.setdefault(class_dir.name, []).extend(images)

        for part_number_raw, images in sorted(images_by_part.items()):
            slug = slugify(part_number_raw)
            dest_dir = args.images_out / slug
            dest_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(images, start=1):
                shutil.copy2(img, dest_dir / f"{i}{img.suffix.lower()}")
                n_images += 1
            rows.append({
                "part_number_raw": part_number_raw,
                "brand": "",  # unknown from filename alone -- leave blank, don't guess
                "category": "",  # same -- fill by hand or in a review pass
                "name": f"{project_dir.name} part {part_number_raw}",
                "oem_number": "",
                "source_url": f"https://universe.roboflow.com/supply-spare-parts/{project_dir.name}",
                "source_site": args.source_site,
                "collected_by": args.collected_by,
                "notes": f"auto-imported from {project_dir.name}, {len(images)} image(s), "
                         f"is_distractor candidate -- verify part number before indexing",
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "part_number_raw", "brand", "category", "name", "oem_number",
            "source_url", "source_site", "collected_by", "notes",
        ])
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} parts, {n_images} images -> {args.out}")
    print("Next: run `make validate` on this file, then hand-fill brand/category "
          "for anything you plan to actually verify.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Consolidate and validate catalogue labelling batches.

Reads every data/raw/catalog_images/batch_*.csv, derives part_number via
the single normalisation function (pipeline/part_number.py), checks each
row against docs/DATASET_SPEC.md §3, and writes the validated rows to
data/processed/catalog_master.csv. Invalid rows are written to
data/processed/catalog_validation_errors.csv instead, with the reason.

A field that is inconsistent in the CSV is a filter that silently returns
nothing once it becomes Qdrant payload -- category and brand are checked
against fixed lists rather than accepted as free text.

Usage:
    python scripts/validate_catalog.py
    python scripts/validate_catalog.py --root data/raw/catalog_images --out data/processed/catalog_master.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.part_number import normalize_part_number
from pipeline.brand_matcher import BrandMatcher

FIELDS = [
    "part_number_raw", "part_number", "brand", "category", "name",
    "oem_number", "source_url", "source_site", "collected_by",
    "is_distractor", "verified", "notes",
]
REQUIRED = [
    "part_number_raw", "category", "name", "source_url", "source_site",
    "collected_by", "is_distractor", "verified",
]
_TRUE = {"true", "1", "yes"}
_FALSE = {"false", "0", "no", ""}


def _to_bool(value: str, field: str, errors: list) -> bool:
    v = (value or "").strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    errors.append(f"{field}='{value}' is not a valid boolean")
    return False


def load_taxonomy(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    return {e["label"] for e in entries}


def validate_row(row: dict, categories: set, brands: BrandMatcher) -> tuple:
    """Returns (clean_row_or_None, list_of_errors)."""
    errors = []
    for field in REQUIRED:
        if not (row.get(field) or "").strip() and field not in ("is_distractor", "verified"):
            errors.append(f"missing required field '{field}'")

    part_number_raw = (row.get("part_number_raw") or "").strip()
    part_number = normalize_part_number(part_number_raw)
    if part_number_raw and not part_number:
        errors.append("part_number_raw normalises to empty string")

    category = (row.get("category") or "").strip()
    if categories and category and category not in categories:
        errors.append(f"category '{category}' not in taxonomy")

    is_distractor = _to_bool(row.get("is_distractor", ""), "is_distractor", errors)
    verified = _to_bool(row.get("verified", ""), "verified", errors)

    brand_raw = (row.get("brand") or "").strip()
    brand = ""
    if brand_raw:
        matched = brands.match(brand_raw)
        if matched is None:
            errors.append(f"brand '{brand_raw}' not in known brand list")
        else:
            brand = matched
    elif not is_distractor:
        errors.append("brand is required for non-distractor (D3) rows")

    if errors:
        return None, errors

    clean = {
        "part_number_raw": part_number_raw,
        "part_number": part_number,
        "brand": brand,
        "category": category,
        "name": (row.get("name") or "").strip(),
        "oem_number": (row.get("oem_number") or "").strip(),
        "source_url": (row.get("source_url") or "").strip(),
        "source_site": (row.get("source_site") or "").strip(),
        "collected_by": (row.get("collected_by") or "").strip(),
        "is_distractor": str(is_distractor).lower(),
        "verified": str(verified).lower(),
        "notes": (row.get("notes") or "").strip(),
    }
    return clean, []


def validate_catalog(root: Path, taxonomy_path: Path, brands_file: str = None):
    categories = load_taxonomy(taxonomy_path)
    brands = BrandMatcher(brands_file)

    batch_files = sorted(root.glob("batch_*.csv"))
    clean_rows, error_rows = [], []

    for batch_file in batch_files:
        with open(batch_file, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f), start=2):
                clean, errors = validate_row(row, categories, brands)
                if clean is not None:
                    image_dir = root / clean["part_number_raw"]
                    if not image_dir.is_dir():
                        clean["notes"] = (clean["notes"] + f" | image dir '{clean['part_number_raw']}' not found").strip(" |")
                    clean_rows.append(clean)
                else:
                    error_rows.append({
                        "batch_file": batch_file.name,
                        "row": i,
                        "part_number_raw": row.get("part_number_raw", ""),
                        "errors": "; ".join(errors),
                    })

    return clean_rows, error_rows


def write_csv(path: Path, fieldnames: list, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/catalog_images")
    parser.add_argument("--out", default="data/processed/catalog_master.csv")
    parser.add_argument("--errors-out", default="data/processed/catalog_validation_errors.csv")
    parser.add_argument("--taxonomy", default="data/processed/taxonomy.json")
    parser.add_argument("--brands-file", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    clean_rows, error_rows = validate_catalog(root, Path(args.taxonomy), args.brands_file)

    write_csv(Path(args.out), FIELDS, sorted(clean_rows, key=lambda r: r["part_number"]))
    write_csv(Path(args.errors_out), ["batch_file", "row", "part_number_raw", "errors"], error_rows)

    print(f"{len(clean_rows)} valid rows -> {args.out}")
    print(f"{len(error_rows)} invalid rows -> {args.errors_out}")
    if error_rows:
        sys.exit(1)


if __name__ == "__main__":
    main()

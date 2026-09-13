#!/usr/bin/env python3
"""Pull a sample of Amazon Reviews 2023 'Automotive' listings into D2 shape.

This is Tier 4 in DATA_SOURCES.md: 2.0M automotive items, each with an
asin/parent_asin identity, brand, a `details` dict that sometimes carries a
Manufacturer Part Number, and hi_res/large/thumb image URLs. It is the only
source that gives you a realistic *catalog to retrieve against* at scale --
use it for distractors and for stress-testing retrieval at catalog sizes
your own capture will never reach, NOT as ground truth for RQ2 (you cannot
verify these part numbers by hand at scale).

CHECK THE LICENCE on the HF dataset card before you cite this
(https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) -- it is
unstated on the project site.

Usage:
    pip install datasets requests --break-system-packages
    python scripts/data/build_catalog_from_amazon.py \
        --n 2000 --out data/raw/catalog_images/catalog_amazon_sample.csv \
        --images-out data/raw/catalog_images/amazon --download-images
"""
import argparse
import csv
import re
from pathlib import Path

MPN_KEYS = ["Manufacturer Part Number", "manufacturer_part_number", "mpn", "MPN", "Part Number"]


def find_mpn(details: dict):
    for k in MPN_KEYS:
        v = details.get(k)
        if v:
            return str(v).strip()
    return ""


def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s.strip()).strip("-").upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="items to sample")
    ap.add_argument("--out", type=Path, default=Path("data/raw/catalog_images/catalog_amazon_sample.csv"))
    ap.add_argument("--images-out", type=Path, default=Path("data/raw/catalog_images/amazon"))
    ap.add_argument("--download-images", action="store_true")
    ap.add_argument("--mpn-only", action="store_true",
                     help="keep only listings with a manufacturer part number (fewer, higher value)")
    ap.add_argument("--collected-by", default="amazon_reviews_2023")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Automotive",
        split="full", streaming=True, trust_remote_code=True,
    )

    rows = []
    n_seen = 0
    for item in ds:
        n_seen += 1
        details = item.get("details") or {}
        mpn = find_mpn(details) if isinstance(details, dict) else ""
        if args.mpn_only and not mpn:
            continue

        images = item.get("images") or []
        img_url = None
        for im in images:
            img_url = im.get("hi_res") or im.get("large") or im.get("thumb")
            if img_url:
                break
        if not img_url:
            continue

        asin = item.get("parent_asin") or item.get("asin") or ""
        title = (item.get("title") or "").strip()
        brand = (item.get("store") or "").strip()
        part_number_raw = mpn or asin  # ASIN is a catalog identity, not a real part
        # is_distractor / verified are DB-level flags -- set later, don't guess here
        rows.append({
            "part_number_raw": part_number_raw,
            "brand": brand,
            "category": "",
            "name": title[:120],
            "oem_number": mpn if mpn and mpn != part_number_raw else "",
            "source_url": f"https://www.amazon.com/dp/{asin}",
            "source_site": "amazon_reviews_2023",
            "collected_by": args.collected_by,
            "notes": ("has_real_mpn" if mpn else "asin_only_no_verified_part_number")
                     + "; is_distractor candidate, not for RQ2 ground truth",
            "_image_url": img_url,
            "_slug": slugify(part_number_raw or asin),
        })
        if len(rows) >= args.n:
            break
        if n_seen % 5000 == 0:
            print(f"scanned {n_seen}, kept {len(rows)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        fieldnames = ["part_number_raw", "brand", "category", "name", "oem_number",
                      "source_url", "source_site", "collected_by", "notes"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"{len(rows)} rows -> {args.out}")

    if args.download_images:
        import requests
        args.images_out.mkdir(parents=True, exist_ok=True)
        ok, fail = 0, 0
        for r in rows:
            dest = args.images_out / r["_slug"]
            dest.mkdir(parents=True, exist_ok=True)
            try:
                resp = requests.get(r["_image_url"], timeout=15)
                resp.raise_for_status()
                (dest / "1.jpg").write_bytes(resp.content)
                ok += 1
            except Exception:
                fail += 1
        print(f"images: {ok} downloaded, {fail} failed (dead links expected -- this is normal)")


if __name__ == "__main__":
    main()

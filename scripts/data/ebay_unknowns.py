#!/usr/bin/env python3
"""Pick near-unknown parts from the eBay items that never made the catalogue.

A near unknown is a real part with a known brand and part number that is NOT in the
catalogue, from a category the system knows (RESEARCH_PROTOCOL 3.3: braking, filters,
engine and powertrain; never the held-out lighting / suspension and steering).
The A6 abstention test needs these as "should answer: not in catalogue" queries.

A part is excluded if its number, or any long number in its title, also appears in a
catalogue part's number or titles: that would be the same product under another label.

Reads data/raw/ebay/raw/items.jsonl and the cleaned catalogue; copies up to
--max-images photos per part (one listing only) to data/interim/ebay_unknowns/.

Usage:
    python scripts/data/ebay_unknowns.py
"""
import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from ebay_clean import LEAF_TO_LABEL, norm, safe

NEAR_LABELS = {"braking", "filters", "engine and powertrain"}
JUNK_MPN = {"", "na", "none", "doesnotapply", "notapplicable", "unknown", "unbranded", "seedescription", "0"}
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{5,}[A-Za-z0-9]")


def tokens(text: str) -> set[str]:
    return {norm(t) for t in TOKEN.findall(text) if re.search(r"\d", t) and len(norm(t)) >= 7}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, default=Path("data/raw/ebay/raw/items.jsonl"))
    ap.add_argument("--images", type=Path, default=Path("data/raw/ebay/images"))
    ap.add_argument("--catalog", type=Path, default=Path("data/interim/ebay_clean_v2/catalog_clean.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/interim/ebay_unknowns"))
    ap.add_argument("--per-category", type=int, default=20)
    ap.add_argument("--max-images", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    with a.catalog.open(encoding="utf-8", newline="") as f:
        cat = list(csv.DictReader(f))
    cat_mpns = {norm(c["mpn"]) for c in cat}
    cat_tokens: set[str] = set()
    for c in cat:
        cat_tokens |= tokens(c["title"]) | {norm(c["mpn"])}

    parts: dict[str, list[dict]] = defaultdict(list)
    for line in a.items.open(encoding="utf-8"):
        it = json.loads(line)
        mpn = norm(it.get("mpn") or "")
        if mpn in JUNK_MPN or len(mpn) < 5 or (it.get("brand") or "unknown") == "unknown":
            continue
        label = LEAF_TO_LABEL.get((it.get("categoryPath") or "").split("|")[-1])
        if label not in NEAR_LABELS:
            continue
        it["_label"], it["_mpn"] = label, mpn
        parts[mpn].append(it)

    skipped = Counter()
    candidates: dict[str, list[dict]] = defaultdict(list)
    for mpn, items in parts.items():
        if mpn in cat_mpns:
            skipped["in_catalogue"] += 1
            continue
        if (tokens(" ".join(i["title"] for i in items)) | {mpn}) & cat_tokens:
            skipped["same_product_as_catalogue"] += 1
            continue
        # one listing with usable images, preferring the one with most photos on disk
        best, best_imgs = None, []
        for it in items:
            imgs = sorted(a.images.glob(f"{it['itemId'].replace('|', '_')}_*.jpg"))[: a.max_images]
            if len(imgs) > len(best_imgs):
                best, best_imgs = it, imgs
        if not best_imgs:
            skipped["no_images"] += 1
            continue
        candidates[best["_label"]].append({"it": best, "imgs": best_imgs, "n_listings": len(items)})

    rng = random.Random(a.seed)
    rows = []
    for label in sorted(NEAR_LABELS):
        pool = candidates[label]
        rng.shuffle(pool)
        for c in pool[: a.per_category]:
            it = c["it"]
            part_id = f"{it['brand']}::{it['_mpn']}"
            for img in c["imgs"]:
                dst = a.out / "near" / label.replace(" ", "_") / safe(part_id) / img.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, dst)
                rows.append({
                    "part_id": part_id, "brand": it["brand"], "mpn": it["_mpn"], "title": it["title"],
                    "category": label, "image": dst.as_posix(), "item_id": it["itemId"],
                    "item_url": it.get("itemWebUrl", "").split("?")[0], "kind": "near_unknown",
                })

    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "unknowns.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    per_cat = Counter()
    for r in {r["part_id"]: r for r in rows}.values():
        per_cat[r["category"]] += 1
    report = {
        "candidates_per_category": {k: len(v) for k, v in candidates.items()},
        "selected_parts_per_category": dict(per_cat),
        "selected_parts": sum(per_cat.values()),
        "images": len(rows),
        "skipped": dict(skipped),
    }
    (a.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

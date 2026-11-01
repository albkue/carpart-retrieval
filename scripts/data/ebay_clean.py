#!/usr/bin/env python3
"""Clean the eBay marketplace set and lay its images out by part.

Reads data/raw/ebay/{catalog,gallery,queries}.csv (made by ebay_collect.py
`build`) and writes a cleaned copy to data/interim/ebay_clean/. Nothing under
data/raw is changed; images are copied, not moved.

Rules (each dropped row is listed in dropped.csv with its reason):
  * brand "unknown": the part is grouped by number alone, the least trustworthy case
  * a query whose title looks like a multi-item listing (front+rear, kit, lot, bulk)
    and does not contain the part number: its photos may show another number
  * a part left with no gallery image or no query image after the above
  * a part whose number is junk text ("DENSO", "UPDATE...") or has no digit
  * a query, or the gallery listing, whose title lacks the part number but names the
    number of ANOTHER part in this set (found by the 2026-10-21 hand check:
    "Bosch 3320" whose photo and title were Bosch 72142)
  * a part the 2026-10-21 hand check marked wrong, bad-number or unsure (n, m, ?) in
    label_check_sample.csv, matched by part number
Short part numbers (4 characters or fewer) are kept but flagged `short_mpn`.
Products listed under different numbers are NOT merged here (DATASET_SPEC 4.3: group
only on direct evidence); see merge_candidates.csv for the split constraint.

Categories are mapped to data/processed/taxonomy.json labels from the eBay leaf
category; anything unmapped is listed in the report, not guessed.

Usage:
    python scripts/data/ebay_clean.py
"""
import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

LEAF_TO_LABEL = {
    "Oil Filters": "filters",
    "Air Filters": "filters",
    "Cabin Air Filters": "filters",
    "Fuel Filters": "filters",
    "Transmission Filters": "filters",
    "Spark Plugs": "battery and electrical",
    "Ignition Coils": "battery and electrical",
    "Other Ignition Systems & Components": "battery and electrical",
    "Alternators & Generators": "battery and electrical",
    "Air Intake & Fuel Sensors": "engine and powertrain",
    "Idlers & Tensioners": "engine and powertrain",
    "Brake Pads": "braking",
    "Brake Disc Rotors": "braking",
    "Light Bulbs & LEDs": "lighting",
    "Control Arms, Ball Joints & Assemblies": "suspension and steering",
    "Wheel Bearings, Hubs & Seals": "suspension and steering",
    "Shocks, Struts & Assemblies": "suspension and steering",
    "Tie Rods & Steering Linkages": "suspension and steering",
    "Water Pumps": "cooling",
    "Thermostats & Housings": "cooling",
}

# Titles that suggest the photos may show more than one part number. Quantities
# of the same part ("2 pcs", "pack of 2") are fine: same number, same photos.
MULTI_ITEM = re.compile(
    r"front\s*[+&/]\s*rear|rear\s*[+&/]\s*front|\bkit\b|\blot\b|\bbulk\b|\bassorted\b",
    re.IGNORECASE,
)
# Lighting is sold as "kits" of one bulb type (a pair of the same LED), so only the
# front+rear pattern counts there.
MIXED_AXLES = re.compile(
    r"front\s*[+&/]\s*rear|rear\s*[+&/]\s*front",
    re.IGNORECASE,
)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


NUMBER = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{2,}[A-Za-z0-9]")


def other_part_number(title: str, own: str, all_mpns: set[str]) -> str | None:
    """A number in `title` that is another part's number, when the title lacks `own`."""
    if own in norm(title):
        return None
    for tok in NUMBER.findall(title):
        t = norm(tok)
        if t != own and t in all_mpns and re.search(r"\d", t):
            return t
    return None


def safe(part_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", part_id.replace("::", "__"))


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or (list(rows[0].keys()) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/ebay"))
    ap.add_argument("--out", type=Path, default=Path("data/interim/ebay_clean"))
    ap.add_argument("--checked", type=Path, default=Path("data/raw/ebay/label_check_sample.csv"),
                    help="hand-check sheet; parts marked n, m or ? are excluded (skipped if missing)")
    ap.add_argument("--no-copy", action="store_true", help="write the CSVs only, skip copying images")
    a = ap.parse_args()

    catalog = read(a.src / "catalog.csv")
    gallery = defaultdict(list)
    for r in read(a.src / "gallery.csv"):
        gallery[r["part_id"]].append(r)
    queries = defaultdict(list)
    for r in read(a.src / "queries.csv"):
        queries[r["part_id"]].append(r)

    bad_checked: dict[str, str] = {}
    if a.checked.exists():
        for r in read(a.checked):
            if r["label_ok(y/n)"] in {"n", "m", "?"}:
                bad_checked[norm(r["mpn"])] = r["label_ok(y/n)"]
    all_mpns = {norm(c["mpn"]) for c in catalog if len(norm(c["mpn"])) >= 4}
    dropped: list[dict] = []
    unmapped = Counter()
    kept_parts, kept_gallery, kept_queries = [], [], []

    for c in catalog:
        pid = c["part_id"]
        if c["brand"] == "unknown":
            dropped.append({"part_id": pid, "image": "", "reason": "unknown_brand"})
            continue
        own = norm(c["mpn"])
        if own in bad_checked:
            dropped.append({"part_id": pid, "image": "", "reason": f"hand_check_{bad_checked[own]}"})
            continue
        if not re.search(r"\d", own) or own.startswith("update") or len(own) < 4:
            dropped.append({"part_id": pid, "image": "", "reason": "junk_mpn"})
            continue
        conflict = other_part_number(c["title"], own, all_mpns)
        if conflict:
            dropped.append({"part_id": pid, "image": "", "reason": f"gallery_title_names_{conflict}"})
            continue
        leaf = c["category"].split("|")[-1]
        label = LEAF_TO_LABEL.get(leaf)
        if label is None:
            unmapped[leaf] += 1
            dropped.append({"part_id": pid, "image": "", "reason": f"unmapped_category:{leaf}"})
            continue

        mpn = norm(c["mpn"])
        qs = []
        for q in queries[pid]:
            pattern = MIXED_AXLES if label == "lighting" else MULTI_ITEM
            if pattern.search(q["title"]) and mpn not in norm(q["title"]):
                dropped.append({"part_id": pid, "image": q["image"], "reason": "multi_item_title"})
            elif other_part_number(q["title"], mpn, all_mpns):
                dropped.append({"part_id": pid, "image": q["image"], "reason": "query_title_names_other_part"})
            else:
                qs.append(q)
        gs = gallery[pid]
        if not gs or not qs:
            dropped.append({"part_id": pid, "image": "", "reason": "no_gallery_or_no_query_left"})
            continue

        sellers = len({q["seller_hash"] for q in qs} | {g["seller_hash"] for g in gs})
        kept_parts.append({
            "part_id": pid, "part_dir": safe(pid), "brand": c["brand"], "mpn": c["mpn"],
            "title": c["title"], "category": label, "ebay_category": c["category"],
            "n_sellers": sellers, "n_gallery": len(gs), "n_query": len(qs),
            "n_gallery_stock": sum(g["stock"] == "1" for g in gs),
            "short_mpn": int(len(mpn) <= 4),
        })
        kept_gallery += [{**g, "part_dir": safe(pid), "category": label} for g in gs]
        kept_queries += [{**q, "part_dir": safe(pid), "category": label} for q in qs]

    a.out.mkdir(parents=True, exist_ok=True)
    write(a.out / "catalog_clean.csv", kept_parts)
    write(a.out / "gallery_clean.csv", kept_gallery)
    write(a.out / "queries_clean.csv", kept_queries)
    write(a.out / "dropped.csv", dropped, ["part_id", "image", "reason"])

    copied = 0
    if not a.no_copy:
        for kind, rows in (("gallery", kept_gallery), ("queries", kept_queries)):
            for r in rows:
                src = Path(r["image"])
                dst = a.out / kind / r["category"].replace(" ", "_") / r["part_dir"] / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)
                    copied += 1

    by_cat = defaultdict(lambda: {"parts": 0, "gallery": 0, "queries": 0})
    for p in kept_parts:
        by_cat[p["category"]]["parts"] += 1
        by_cat[p["category"]]["gallery"] += p["n_gallery"]
        by_cat[p["category"]]["queries"] += p["n_query"]
    report = {
        "parts_in": len(catalog),
        "parts_kept": len(kept_parts),
        "gallery_images": len(kept_gallery),
        "query_images": len(kept_queries),
        "images_copied": copied,
        "dropped_by_reason": dict(Counter(d["reason"].split(":")[0] for d in dropped)),
        "unmapped_categories": dict(unmapped),
        "short_mpn_parts_kept": sum(p["short_mpn"] for p in kept_parts),
        "by_category": dict(by_cat),
    }
    (a.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Metadata-only catalogue pilot from the MANN-FILTER online catalogue (approved 2026-10-23).

Reads product pages already scraped with Firecrawl into .firecrawl/mann_pilot/<slug>.md and writes
up to 10 candidate rows in the DATASET_SPEC §3 schema, plus a separate image-URL manifest.
No images are downloaded. Every row is verified=false and is_distractor=true, and the output sits
outside data/raw/catalog_images/, so validate_catalog.py never picks it up and nothing is indexed.

Rights: MANN+HUMMEL's legal notice grants no licence (vault note "Catalogue Source Review 2026-10-23").
Keep the output local: do not push data/interim/catalog_pilot_mann to the gdrive remote.

part_number comes only from pipeline.part_number.normalize_part_number. oem_number stays empty:
the pages list OE numbers, but a supplier's cross-reference is not accepted evidence (§4.3).

Usage: python scripts/data/catalog_pilot_mann.py
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.brand_matcher import BrandMatcher
from pipeline.part_number import normalize_part_number

PAGES = ROOT / ".firecrawl" / "mann_pilot"
OUT = ROOT / "data" / "interim" / "catalog_pilot_mann"
BASE_URL = "https://www.mann-filter.com/en/catalog/search-results/product.html/{}_mann-filter.html"
SCRAPED_AT = "2026-10-23"

# slug -> Firecrawl scrape ID, in the order scraped. Only car-application pages are listed;
# cu4795/1 (buses and trucks) and w68/3 (mower tractors first) were scraped and left out.
SCRAPES = {
    "w712/95": "01a167d2-9da6-765d-b691-c46118fda89f",
    "w712/94": "01a167d2-dbac-71f9-8f69-515a2cb64e73",
    "w610/9": "01a167d3-1862-72a8-b606-21b02a745704",
    "hu816x": "01a167d3-91b7-77a7-900a-00e6a03da804",
    "cuk26023": "01a167d3-cb1d-7599-8c8e-1ce25f1f2fd2",
    "cuk2226": "01a167d4-08a2-744e-a535-fddbf89e7839",
    "c26003": "01a167d4-7adc-731b-9782-6b8461d30830",
    "c23018": "01a167d4-b64b-7629-b146-a53c88b1013d",
    "c2620": "01a167d4-ef23-745d-abed-07406f3ba92b",
    "c30009": "01a167d5-2ba4-75ab-9a40-c16aab20b71c",
}

CATALOG_FIELDS = ["part_number_raw", "part_number", "brand", "category", "name", "oem_number",
                  "source_url", "source_site", "collected_by", "is_distractor", "verified", "notes"]
IMAGE_FIELDS = ["part_number_raw", "part_number", "image_url", "view", "source_url",
                "firecrawl_scrape_id", "scraped_at", "rights", "downloaded"]

# Product photos carry alt "Product Image <ID>_MANN-FILTER" or the bare part number; editorial
# photos (workshop scenes, maintenance tips) carry a sentence and are skipped.
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((https://s7g10\.scene7\.com/is/image/mannhummel/[^)\s]+)\)")


def parse_heading(line: str) -> tuple[str, str]:
    """'# Oil Filter    W 712/94' -> ('Oil Filter', 'W 712/94'). The number is kept as shown."""
    parts = re.split(r"\s{2,}", line.lstrip("# ").strip())
    name, number = parts[0].replace("\\|", "|"), parts[1]
    return name, number


def image_view(url: str) -> str:
    stem = url.split("/mannhummel/")[1].split("?")[0]
    if "filter-with-box" in stem:
        return "part_with_box"
    if "-dim" in stem:
        return "dimension_drawing"
    if stem.endswith("-all"):
        return "unclear_check_by_eye"
    return "part"


def parse_page(text: str) -> dict:
    heading = next(line for line in text.splitlines() if line.startswith("# "))
    name, number = parse_heading(heading)
    gtin = re.search(r"GTIN Code:\s*(\d+)", text)
    main_app = re.search(r"Main application:\s*(.+)", text)
    images = list(dict.fromkeys(url for alt, url in _IMAGE_RE.findall(text)
                                if alt.startswith("Product Image") or alt == number))  # unique, page order
    return {"name": name, "part_number_raw": number,
            "gtin": gtin.group(1) if gtin else "",
            "main_application": main_app.group(1).strip() if main_app else "",
            "images": images, "lists_oe_numbers": "### OE Numbers" in text}


def main():
    brand = BrandMatcher().match("MANN-FILTER")
    OUT.mkdir(parents=True, exist_ok=True)
    rows, image_rows = [], []
    for slug, scrape_id in SCRAPES.items():
        page = parse_page((PAGES / f"{slug.replace('/', '_')}.md").read_text(encoding="utf-8"))
        url = BASE_URL.format(slug)
        part_number = normalize_part_number(page["part_number_raw"])
        notes = [f"metadata-only pilot, scraped {SCRAPED_AT} (Firecrawl {scrape_id})",
                 f"GTIN {page['gtin']}" if page["gtin"] else "GTIN not shown",
                 f"main application as listed: {page['main_application']}",
                 "page lists OE numbers; not copied (supplier cross-reference, DATASET_SPEC 4.3)"
                 if page["lists_oe_numbers"] else "no OE numbers listed",
                 "images not downloaded; MANN+HUMMEL grants no licence"]
        rows.append({"part_number_raw": page["part_number_raw"], "part_number": part_number,
                     "brand": brand, "category": "filters", "name": page["name"], "oem_number": "",
                     "source_url": url, "source_site": "www.mann-filter.com",
                     "collected_by": "claude_firecrawl_pilot", "is_distractor": "true",
                     "verified": "false", "notes": "; ".join(notes)})
        for img in page["images"]:
            image_rows.append({"part_number_raw": page["part_number_raw"], "part_number": part_number,
                               "image_url": img, "view": image_view(img), "source_url": url,
                               "firecrawl_scrape_id": scrape_id, "scraped_at": SCRAPED_AT,
                               "rights": "all rights reserved (MANN+HUMMEL legal notice); no permission",
                               "downloaded": "false"})
    for name, fields, data in (("candidates_mann_pilot.csv", CATALOG_FIELDS, rows),
                               ("image_manifest_mann_pilot.csv", IMAGE_FIELDS, image_rows)):
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    print(f"{len(rows)} candidate rows, {len(image_rows)} image URLs -> {OUT}")


if __name__ == "__main__":
    main()

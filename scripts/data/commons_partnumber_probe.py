#!/usr/bin/env python3
"""Feasibility probe: does Wikimedia Commons hold 2+ open-licence photos of the SAME car part, found by its exact model code?

Metadata only: searches file pages for each code (exact phrase), keeps CC0 / public domain / CC BY photos, and CC BY-SA flagged as share_alike (JPEG/PNG),
whose title or description really contains the code, and counts hits per code. Saves 200 px thumbnails of the hits
for codes with 2+ photos so they can be checked by eye. Follows the API etiquette of commons_collect.py
(User-Agent, serial requests, maxlag, back-off).

Usage: python scripts/data/commons_partnumber_probe.py
Writes E:\\car_search_data\\eda\\commons_partnumber_probe\\{hits.csv, summary.csv, thumbs/}
"""
import csv
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commons_collect import UA, api, licence_ok, strip_html

OUT = Path(r"E:\car_search_data\eda\commons_partnumber_probe")
# Common car-part model codes: spark plugs, filters, bulbs, brake pads, belts, sensors, OEM numbers.
CODES = ["NGK BKR6E", "NGK BPR6ES", "NGK BKR5E", "NGK BP6ES", "NGK IFR6T11", "NGK ZFR5F-11", "NGK LFR5A-11",
         "Bosch FR7DC", "Bosch WR7DC", "Denso K20TT", "Denso IK20", "Champion RC12YC", "Champion N9YC",
         "Mann W 712", "Mann W 610/3", "Fram PH3614", "Fram PH8A", "Mobil 1 M1-110", "K&N HP-1010", "Wix 51515",
         "Mahle OC 90", "K&N 33-2304", "K&N 33-2129", "Optima RedTop", "Bosch S4 005", "Bosch Aerotwin",
         "Osram 64193", "Osram 64210", "Philips 12342", "Philips 9005", "Brembo P 85 020", "TRW GDB1330",
         "Ferodo FDB1636", "Gates 5578XS", "Contitech CT1028", "Gates WP0058", "Bosch 0 221 504 470",
         "Bosch 0 258 006 028", "Denso 234-4209", "Wahler 4256", "Bosch 0 580 314 076", "90915-YZZD2",
         "04465-0K240", "17801-0Y040", "15400-PLM-A02", "15208-65F0E", "Motorcraft FL-820S", "AC Delco PF48",
         "06A 115 561 B", "Bosch 0 986 452 041"]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def model_of(code: str) -> str:
    """Drop a leading brand word (no digits, followed by something with digits): "NGK BKR6E" -> "BKR6E".
    Pages often name only the model ("NGK Spark Plug Type: BKR6E"), so searching "NGK BKR6E" finds nothing."""
    brand, _, rest = code.partition(" ")
    return rest if rest and not re.search(r"\d", brand) and re.search(r"\d", rest) else code


def code_in(code: str, text: str) -> bool:
    """The model part of the code must appear in the text, ignoring spaces and punctuation."""
    return norm(model_of(code)) in norm(text)


def main():
    s = requests.Session()
    s.headers["User-Agent"] = UA
    (OUT / "thumbs").mkdir(parents=True, exist_ok=True)
    hits, scanned = defaultdict(list), {}
    for code in CODES:
        d = api(s, {"action": "query", "generator": "search", "gsrsearch": f'"{model_of(code)}" filetype:bitmap', "gsrnamespace": "6",
                    "gsrlimit": "50", "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": "200"})
        pages = d.get("query", {}).get("pages", [])
        scanned[code] = len(pages)
        for p in pages:
            ii = (p.get("imageinfo") or [{}])[0]
            md = ii.get("extmetadata", {})
            lic = (md.get("LicenseShortName") or {}).get("value", "")
            desc = strip_html((md.get("ImageDescription") or {}).get("value", ""))
            if licence_ok(lic, allow_sa=True) and ii.get("mime") in ("image/jpeg", "image/png") and code_in(code, p["title"] + " " + desc):
                hits[code].append({"code": code, "title": p["title"], "licence": lic, "share_alike": not licence_ok(lic), "width": ii.get("width"), "height": ii.get("height"),
                                   "page_url": f"https://commons.wikimedia.org/wiki/{p['title'].replace(' ', '_')}",
                                   "thumb": ii.get("thumburl", ""), "description": desc[:200]})
        print(f"{code}: {len(pages)} results, {len(hits[code])} open-licence photos naming the code", flush=True)
        time.sleep(1)
    rows = [h for c in CODES for h in hits[c]]
    with (OUT / "hits.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["code", "title", "licence", "share_alike", "width", "height", "page_url", "thumb", "description"])
        w.writeheader()
        w.writerows(rows)
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "search_results", "open_photos_naming_code"])
        w.writerows([c, scanned[c], len(hits[c])] for c in CODES)
    for c in CODES:  # thumbnails only for codes that pass the 2+ bar, to check by eye that they show one part
        if len(hits[c]) >= 2:
            for i, h in enumerate(hits[c]):
                if h["thumb"]:
                    r = s.get(h["thumb"], timeout=30)
                    if r.ok:
                        (OUT / "thumbs" / f"{norm(c)}_{i}.jpg").write_bytes(r.content)
                    time.sleep(1.5)
    n2 = sum(len(hits[c]) >= 2 for c in CODES)
    print(f"codes {len(CODES)}; with any open photo {sum(bool(hits[c]) for c in CODES)}; with 2+ {n2}; total photos {len(rows)} -> {OUT}")


if __name__ == "__main__":
    main()

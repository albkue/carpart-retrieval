#!/usr/bin/env python3
"""Collect openly licensed car-part images from Wikimedia Commons (official API).

Only keeps files whose licence allows reuse for training: CC0, public domain, CC BY.
CC BY-SA is kept only with --allow-sa (share-alike). NC and ND are never kept.
Every row records licence, creator and source page, so credit lines can be built later.
Labels are weak: the search query only. Cleaning and labelling come after collection.

Does NOT touch marketplaces (eBay, Amazon, Alibaba) or any site that bans collection.
Set SCRAPER_CONTACT to an email or URL so Wikimedia can reach you (their API etiquette).

Usage:
    python scripts/data/commons_collect.py search   --per-query 150
    python scripts/data/commons_collect.py download --max-side 1024
Output (git-ignored under data/raw/): data/raw/commons/metadata.jsonl, images/, attribution.csv
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from html import unescape
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
# Set CARPARTS_COMMONS_DIR (e.g. E:\car_search_data\raw\commons) to collect outside the repo.
ROOT = Path(os.environ.get("CARPARTS_COMMONS_DIR", REPO / "data" / "raw" / "commons"))
META, IMG, ATTR = ROOT / "metadata.jsonl", ROOT / "images", ROOT / "attribution.csv"
API = "https://commons.wikimedia.org/w/api.php"
UA = f"car-search-thesis/0.1 (student research; {os.environ.get('SCRAPER_CONTACT', 'contact not set')})"
QUERIES = ["oil filter", "air filter", "cabin air filter", "spark plug", "brake pad", "brake disc", "car battery",
           "alternator", "starter motor", "headlight", "shock absorber", "radiator car", "timing belt",
           "wiper blade", "ignition coil", "fuel pump", "water pump car", "clutch disc", "car bulb", "exhaust muffler"]


def licence_ok(short: str, allow_sa: bool = False) -> bool:
    """True for CC0, public domain and CC BY (plus BY-SA if allowed); never NC or ND."""
    s = (short or "").lower().replace("_", " ").strip()
    if not s or "nc" in s.split() or "-nc" in s or " nc" in s or "-nd" in s or " nd" in s:
        return False
    if s.startswith(("cc0", "public domain")) or s in {"pd", "pdm"}:
        return True
    if s.startswith(("cc by-sa", "cc-by-sa")):
        return allow_sa
    return s.startswith(("cc by ", "cc-by-")) or s == "cc by"


def strip_html(x: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", x or ""))).strip()


def api(session, params):
    params = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
    for attempt in range(8):
        r = session.get(API, params=params, timeout=30)
        if r.status_code == 200 and "error" not in r.json():
            return r.json()
        wait = int(r.headers.get("Retry-After", 0)) if r.status_code == 429 else 0
        time.sleep(max(wait, 15 * (attempt + 1)) if r.status_code == 429 else 2 * (attempt + 1))  # back off on 429
    r.raise_for_status()
    sys.exit(f"API error: {r.text[:200]}")


def cmd_search(a):
    ROOT.mkdir(parents=True, exist_ok=True)
    seen = {json.loads(l)["page_url"] for l in META.open(encoding="utf-8")} if META.exists() else set()
    s = requests.Session()
    s.headers["User-Agent"] = UA
    kept = skipped = 0
    with META.open("a", encoding="utf-8") as out:
        for q in a.queries or QUERIES:
            offset, got = 0, 0
            while got < a.per_query:
                d = api(s, {"action": "query", "generator": "search", "gsrsearch": f"{q} filetype:bitmap",
                            "gsrnamespace": "6", "gsrlimit": "50", "gsroffset": offset, "prop": "imageinfo",
                            "iiprop": "url|size|mime|sha1|extmetadata", "iiurlwidth": a.max_side})
                pages = d.get("query", {}).get("pages", [])
                for p in pages:
                    got += 1
                    ii = (p.get("imageinfo") or [{}])[0]
                    md = ii.get("extmetadata", {})
                    lic = (md.get("LicenseShortName") or {}).get("value", "")
                    url = f"https://commons.wikimedia.org/wiki/{p['title'].replace(' ', '_')}"
                    if url in seen or not licence_ok(lic, a.allow_sa) or ii.get("mime") not in ("image/jpeg", "image/png"):
                        skipped += 1
                        continue
                    if min(ii.get("width", 0), ii.get("height", 0)) < a.min_side:
                        skipped += 1
                        continue
                    seen.add(url)
                    kept += 1
                    out.write(json.dumps({
                        "page_url": url, "download_url": ii.get("thumburl") or ii["url"], "sha1": ii.get("sha1"),
                        "width": ii.get("width"), "height": ii.get("height"), "query": q, "title": p["title"],
                        "licence": lic, "licence_url": (md.get("LicenseUrl") or {}).get("value", ""),
                        "creator": strip_html((md.get("Artist") or {}).get("value", "")),
                        "credit": strip_html((md.get("Credit") or {}).get("value", "")),
                    }) + "\n")
                if "continue" not in d:
                    break
                offset = d["continue"]["gsroffset"]
                time.sleep(1)
            print(f"{q}: scanned {got}, kept so far {kept}", flush=True)
    print(f"search done: kept {kept}, skipped {skipped} (licence, size or type) -> {META}")


def cmd_download(a):
    IMG.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in META.open(encoding="utf-8")]
    s = requests.Session()
    s.headers["User-Agent"] = UA
    ok = skip = fail = 0
    seen_sha = set()
    for i, r in enumerate(rows):
        ext = ".png" if r["download_url"].lower().endswith(".png") else ".jpg"
        path = IMG / (hashlib.sha1(r["page_url"].encode()).hexdigest()[:16] + ext)
        r["file"] = path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else path.as_posix()
        if r["sha1"] in seen_sha:
            r["duplicate"] = True
        seen_sha.add(r["sha1"])
        if path.exists():
            skip += 1
            continue
        try:
            for attempt in range(6):
                resp = s.get(r["download_url"], timeout=60)
                if resp.status_code != 429:
                    break
                time.sleep(max(int(resp.headers.get("Retry-After", 0)), 20 * (attempt + 1)))  # back off on 429
            resp.raise_for_status()
            path.write_bytes(resp.content)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"fail {r['page_url']}: {e}")
        time.sleep(1.5)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(rows)} ok {ok} skip {skip} fail {fail}", flush=True)
    with META.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with ATTR.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "title", "creator", "licence", "licence_url", "source"])
        for r in rows:
            w.writerow([r.get("file"), r["title"], r["creator"], r["licence"], r["licence_url"], r["page_url"]])
    print(f"download done: ok {ok} skip {skip} fail {fail} -> {IMG}; credits -> {ATTR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    se = sub.add_parser("search")
    se.add_argument("--queries", nargs="*")
    se.add_argument("--per-query", type=int, default=150)
    se.add_argument("--min-side", type=int, default=300)
    se.add_argument("--max-side", type=int, default=1024)
    se.add_argument("--allow-sa", action="store_true", help="also keep CC BY-SA (share-alike)")
    se.set_defaults(func=cmd_search)
    dl = sub.add_parser("download")
    dl.add_argument("--max-side", type=int, default=1024)
    dl.set_defaults(func=cmd_download)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()

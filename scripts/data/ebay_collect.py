"""
eBay car-part catalog collector (backup data plan for the car-part retrieval thesis).

Builds a catalog of (images, part number, brand, metadata) from eBay listings and
splits it into GALLERY (catalog) and QUERY (real seller photos) sets, grouped by
manufacturer part number (MPN).

Pipeline (each step is resumable; re-run safely):
  1. search    -> data/raw/ebay/raw/summaries.jsonl   (item ids from Browse API search)
  2. details   -> data/raw/ebay/raw/items.jsonl       (full item: aspects, images, seller)
  3. download  -> data/raw/ebay/images/<item_id>_<k>.jpg
  4. build     -> data/raw/ebay/catalog.csv, gallery.csv, queries.csv, build_report.json

Setup:
  pip install requests pillow imagehash
  Create an eBay developer account (developer.ebay.com) -> Production keyset.
  Windows PowerShell:
    $env:EBAY_CLIENT_ID="..." ; $env:EBAY_CLIENT_SECRET="..."

Example (run from the repo root, inside .venv):
  python scripts/data/ebay_collect.py search   --query "oil filter" --max-items 3000
  python scripts/data/ebay_collect.py search   --query "spark plug" --max-items 3000
  python scripts/data/ebay_collect.py details
  python scripts/data/ebay_collect.py download --max-images-per-item 4
  python scripts/data/ebay_collect.py build    --min-sellers 3

Research-use notes:
  - Uses the official Browse API (default quota ~5,000 calls/day). `details` makes one
    call per item, so spread large runs over several days; it resumes where it stopped.
  - Keep images private for research; cite eBay as source; do not redistribute.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

API = "https://api.ebay.com"
TOKEN_URL = f"{API}/identity/v1/oauth2/token"
SCOPE = "https://api.ebay.com/oauth/api_scope"
PARTS_ACCESSORIES_CATEGORY = "6030"  # eBay Motors > Parts & Accessories

ROOT = Path("data/raw/ebay")  # data/raw is git-ignored
RAW = ROOT / "raw"
IMG = ROOT / "images"
SUMMARIES = RAW / "summaries.jsonl"
ITEMS = RAW / "items.jsonl"
FAILED = RAW / "failed_ids.txt"

MPN_KEYS = ("Manufacturer Part Number", "MPN", "OE/OEM Part Number", "Part Number")
BRAND_KEYS = ("Brand",)
JUNK_MPN = {"", "NA", "NONE", "DOESNOTAPPLY", "NOTAPPLICABLE", "UNKNOWN", "UNBRANDED", "SEEDESCRIPTION", "0"}


# --------------------------------------------------------------------------- utils
def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a half-written last line after a crash


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def normalize_mpn(raw: str | None) -> str:
    if not raw:
        return ""
    # Some listings put several numbers in one field: keep the first.
    first = re.split(r"[,;/|]| or ", raw, maxsplit=1)[0]
    norm = re.sub(r"[^A-Z0-9]", "", first.upper())
    if norm in JUNK_MPN or len(norm) < 3:
        return ""
    return norm


def normalize_brand(raw: str | None) -> str:
    b = re.sub(r"\s+", " ", (raw or "").strip().lower())
    return "" if b in {"", "unbranded", "does not apply", "n/a", "generic"} else b


def hash_seller(name: str | None) -> str:
    """Store only a hash of the seller username, never the raw value."""
    return hashlib.sha1((name or "unknown").encode()).hexdigest()[:12]


def aspect(item: dict, keys) -> str:
    for a in item.get("localizedAspects") or []:
        if a.get("name") in keys and a.get("value"):
            return a["value"]
    return ""


# --------------------------------------------------------------------------- API client
class EbayClient:
    def __init__(self, marketplace: str = "EBAY_US"):
        self.cid = os.environ.get("EBAY_CLIENT_ID")
        self.secret = os.environ.get("EBAY_CLIENT_SECRET")
        if not self.cid or not self.secret:
            sys.exit("Set EBAY_CLIENT_ID and EBAY_CLIENT_SECRET environment variables.")
        self.marketplace = marketplace
        self.session = requests.Session()
        self._token, self._exp = None, 0.0

    def token(self) -> str:
        if self._token and time.time() < self._exp - 120:
            return self._token
        basic = base64.b64encode(f"{self.cid}:{self.secret}".encode()).decode()
        r = self.session.post(
            TOKEN_URL,
            headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "client_credentials", "scope": SCOPE},
            timeout=30,
        )
        r.raise_for_status()
        j = r.json()
        self._token, self._exp = j["access_token"], time.time() + int(j.get("expires_in", 7200))
        return self._token

    def get(self, path: str, params: dict | None = None, retries: int = 5) -> dict | None:
        for attempt in range(retries):
            try:
                r = self.session.get(
                    API + path,
                    params=params,
                    headers={"Authorization": f"Bearer {self.token()}", "X-EBAY-C-MARKETPLACE-ID": self.marketplace},
                    timeout=30,
                )
            except requests.RequestException as e:
                log(f"network error: {e}; retry {attempt + 1}")
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 401:
                self._token = None
                continue
            if r.status_code == 404:
                return None  # listing ended / removed
            if r.status_code == 429:
                sys.exit("Daily API quota reached (HTTP 429). Re-run tomorrow; progress is saved.")
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            log(f"HTTP {r.status_code} on {path}: {r.text[:300]}")
            return None
        return None


# --------------------------------------------------------------------------- step 1: search
def cmd_search(a) -> None:
    client = EbayClient(a.marketplace)
    seen = {s["itemId"] for s in read_jsonl(SUMMARIES)}
    log(f"{len(seen)} item ids already saved")
    added, offset, page = 0, 0, 200
    while added < a.max_items and offset < 10000:  # Browse API caps offset at 10k
        params = {
            "q": a.query,
            "category_ids": a.category,
            "limit": page,
            "offset": offset,
            "filter": "buyingOptions:{FIXED_PRICE|AUCTION}",
        }
        if a.condition:
            params["filter"] += f",conditions:{{{a.condition}}}"
        res = client.get("/buy/browse/v1/item_summary/search", params)
        rows = (res or {}).get("itemSummaries") or []
        if not rows:
            break
        for s in rows:
            if s["itemId"] in seen:
                continue
            seen.add(s["itemId"])
            append_jsonl(SUMMARIES, {
                "itemId": s["itemId"],
                "query": a.query,
                "title": s.get("title"),
                "seller": hash_seller((s.get("seller") or {}).get("username")),
                "categoryId": (s.get("categories") or [{}])[0].get("categoryId"),
            })
            added += 1
            if added >= a.max_items:
                break
        offset += page
        log(f"offset {offset}: +{added} new for '{a.query}'")
        time.sleep(a.sleep)
    log(f"done: {added} new items for '{a.query}'")


# --------------------------------------------------------------------------- step 2: details
def cmd_details(a) -> None:
    client = EbayClient(a.marketplace)
    done = {i["itemId"] for i in read_jsonl(ITEMS)}
    failed = set(FAILED.read_text().split()) if FAILED.exists() else set()
    todo = [s for s in read_jsonl(SUMMARIES) if s["itemId"] not in done and s["itemId"] not in failed]
    if a.limit:
        todo = todo[: a.limit]
    log(f"{len(done)} done, {len(todo)} to fetch")
    for n, s in enumerate(todo, 1):
        item = client.get(f"/buy/browse/v1/item/{s['itemId']}")
        if not item:
            with FAILED.open("a") as f:
                f.write(s["itemId"] + "\n")
            continue
        images = [(item.get("image") or {}).get("imageUrl")]
        images += [x.get("imageUrl") for x in item.get("additionalImages") or []]
        mpn_raw = item.get("mpn") or aspect(item, MPN_KEYS)
        brand_raw = item.get("brand") or aspect(item, BRAND_KEYS)
        append_jsonl(ITEMS, {
            "itemId": item["itemId"],
            "query": s.get("query"),
            "title": item.get("title"),
            "seller": hash_seller((item.get("seller") or {}).get("username")),
            "condition": item.get("condition"),
            "categoryPath": item.get("categoryPath"),
            "mpn_raw": mpn_raw,
            "mpn": normalize_mpn(mpn_raw),
            "brand_raw": brand_raw,
            "brand": normalize_brand(brand_raw),
            "aspects": {x["name"]: x.get("value") for x in item.get("localizedAspects") or []},
            "image_urls": [u for u in images if u],
            "itemWebUrl": item.get("itemWebUrl"),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        if n % 50 == 0:
            log(f"{n}/{len(todo)} fetched")
        time.sleep(a.sleep)
    log("details done")


# --------------------------------------------------------------------------- step 3: download
def cmd_download(a) -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    items = [i for i in read_jsonl(ITEMS) if i.get("mpn")] if a.only_with_mpn else list(read_jsonl(ITEMS))
    log(f"{len(items)} items to download images for")
    ok = skip = fail = 0
    for n, it in enumerate(items, 1):
        for k, url in enumerate(it["image_urls"][: a.max_images_per_item]):
            out = IMG / f"{it['itemId'].replace('|', '_')}_{k}.jpg"
            if out.exists() and out.stat().st_size > 0:
                skip += 1
                continue
            try:
                r = sess.get(url, timeout=30)
                r.raise_for_status()
                out.write_bytes(r.content)
                ok += 1
            except requests.RequestException:
                fail += 1
            time.sleep(a.sleep)
        if n % 100 == 0:
            log(f"{n}/{len(items)} items | ok {ok} skip {skip} fail {fail}")
    log(f"download done | ok {ok} skip {skip} fail {fail}")


# --------------------------------------------------------------------------- step 4: build
def cmd_build(a) -> None:
    out_dir = Path(a.out) if a.out else ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        import imagehash
    except ImportError:
        sys.exit("pip install pillow imagehash")

    items = [i for i in read_jsonl(ITEMS) if i.get("mpn")]
    # Key = brand + MPN so different brands sharing a number don't merge.
    # --key mpn_long: a long part number (>= --min-mpn-len characters) is treated as one
    # part whatever brand the seller typed ("denso", "toyota", "factory/oem" ...); its
    # label uses the most common brand. Short numbers still need the brand to tell
    # parts apart.
    brand_votes: dict[str, defaultdict] = defaultdict(lambda: defaultdict(int))
    for it in items:
        brand_votes[it["mpn"]][it["brand"] or "unknown"] += 1
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        brand = it["brand"] or "unknown"
        if a.key == "mpn_long" and len(it["mpn"]) >= a.min_mpn_len:
            votes = brand_votes[it["mpn"]]
            known = {b: n for b, n in votes.items() if b != "unknown"}
            brand = max(known or votes, key=lambda b: (known or votes)[b])
        groups[f"{brand}::{it['mpn']}"].append(it)

    # Perceptual hash of every image on disk.
    records = []  # (part_key, item, path, phash)
    bad_images = 0
    for key, its in groups.items():
        for it in its:
            for p in sorted(IMG.glob(f"{it['itemId'].replace('|', '_')}_*.jpg")):
                try:
                    with Image.open(p) as im:
                        im = im.convert("RGB")
                        if min(im.size) < a.min_side:
                            continue
                        h = imagehash.phash(im)
                except Exception:
                    bad_images += 1
                    continue
                records.append((key, it, p, h))

    # Flag "stock" photos: near-duplicates seen under 2+ different sellers anywhere.
    # Stock photos are fine for the gallery but must never be queries (they'd be trivial).
    # Single pass: assign each image to a near-duplicate cluster (exact-hash lookup first, fast).
    clusters: list[tuple] = []          # (representative hash, set of sellers)
    exact: dict[str, int] = {}
    rec_cluster: list[int] = []
    for _, it, _, h in records:
        cid = exact.get(str(h))
        if cid is None:
            cid = next((i for i, (hh, _) in enumerate(clusters) if h - hh <= a.dup_dist), None)
            if cid is None:
                clusters.append((h, set()))
                cid = len(clusters) - 1
            exact[str(h)] = cid
        clusters[cid][1].add(it["seller"])
        rec_cluster.append(cid)

    by_part: dict[str, list] = defaultdict(list)
    for (key, it, p, h), cid in zip(records, rec_cluster):
        by_part[key].append((it, p, h, len(clusters[cid][1]) > 1))

    rng = random.Random(a.seed)
    gallery_rows, query_rows, catalog_rows = [], [], []
    dropped_few_sellers = 0
    for key in sorted(by_part):
        rows = by_part[key]
        sellers = sorted({r[0]["seller"] for r in rows})
        if len(sellers) < a.min_sellers:
            dropped_few_sellers += 1
            continue
        # Gallery seller: the one with most images (ties broken by seeded shuffle).
        rng.shuffle(sellers)
        g_seller = max(sellers, key=lambda s: sum(r[0]["seller"] == s for r in rows))
        g_rows = [r for r in rows if r[0]["seller"] == g_seller]
        g_hashes = [r[2] for r in g_rows]
        q_rows = [
            r for r in rows
            if r[0]["seller"] != g_seller
            and not r[3]  # not a stock photo
            and all(r[2] - gh > a.dup_dist for gh in g_hashes)  # not a copy of a gallery image
        ]
        # Dedupe queries among themselves.
        uniq, kept = [], []
        for r in q_rows:
            if all(r[2] - u > a.dup_dist for u in uniq):
                uniq.append(r[2])
                kept.append(r)
        if not kept:
            dropped_few_sellers += 1
            continue

        brand, mpn = key.split("::", 1)
        any_item = g_rows[0][0]
        catalog_rows.append({
            "part_id": key, "brand": brand, "mpn": mpn,
            "title": any_item["title"], "query_term": any_item.get("query"),
            "category": any_item.get("categoryPath"),
            "n_sellers": len(sellers), "n_gallery": len(g_rows), "n_query": len(kept),
        })
        for it, p, h, stock in g_rows:
            gallery_rows.append({"part_id": key, "image": p.as_posix(), "item_id": it["itemId"],
                                 "seller_hash": hashlib.sha1(it["seller"].encode()).hexdigest()[:10],
                                 "phash": str(h), "stock": int(stock)})
        for it, p, h, _ in kept:
            query_rows.append({"part_id": key, "image": p.as_posix(), "item_id": it["itemId"],
                               "seller_hash": hashlib.sha1(it["seller"].encode()).hexdigest()[:10],
                               "phash": str(h), "title": it["title"]})

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(out_dir / "catalog.csv", catalog_rows)
    write_csv(out_dir / "gallery.csv", gallery_rows)
    write_csv(out_dir / "queries.csv", query_rows)
    report = {
        "items_with_mpn": len(items),
        "candidate_parts": len(groups),
        "images_hashed": len(records),
        "unreadable_images": bad_images,
        "parts_dropped_lt_min_sellers_or_no_queries": dropped_few_sellers,
        "final_parts": len(catalog_rows),
        "gallery_images": len(gallery_rows),
        "query_images": len(query_rows),
        "params": vars(a) | {"func": None},
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2, default=str))
    log(json.dumps({k: v for k, v in report.items() if k != "params"}, indent=2))


# --------------------------------------------------------------------------- CLI
def main() -> None:
    p = argparse.ArgumentParser(description="eBay car-part catalog collector")
    p.add_argument("--marketplace", default="EBAY_US")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("search", help="collect item ids for a query")
    s.add_argument("--query", required=True)
    s.add_argument("--category", default=PARTS_ACCESSORIES_CATEGORY)
    s.add_argument("--condition", default="", help="NEW or USED; empty = any (used listings give more real photos)")
    s.add_argument("--max-items", type=int, default=3000)
    s.add_argument("--sleep", type=float, default=0.3)
    s.set_defaults(func=cmd_search)

    d = sub.add_parser("details", help="fetch full item details (1 API call per item)")
    d.add_argument("--limit", type=int, default=0, help="max items this run (0 = all)")
    d.add_argument("--sleep", type=float, default=0.2)
    d.set_defaults(func=cmd_details)

    dl = sub.add_parser("download", help="download listing images")
    dl.add_argument("--max-images-per-item", type=int, default=4)
    dl.add_argument("--only-with-mpn", action=argparse.BooleanOptionalAction, default=True)
    dl.add_argument("--sleep", type=float, default=0.05)
    dl.set_defaults(func=cmd_download)

    b = sub.add_parser("build", help="dedupe and split into gallery/query")
    b.add_argument("--min-sellers", type=int, default=3)
    b.add_argument("--dup-dist", type=int, default=6, help="max pHash Hamming distance for near-duplicates")
    b.add_argument("--min-side", type=int, default=200, help="drop images smaller than this (px)")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--key", choices=["brand_mpn", "mpn_long"], default="brand_mpn",
                   help="how to group listings into one part (see cmd_build)")
    b.add_argument("--min-mpn-len", type=int, default=7, help="shortest part number merged across brands")
    b.add_argument("--out", default="", help="write catalog/gallery/queries/build_report here (default data/raw/ebay)")
    b.set_defaults(func=cmd_build)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()

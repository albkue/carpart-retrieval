#!/usr/bin/env python3
"""Round 2: big Wikimedia Commons scrape for the 20 car-part classes. No splits, no CLIP resizing.

Per class: search 12 query variants (official API), record licence-OK candidates up to --quota, then
download, validate (magic bytes, >= 320 px, RGB, alpha on white), reject near-duplicates against round 1 and
round 2 (pHash Hamming <= 6), score with the CLIP relevance filter (>= 0.85) and keep accepted originals in
raw_round2/<class>/. Stops a class at --soft accepted; if the quota is spent with fewer, extends to --hard
candidates. Rejects keep a row and a reason (the file itself is not kept). Round 1 files are never touched.
Resumable: reruns skip every URL already in the CSVs and classes that reached --soft.

Usage: python scripts/data/commons_round2.py            (set SCRAPER_CONTACT)
"""
import argparse
import csv
import io
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import imagehash
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_clip_dataset import FORMATS, magic_format, to_rgb
from commons_collect import UA, api, licence_ok, strip_html
from commons_filter import NEG

DATA = Path(r"E:\car_search_data")
# class -> search/CLIP noun. Priority classes first. The 20 classes are the round-1 set (no other list exists in the vault).
NAMES = {"water pump": "water pump", "timing belt": "timing belt", "bulb": "car light bulb",
         "cabin air filter": "cabin air filter", "clutch disc": "clutch disc", "fuel pump": "fuel pump",
         "starter motor": "starter motor", "spark plug": "spark plug", "brake disc": "brake disc",
         "radiator": "car radiator", "air filter": "air filter", "shock absorber": "shock absorber",
         "headlight": "headlight", "oil filter": "oil filter", "battery": "car battery", "brake pad": "brake pad",
         "alternator": "alternator", "exhaust muffler": "exhaust muffler", "wiper blade": "wiper blade",
         "ignition coil": "ignition coil"}
SUFFIXES = ["", "car", "automobile", "vehicle", "spare part", "replacement part", "diagram", "exploded view",
            "catalog", "workshop", "disassembled", "close-up"]
MIMES = {"image/jpeg", "image/png", "image/webp"}
HDR = {"candidates": ["class", "query", "page_url", "download_url", "title", "author", "licence", "licence_url",
                      "attribution", "width", "height", "file_size_bytes", "mime"],
       "accepted": ["image_id", "class", "query", "page_url", "download_url", "title", "author", "licence", "licence_url",
                    "attribution", "width", "height", "file_size_bytes", "format", "downloaded_at", "file_path", "phash", "clip_relevance"],
       "rejected": ["class", "query", "page_url", "title", "licence", "reason", "clip_relevance"],
       "duplicates": ["class", "page_url", "dropped_phash", "matches", "hamming"]}


def variants(name: str) -> list:
    """The 12 query variants; a suffix already inside the name is skipped (no 'car radiator car')."""
    return [f"{name} {s}".strip() for s in SUFFIXES if not s or s not in name.split()]


def near_dup(ph: int, hashes: list, max_dist=6):
    """(label, distance) of the first stored hash within max_dist, else None."""
    for h, label in hashes:
        d = (ph ^ h).bit_count()
        if d <= max_dist:
            return label, d
    return None


class Table:
    def __init__(self, path: Path, header: list):
        self.path, self.header = path, header
        self.rows = list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)

    def add(self, row: dict):
        self.rows.append({k: str(row.get(k, "")) for k in self.header})
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self.header])


class Clip:
    def __init__(self, model_id="openai/clip-vit-large-patch14"):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = CLIPModel.from_pretrained(model_id).to(self.dev).eval()
        self.p = CLIPProcessor.from_pretrained(model_id)
        self.tok = {}

    def score(self, im: Image.Image, cls: str) -> float:
        if cls not in self.tok:
            part = NAMES[cls]
            texts = [f"a photo of a {part}, a car part", f"a product photo of a {part}"] + NEG
            self.tok[cls] = self.p(text=texts, return_tensors="pt", padding=True).to(self.dev)
        with torch.no_grad():
            px = self.p(images=im, return_tensors="pt")["pixel_values"].to(self.dev)
            return float(self.m(**self.tok[cls], pixel_values=px).logits_per_image.softmax(-1)[0][:2].sum())


def backoff_get(s, url):
    for n in range(7):
        try:
            r = s.get(url, timeout=60)
        except requests.RequestException:
            time.sleep(min(5 * 2 ** n, 120))
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            try:
                ra = int(r.headers.get("Retry-After", 0))
            except ValueError:
                ra = 0
            time.sleep(max(ra, min(5 * 2 ** n, 120)))
            continue
        r.raise_for_status()
        return r.content
    raise RuntimeError("retries exhausted")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DATA / "dataset")
    ap.add_argument("--classes", nargs="*", default=list(NAMES))
    ap.add_argument("--quota", type=int, default=300)
    ap.add_argument("--hard", type=int, default=400)
    ap.add_argument("--soft", type=int, default=80)
    ap.add_argument("--threshold", type=float, default=0.85)
    a = ap.parse_args()
    md = a.out / "metadata"
    T = {k: Table(md / f"{k}_round2.csv", HDR[k]) for k in HDR}
    import json
    old_urls = {json.loads(l)["page_url"] for l in (DATA / "raw/commons/metadata.jsonl").open(encoding="utf-8")}
    hashes = []  # (int phash, label): round 1 accepted raw files, then round 2 accepted
    for f in (a.out / "raw").rglob("*"):
        if f.is_file():
            with Image.open(f) as im:
                hashes.append((int(str(imagehash.phash(to_rgb(im))), 16), f"round1/{f.name}"))
    hashes += [(int(r["phash"], 16), r["image_id"]) for r in T["accepted"].rows]
    done = old_urls | {r["page_url"] for r in T["accepted"].rows} | {r["page_url"] for r in T["rejected"].rows} \
        | {r["page_url"] for r in T["duplicates"].rows}
    cand_urls = {r["page_url"] for r in T["candidates"].rows}
    print(f"round1 hashes+round2 so far {len(hashes)}, urls already seen {len(done)}", flush=True)
    s = requests.Session()
    s.headers["User-Agent"] = UA
    clip = Clip()

    def candidates(cls):
        for q in variants(NAMES[cls]):
            offset = 0
            for _ in range(3):  # 3 pages x 50 per variant spreads the quota over the variants
                d = api(s, {"action": "query", "generator": "search", "gsrsearch": f"{q} filetype:bitmap", "gsrnamespace": "6",
                            "gsrlimit": "50", "gsroffset": offset, "prop": "imageinfo", "iiprop": "url|size|mime|sha1|extmetadata",
                            "iiurlwidth": 1024})
                for p in d.get("query", {}).get("pages", []):
                    ii = (p.get("imageinfo") or [{}])[0]
                    ext = ii.get("extmetadata", {})
                    val = lambda k, ext=ext: (ext.get(k) or {}).get("value", "")
                    url = f"https://commons.wikimedia.org/wiki/{p['title'].replace(' ', '_')}"
                    if url in done:
                        continue
                    base = {"class": cls, "query": q, "page_url": url, "title": p["title"], "licence": val("LicenseShortName")}
                    why = None
                    if not ii:
                        why = "no_imageinfo"
                    elif not licence_ok(val("LicenseShortName")):
                        why = "licence_not_allowed"
                    elif ii.get("mime") not in MIMES:
                        why = f"mime_{ii.get('mime')}"
                    elif min(ii.get("width", 0), ii.get("height", 0)) < 320:
                        why = "below_320px_api"
                    if why:
                        done.add(url)
                        T["rejected"].add({**base, "reason": why})
                        continue
                    done.add(url)
                    yield {**base, "download_url": ii.get("thumburl") or ii["url"], "author": strip_html(val("Artist")),
                           "licence_url": val("LicenseUrl"), "attribution": strip_html(val("Attribution") or val("Credit")),
                           "width": ii.get("width"), "height": ii.get("height"), "file_size_bytes": ii.get("size"), "mime": ii.get("mime")}
                if "gsroffset" not in d.get("continue", {}):  # absent on the last page or on an imageinfo-only continue
                    break
                offset = d["continue"]["gsroffset"]
                time.sleep(1)
            time.sleep(1)

    def handle(c):
        """Download and run every check; one row lands in accepted, rejected or duplicates."""
        rej = lambda why, sc="": T["rejected"].add({**c, "reason": why, "clip_relevance": sc})
        try:
            raw = backoff_get(s, c["download_url"])
        except Exception as e:
            return rej(f"download_failed:{type(e).__name__}")
        time.sleep(1)
        fmt = magic_format(raw[:512])
        if fmt not in FORMATS:
            return rej(f"format_{fmt}")
        try:
            with Image.open(io.BytesIO(raw)) as im:
                im.load()
                w, h = im.size
                rgb = to_rgb(im)
        except Exception:
            return rej("corrupt_or_unreadable")
        if min(w, h) < 320:
            return rej("below_320px")
        ph = int(str(imagehash.phash(rgb)), 16)
        hit = near_dup(ph, hashes)
        if hit:
            T["duplicates"].add({"class": c["class"], "page_url": c["page_url"], "dropped_phash": f"{ph:016x}", "matches": hit[0], "hamming": hit[1]})
            return rej("near_duplicate")
        sc = clip.score(rgb, c["class"])
        if sc < a.threshold:
            return rej(f"off_topic_clip<{a.threshold}", f"{sc:.3f}")
        n = sum(r["class"] == c["class"] for r in T["accepted"].rows) + 1
        iid = f"{c['class'].replace(' ', '_')}_r2_{n:04d}"
        dst = a.out / "raw_round2" / c["class"].replace(" ", "_") / (iid + {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}[fmt])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(raw)  # original bytes as served
        hashes.append((ph, iid))
        T["accepted"].add({**c, "image_id": iid, "format": fmt, "width": w, "height": h, "file_size_bytes": len(raw),
                           "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "file_path": dst.as_posix(), "phash": f"{ph:016x}", "clip_relevance": f"{sc:.3f}"})

    def report():
        acc, rej_, cnd = T["accepted"].rows, T["rejected"].rows, T["candidates"].rows
        lines = [(f"# Round 2 scrape progress\n\nUpdated {datetime.now().astimezone().isoformat(timespec='seconds')}. Commons API only, CC0, public domain, CC BY. "
                 f"Soft stop {a.soft} accepted, quota {a.quota}, hard stop {a.hard} candidates per class, CLIP >= {a.threshold}.\n"),
                 "| class | candidates | accepted | rejected | near-dup |\n|---|---|---|---|---|"]
        for cls in NAMES:
            lines.append(f"| {cls} | {sum(r['class'] == cls for r in cnd)} | {sum(r['class'] == cls for r in acc)} | "
                         f"{sum(r['class'] == cls and r['reason'] != 'licence_not_allowed' for r in rej_)} | "
                         f"{sum(r['class'] == cls for r in T['duplicates'].rows)} |")
        reasons = Counter(r["reason"].split(":")[0] for r in rej_)
        low = [c for c in NAMES if sum(r["class"] == c for r in acc) < 50]
        lines += [f"\nRejection reasons (all, incl. API-level): {dict(reasons)}", f"Licences accepted: {dict(Counter(r['licence'] for r in acc))}",
                  f"Duplicates: {len(T['duplicates'].rows)}", f"Classes below 50 accepted: {low or 'none'}"]
        (a.out / "reports").mkdir(exist_ok=True)
        (a.out / "reports/round2_scrape_progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for cls in sorted(a.classes, key=list(NAMES).index):
        acc = lambda cls=cls: sum(r["class"] == cls for r in T["accepted"].rows)
        total = lambda cls=cls: sum(r["class"] == cls for r in T["candidates"].rows)
        if acc() >= a.soft:
            continue
        gen, state = candidates(cls), {"exhausted": False}

        def pull(limit, total=total, gen=gen, state=state):
            items = []
            while total() < limit:
                c = next(gen, None)
                if c is None:
                    state["exhausted"] = True
                    break
                if c["page_url"] not in cand_urls:
                    cand_urls.add(c["page_url"])
                    T["candidates"].add(c)
                items.append(c)
            return items
        batch = pull(a.quota)
        while True:
            for c in batch:
                if acc() >= a.soft:
                    break
                handle(c)
            if acc() >= a.soft or state["exhausted"] or total() >= a.hard:
                break
            batch = pull(min(total() + 25, a.hard))
            if not batch:
                break
        print(f"{cls}: candidates {total()}, accepted {acc()}", flush=True)
        report()
    report()


if __name__ == "__main__":
    main()

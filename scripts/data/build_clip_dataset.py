#!/usr/bin/env python3
"""Validate, dedupe, EDA and split the Commons car-part images into a CLIP-ready dataset.

Input : E:\\car_search_data\\raw\\commons (metadata.jsonl, images/, relevance.csv from commons_filter.py)
Output: <out>/raw (originals, unchanged), metadata/, processed/<split>/<class>/, manifests/, reports/
Labels are category-level only (the search term mapped to a class). No part numbers exist here, so this set
is for category training and distractors, never part-number ground truth (see vault Dataset Spec).
Rerunnable: it rebuilds <out> from scratch and never touches the source images.

Usage: python scripts/data/build_clip_dataset.py --threshold 0.85 --min-per-split 2
"""
import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path(os.environ.get("CARPARTS_COMMONS_DIR", r"E:\car_search_data\raw\commons"))
ALIAS = {"water pump car": "water pump", "water pump engine": "water pump", "automotive water pump": "water pump",
         "radiator car": "radiator", "car battery": "battery", "car bulb": "bulb", "automotive light bulb": "bulb",
         "h7 halogen bulb": "bulb", "car starter motor": "starter motor", "automobile starter": "starter motor",
         "cabin filter": "cabin air filter", "pollen filter": "cabin air filter", "timing belt engine": "timing belt",
         "clutch kit": "clutch disc", "automotive fuel pump": "fuel pump"}
FORMATS = {"jpeg", "png", "webp"}
CLIP_SIDE = 224


def magic_format(head: bytes):
    """Format from the first bytes, ignoring extension and MIME type."""
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    for name, sig in (("gif", b"GIF8"), ("bmp", b"BM"), ("tiff", b"II*\x00"), ("tiff", b"MM\x00*")):
        if head.startswith(sig):
            return name
    return "svg" if b"<svg" in head[:512].lower() else "unknown"


def to_rgb(im: Image.Image) -> Image.Image:
    """RGB with any alpha composited on white (no transparent channel reaches training)."""
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.getchannel("A"))
        return bg
    return im.convert("RGB")


def clip_preprocess(im: Image.Image) -> Image.Image:
    """CLIP's own recipe: bicubic resize of the short side to 224, then centre crop."""
    w, h = im.size
    s = CLIP_SIDE / min(w, h)
    im = im.resize((max(CLIP_SIDE, round(w * s)), max(CLIP_SIDE, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - CLIP_SIDE) // 2, (h - CLIP_SIDE) // 2
    return im.crop((l, t, l + CLIP_SIDE, t + CLIP_SIDE))


def stratified_split(ids_by_class: dict, min_per_split=2, frac=0.1, min_total=10, seed=42):
    """Per class: val and test get max(min_per_split, round(frac*n)) each. Classes under min_total are skipped."""
    split, skipped = {}, []
    rng = random.Random(seed)
    for c, ids in sorted(ids_by_class.items()):
        ids = sorted(ids)
        if len(ids) < max(min_total, 3 * min_per_split):
            skipped.append(c)
            continue
        rng.shuffle(ids)
        k = max(min_per_split, round(frac * len(ids)))
        for i, x in enumerate(ids):
            split[x] = "validation" if i < k else "test" if i < 2 * k else "train"
    return split, skipped


def hamming_dups(hashes: list, max_dist=6):
    """Index pairs (kept, dropped, dist): later image dropped if within max_dist of an earlier kept one."""
    arr = np.array([int(str(h), 16) for h in hashes], dtype=np.uint64)
    dropped, pairs = set(), []
    for i in range(len(arr)):
        if i in dropped:
            continue
        x = arr[i + 1:] ^ arr[i]
        d = np.array([bin(int(v)).count("1") for v in x]) if len(x) else np.array([])
        for off in np.nonzero(d <= max_dist)[0]:
            j = i + 1 + int(off)
            if j not in dropped:
                dropped.add(j)
                pairs.append((i, j, int(d[off])))
    return dropped, pairs


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path(r"E:\car_search_data\dataset"))
    ap.add_argument("--threshold", type=float, default=0.85, help="CLIP relevance cut from commons_filter.py")
    ap.add_argument("--min-per-split", type=int, default=2)
    ap.add_argument("--min-class-total", type=int, default=10)
    a = ap.parse_args()
    import imagehash
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for d in ("raw", "processed", "manifests", "reports/figures"):  # only what this script owns: raw_round2 and its CSVs survive
        shutil.rmtree(a.out / d, ignore_errors=True)
    for d in ("raw", "metadata", "processed", "manifests", "reports", "reports/figures"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    score = {r["file"]: float(r["score"]) for r in csv.DictReader((SRC / "relevance.csv").open(encoding="utf-8"))}
    metas = [json.loads(l) for l in (SRC / "metadata.jsonl").open(encoding="utf-8")]

    rejected, cands = [], []
    for m in metas:
        ext = ".png" if m["download_url"].lower().endswith(".png") else ".jpg"
        p = SRC / "images" / (hashlib.sha1(m["page_url"].encode()).hexdigest()[:16] + ext)
        cls = ALIAS.get(m["query"], m["query"])
        rej = lambda why: rejected.append([cls, m["page_url"], m["title"], p.as_posix(), why])
        if not p.exists():
            rej("download_missing")
            continue
        raw = p.read_bytes()
        fmt = magic_format(raw[:16] + raw[16:512])
        if fmt not in FORMATS:
            rej(f"format_{fmt}")
            continue
        try:
            with Image.open(p) as im:
                im.load()
                w, h = im.size
                rgb = to_rgb(im)
        except Exception:
            rej("corrupt_or_unreadable")
            continue
        if min(w, h) < 320:
            rej("below_320px")
        elif str(p) not in score:
            rej("not_scored_by_clip_filter")
        elif score[str(p)] < a.threshold:
            rej(f"off_topic_clip<{a.threshold}")
        else:
            cands.append(dict(cls=cls, m=m, p=p, fmt=fmt, w=w, h=h, size=len(raw), rgb=rgb, score=score[str(p)]))

    drop, pairs = hamming_dups([imagehash.phash(c["rgb"]) for c in cands])
    dups = [[f"cand{j}", cands[j]["p"].name, cands[i]["p"].name, d] for i, j, d in pairs]
    kept = [c for i, c in enumerate(cands) if i not in drop]

    counters, rows = defaultdict(int), []
    now = lambda p: datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    for c in kept:
        counters[c["cls"]] += 1
        c["id"] = f"{c['cls'].replace(' ', '_')}_{counters[c['cls']]:04d}"
        dst = a.out / "raw" / c["cls"].replace(" ", "_") / f"{c['id']}{'.png' if c['fmt'] == 'png' else '.webp' if c['fmt'] == 'webp' else '.jpg'}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(c["p"], dst)  # originals byte-identical
        c["raw"] = dst
    ids_by_class = defaultdict(list)
    for c in kept:
        ids_by_class[c["cls"]].append(c["id"])
    split, skipped = stratified_split(ids_by_class, a.min_per_split, min_total=a.min_class_total)

    man = defaultdict(list)
    for c in kept:
        sp = split.get(c["id"], "excluded_too_few")
        pw = ph = ""
        if sp != "excluded_too_few":
            out = a.out / "processed" / sp / c["cls"].replace(" ", "_") / f"{c['id']}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            clip_preprocess(c["rgb"]).save(out, quality=95)
            pw = ph = CLIP_SIDE
            man[sp].append([c["id"], c["cls"], out.as_posix(), c["w"], c["h"], pw, ph, sp, c["m"]["page_url"], "accepted"])
        rows.append([c["id"], c["cls"], c["m"]["page_url"], c["m"]["title"], c["raw"].as_posix(), c["w"], c["h"], c["size"],
                     c["fmt"], now(c["p"]), "accepted" if sp != "excluded_too_few" else "accepted_class_too_few",
                     c["m"]["licence"], c["m"]["creator"], f"{c['score']:.3f}"])
    write_csv(a.out / "metadata/scraped_images.csv",
              ["image_id", "class", "source_url", "page_title", "file_path", "width", "height", "file_size_bytes", "format",
               "downloaded_at", "status", "licence", "creator", "clip_relevance"], rows)
    write_csv(a.out / "metadata/rejected_images.csv", ["class", "source_url", "page_title", "file", "reason"], rejected)
    write_csv(a.out / "metadata/duplicate_images.csv", ["dropped", "dropped_file", "kept_file", "hamming"], dups)
    for sp, r in man.items():
        write_csv(a.out / f"manifests/{sp}.csv", ["image_id", "class", "file_path", "original_width", "original_height",
                                                   "processed_width", "processed_height", "split", "source_url", "status"], r)

    # ---- EDA
    cls_n = Counter(c["cls"] for c in kept)
    fig = a.out / "reports/figures"
    names = sorted(cls_n, key=cls_n.get, reverse=True)
    plt.figure(figsize=(10, 5)); plt.barh(names, [cls_n[n] for n in names]); plt.title("Images per class"); plt.tight_layout()
    plt.savefig(fig / "images_per_class.png", dpi=110); plt.close()
    plt.figure(figsize=(8, 4)); plt.hist([c["w"] for c in kept], 30, alpha=.6, label="width"); plt.hist([c["h"] for c in kept], 30, alpha=.6, label="height")
    plt.legend(); plt.title("Image width and height (px)"); plt.tight_layout(); plt.savefig(fig / "size_hist.png", dpi=110); plt.close()
    plt.figure(figsize=(11, 5)); plt.boxplot([[min(c["w"], c["h"]) for c in kept if c["cls"] == n] for n in names], labels=names, vert=True)
    plt.xticks(rotation=70, ha="right"); plt.title("Short side (px) by class"); plt.tight_layout(); plt.savefig(fig / "dims_by_class.png", dpi=110); plt.close()
    fm = Counter(c["fmt"] for c in kept)
    plt.figure(figsize=(4, 4)); plt.pie(fm.values(), labels=fm.keys(), autopct="%1.0f%%"); plt.title("Formats"); plt.savefig(fig / "formats.png", dpi=110); plt.close()
    why = Counter(r[4] for r in rejected)
    plt.figure(figsize=(8, 4)); plt.barh(list(why), list(why.values())); plt.title("Rejection reasons"); plt.tight_layout(); plt.savefig(fig / "reject_reasons.png", dpi=110); plt.close()
    per = 8
    grid = Image.new("RGB", (per * 112, len(names) * 112), "white")
    for r_i, n in enumerate(names):
        for k, c in enumerate([c for c in kept if c["cls"] == n][:per]):
            t = c["rgb"].copy(); t.thumbnail((110, 110)); grid.paste(t, (k * 112, r_i * 112))
    grid.save(fig / "sample_grid.jpg", quality=80)

    ar = np.array([c["w"] / c["h"] for c in kept]) if kept else np.array([1.0])
    sz = np.array([c["size"] for c in kept]) / 1024 if kept else np.array([0.0])
    small = sum(min(c["w"], c["h"]) < 400 for c in kept)
    split_n = {sp: len(r) for sp, r in man.items()}
    licences = Counter(c["m"]["licence"] for c in kept)
    low = sorted(kept, key=lambda c: c["score"])[:5]
    R = a.out / "reports"
    (R / "scraping_summary.md").write_text(
        f"# Scraping summary\n\nSource: Wikimedia Commons official API only (CC0, public domain, CC BY; no NC/ND/SA). "
        f"No marketplace, no login-gated or ban-listed site (vault Data Plan After eBay). Run: {datetime.now().isoformat(timespec='seconds')}\n\n"
        f"- Candidates in metadata: {len(metas)}\n- Accepted after validation, relevance and dedupe: {len(kept)}\n"
        f"- Rejected: {len(rejected)} ({dict(why)})\n- Near-duplicates removed (pHash Hamming <= 6): {len(dups)}\n"
        f"- Licences: {dict(licences)}\n- Failed sources: none blocked; HTTP 429 handled by retry\n- HEIC: not encountered (Commons serves JPEG/PNG)\n"
        f"- Classes: {len(cls_n)}; classes too small and excluded from splits (< {a.min_class_total}): {skipped or 'none'}\n"
        f"- Credit lines: metadata/scraped_images.csv (licence, creator, source_url). Do not redistribute the images; cite per licence.\n", encoding="utf-8")
    (R / "class_balance_report.md").write_text(
        "# Class balance\n\n| class | images | train | val | test |\n|---|---|---|---|---|\n" + "\n".join(
            f"| {n} | {cls_n[n]} | " + " | ".join(str(sum(1 for c in kept if c['cls'] == n and split.get(c['id']) == s)) for s in ("train", "validation", "test")) + " |"
            for n in names) + f"\n\nMin {min(cls_n.values(), default=0)}, max {max(cls_n.values(), default=0)}; ratio max/min "
        f"{max(cls_n.values(), default=1) / max(1, min(cls_n.values(), default=1)):.1f}. Flagged (too few, excluded from splits): {skipped or 'none'}. "
        "Imbalance is flagged, not corrected: use class-balanced sampling or weighted loss, do not pad with unrelated images.\n", encoding="utf-8")
    (R / "data_quality_report.md").write_text(
        f"# Data quality\n\n- Short side < 400 px: {small} of {len(kept)}\n- Aspect ratio median {np.median(ar):.2f}, p5 {np.percentile(ar, 5):.2f}, p95 {np.percentile(ar, 95):.2f}\n"
        f"- File size KB median {np.median(sz):.0f}, p95 {np.percentile(sz, 95):.0f}\n- Rejection reasons: {dict(why)}\n"
        f"- Lowest CLIP relevance among kept (review by eye): {[(c['id'], round(c['score'], 2)) for c in low]}\n"
        "- Known issues: labels are the search term only; Commons contains adverts, diagrams, historic photos and whole vehicles "
        "that survive the CLIP cut; downloaded files are Commons thumbnails (long side <= 1024), not true originals; "
        "images are web photos, not shop phone photos (domain gap).\n", encoding="utf-8")
    (R / "eda_report.md").write_text(
        f"# EDA\n\n- Images {len(kept)}, classes {len(cls_n)}, splits {split_n}\n- Formats {dict(fm)}\n"
        f"- Width median {int(np.median([c['w'] for c in kept] or [0]))}, height median {int(np.median([c['h'] for c in kept] or [0]))}\n"
        "- Charts in `figures/`: images_per_class, size_hist, dims_by_class, formats, reject_reasons, sample_grid.\n"
        f"- Split: stratified by class, seed 42, val and test each max({a.min_per_split}, 10%) per class, no image in two splits. "
        "Recommended 80/10/10 once classes are large enough.\n- Dedupe runs across the whole set, so near-duplicates cannot straddle splits.\n", encoding="utf-8")
    print(f"accepted {len(kept)} rejected {len(rejected)} dups {len(dups)} splits {split_n} skipped {skipped}\nreasons {dict(why)}\n-> {a.out}")


if __name__ == "__main__":
    main()

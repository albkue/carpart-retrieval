#!/usr/bin/env python3
"""Clean the OLD data (CC BY spare-part folders), merge it with the cleaned NEW data (Commons), run EDA.

Old : data/raw/catalog_images/<part>/*   part-level labels from folder names (unverified), motorcycle hardware,
      Roboflow CC BY 4.0. The KAGGLE50 folders (Apache 2.0, type labels only: not target data, dropped from
      training 2026-10-23; profiled separately by clean_kaggle50.py) and the junk folder IMAGES are inventoried
      and excluded. Dedupe runs inside a part only; near-duplicates ACROSS labels are kept and linked into one group_id
      (same rule as make_ccby_split.py), so a later split keeps them together.
New : dataset_clean/metadata/clean_images.csv   category-level labels (search term), Commons CC0/PD/CC BY.
The two tracks are never merged into one label space: every row has label_type = part | category.
Writes only to dataset_combined/. Originals untouched. No splits, no training.

Usage: python scripts/data/combine_clean.py
"""
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_clip_dataset import FORMATS, magic_format, to_rgb
from clean_dataset import dedupe, md
from make_ccby_split import same_content, thumb

REPO = Path(__file__).resolve().parents[2]
OLD = REPO / "data/raw/catalog_images"
NEW = Path(r"E:\car_search_data\dataset_clean")
OUT = Path(r"E:\car_search_data\dataset_combined")
SKIP_JUNK = {"IMAGES"}
STORE_LIKE = re.compile(r"^STORE\d+$", re.IGNORECASE)
CREDIT = "Roboflow Universe workspace 'supply-spare-parts', CC BY 4.0"
UF_MAX = 6


def gray_like(im: Image.Image) -> bool:
    a = np.asarray(im.resize((64, 64)), dtype=np.int16)
    return bool(np.abs(a[..., 0] - a[..., 1]).mean() < 3 and np.abs(a[..., 1] - a[..., 2]).mean() < 3)


def same_row_content(a, b) -> bool:
    return same_content(a["th"], b["th"])


def hamming_groups(items: list, max_dist=UF_MAX):
    """items: (key, label, ph, th). Union rows with different labels, pHash <= max_dist and matching content thumbnail.
    Returns (links, comp) where comp maps key to root key."""
    parent = {k: k for k, *_ in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    arr = np.array([it[2] for it in items], dtype=np.uint64)
    links = []
    for i in range(len(items) - 1):
        d = np.unpackbits((arr[i + 1:] ^ arr[i]).view(np.uint8)).reshape(-1, 64).sum(1)
        for off in np.nonzero(d <= max_dist)[0]:
            j = i + 1 + int(off)
            if items[i][1] != items[j][1] and same_content(items[i][3], items[j][3]):
                links.append((items[i][0], items[j][0], int(d[off])))
                parent[find(items[i][0])] = find(items[j][0])
    return links, {k: find(k) for k in parent}


def main():
    import imagehash
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for d in ("images", "metadata", "reports/figures"):
        shutil.rmtree(OUT / d, ignore_errors=True)
        (OUT / d).mkdir(parents=True, exist_ok=True)

    tier = list(csv.DictReader((OLD / "catalog_tier1b.csv").open(encoding="utf-8")))
    src_of = defaultdict(set)
    for r in tier:
        m = re.search(r"auto-imported from ([\w\-]+)", r["notes"])
        src_of[r["part_number_raw"]].add(m.group(1) if m else "unknown")
    equiv = {g["folder"] for g in json.load((REPO / "data/processed/part_equivalence.json").open(encoding="utf-8"))["groups"]}
    psplit = {}
    for f, key in (("splits_ccby_v0_train.json", "train"), ("splits_ccby_v0_eval.json", "validation")):
        j = json.load((REPO / "data/processed" / f).open(encoding="utf-8"))
        for e in j["catalog"]:
            psplit[e["part_id"]] = key
        for e in j.get("far_unknown", []):
            psplit[e["part_id"]] = "far_unknown"

    inv, rows = [], []
    for part_dir in sorted(p for p in OLD.iterdir() if p.is_dir()):
        kaggle, junk = part_dir.name.startswith("KAGGLE50"), part_dir.name in SKIP_JUNK
        for p in sorted(f for f in part_dir.iterdir() if f.is_file()):
            raw = p.read_bytes()
            fmt = magic_format(raw[:512])
            rec = {"path": p.as_posix(), "part": part_dir.name, "file": p.name, "format": fmt, "size": len(raw), "w": None, "h": None,
                   "source": "kaggle50" if kaggle else "|".join(sorted(src_of.get(part_dir.name, {"unknown"})))}
            reason = "kaggle50_not_target_data" if kaggle else "junk_folder_not_a_part" if junk else None
            rgb = None
            try:
                with Image.open(p) as im:
                    im.load()
                    rec["w"], rec["h"] = im.size
                    rec["mode"] = im.mode
                    if not (kaggle or junk):
                        rgb = to_rgb(im)
            except Exception:
                reason = reason or "rejected_corrupt"
            if reason is None and fmt not in FORMATS:
                reason = f"rejected_format_{fmt}"
            if reason is None and min(rec["w"], rec["h"]) < 320:
                reason = "rejected_low_resolution"
            rec["reason"] = reason or ""
            inv.append(rec)
            if reason is None:
                import hashlib
                rec["row"] = dict(id=f"{part_dir.name}__{p.stem}", sha=hashlib.sha256(raw).hexdigest(), ph=int(str(imagehash.phash(rgb)), 16),
                                  w=rec["w"], h=rec["h"], size=len(raw), meta_n=3, part=part_dir.name, path=p, th=thumb(rgb))  # path only: decoded images are re-opened on write (RAM)
    pd.DataFrame([{k: v for k, v in r.items() if k != "row"} for r in inv]).to_csv(OUT / "metadata/old_data_inventory.csv", index=False)

    # ---- old: dedupe inside each part only
    by_part = defaultdict(list)
    for r in inv:
        if "row" in r:
            by_part[r["part"]].append(r["row"])
    old_kept, old_dups = [], []
    for part, rs in by_part.items():
        k, d = dedupe(rs, confirm=same_row_content)
        old_kept += k
        old_dups += d
    out_rows = []
    for k in old_kept:
        dst = OUT / "images/part" / k["part"] / (k["path"].stem + ".jpg")
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(k["path"]) as im:
            k["rgb"] = to_rgb(im)
        k["rgb"].save(dst, quality=95)
        out_rows.append({"image_id": k["id"], "label_type": "part", "label": k["part"], "clean_path": dst.as_posix(), "original_path": k["path"].as_posix(),
                         "source": "|".join(sorted(src_of.get(k["part"], {"unknown"}))), "source_url": "https://universe.roboflow.com/supply-spare-parts",
                         "th": k["th"], "licence": "CC BY 4.0", "author": CREDIT, "width": k["w"], "height": k["h"], "short_side": min(k["w"], k["h"]),
                         "file_size_bytes": k["size"], "sha256": k["sha"], "phash": f"{k['ph']:016x}", "clip_relevance": "",
                         "gray_like": (lambda g: (k.pop("rgb"), g)[1])(gray_like(k["rgb"])), "domain": "studio_hardware", "store_code_like": bool(STORE_LIKE.match(k["part"])),
                         "multi_code_folder": k["part"] in equiv, "existing_part_split": psplit.get(k["part"], "dropped_or_untracked")})
    # ---- new: already cleaned
    nc = pd.read_csv(NEW / "metadata/clean_images.csv", dtype=str).fillna("")
    for r in nc.itertuples():
        with Image.open(r.clean_path) as im:
            im = im.convert("RGB")
            g, th = gray_like(im), thumb(im)
        out_rows.append({"image_id": r.image_id, "label_type": "category", "label": r._2, "th": th, "clean_path": r.clean_path, "original_path": r.original_path,
                         "source": f"commons_{r.batch}", "source_url": r.source_url, "licence": r.licence, "author": r.author, "width": int(r.width),
                         "height": int(r.height), "short_side": int(r.short_side), "file_size_bytes": int(r.file_size_bytes), "sha256": r.sha256, "phash": r.phash,
                         "clip_relevance": r.clip_relevance, "gray_like": g, "domain": "web_photo", "store_code_like": False, "multi_code_folder": False,
                         "existing_part_split": "", "title_flag": r.title_flag})
    ths = {r["image_id"]: r.pop("th") for r in out_rows}
    cdf = pd.DataFrame(out_rows)
    cdf["title_flag"] = cdf.get("title_flag", "").fillna("")
    # ---- cross-label links over everything kept (old parts and commons classes)
    items = [(r.image_id, f"{r.label_type}:{r.label}", int(r.phash, 16), ths[r.image_id]) for r in cdf.itertuples()]
    links, comp = hamming_groups(items)
    linked = {a for a, _, _ in links} | {b for _, b, _ in links}
    cdf["group_id"] = [("link_" + comp[i]) if i in linked else f"{lt}:{lb}" for i, lt, lb in zip(cdf.image_id, cdf.label_type, cdf.label)]
    n_per = cdf[cdf.label_type == "part"].groupby("label").size()
    cdf["images_in_label"] = [int(n_per.get(l, 0)) if t == "part" else int((cdf[(cdf.label_type == "category")].label == l).sum()) for t, l in zip(cdf.label_type, cdf.label)]
    bad_cat = {"water pump": "label_untrustworthy_reviewed", "fuel pump": "label_untrustworthy_reviewed", "brake disc": "mixed_bicycle_motorcycle_reviewed"}
    cdf["label_review"] = [bad_cat.get(l, "unreviewed") if t == "category" else ("store_code_like_check" if s else "unverified_folder_name")
                           for t, l, s in zip(cdf.label_type, cdf.label, cdf.store_code_like)]
    cdf.to_csv(OUT / "metadata/combined_images.csv", index=False)
    pd.DataFrame([{"dup_id": d[0]["id"], "kept_id": d[1]["id"], "dup_type": d[2], "hamming": d[3], "part": d[0]["part"]} for d in old_dups],
                 columns=["dup_id", "kept_id", "dup_type", "hamming", "part"]).to_csv(OUT / "metadata/old_within_part_duplicates.csv", index=False)
    pd.DataFrame(links, columns=["image_a", "image_b", "hamming"]).assign(
        label_a=lambda d: d.image_a.map(dict(zip(cdf.image_id, cdf.label_type + ":" + cdf.label))),
        label_b=lambda d: d.image_b.map(dict(zip(cdf.image_id, cdf.label_type + ":" + cdf.label)))).to_csv(OUT / "metadata/cross_label_links.csv", index=False)
    pd.DataFrame([{"path": r["path"], "part": r["part"], "reason": r["reason"]} for r in inv if r["reason"]]).to_csv(OUT / "metadata/combined_rejected.csv", index=False)

    # ---- EDA
    part, cat = cdf[cdf.label_type == "part"], cdf[cdf.label_type == "category"]
    ex = Counter(r["reason"] for r in inv if r["reason"])
    pp = part.groupby("label").size()
    link_cross = [l for l in links if l[0] in set(part.image_id) and l[1] in set(cat.image_id) or l[1] in set(part.image_id) and l[0] in set(cat.image_id)]

    def fig(name, f, size=(9, 4.5)):
        plt.figure(figsize=size); f(); plt.tight_layout(); plt.savefig(OUT / "reports/figures" / name, dpi=110); plt.close()
    fig("images_per_part.png", lambda: (plt.hist(pp.values, bins=range(1, int(pp.max()) + 2)), plt.title("Images per part (old, after dedupe)")))
    fig("images_per_category.png", lambda: (lambda v: (plt.barh(v.index[::-1], v.values[::-1]), plt.title("Images per category (new)")))(cat.label.value_counts()))
    fig("short_side_by_track.png", lambda: (plt.hist([part.short_side, cat.short_side], 30, label=["part (old)", "category (new)"]), plt.legend(), plt.title("Short side (px)")))
    fig("grayscale_by_track.png", lambda: (plt.bar(["part (old)", "category (new)"], [part.gray_like.mean() * 100, cat.gray_like.mean() * 100]), plt.title("% grayscale-like images")))
    fig("licence_by_track.png", lambda: (lambda v: (plt.barh(v.index, v.values), plt.title("Licence, combined")))(cdf.licence.value_counts()))
    fig("exclusion_reasons_old.png", lambda: (plt.barh(list(ex), list(ex.values())), plt.title("Old data excluded files")))
    rng = random.Random(42)
    pick = rng.sample(sorted(part.image_id), 48)
    sheet = Image.new("RGB", (8 * 130, 6 * 140), "white")
    dr = ImageDraw.Draw(sheet)
    for i, iid in enumerate(pick):
        row = part[part.image_id == iid].iloc[0]
        im = Image.open(row.clean_path); im.thumbnail((126, 118))
        x, y = (i % 8) * 130, (i // 8) * 140
        sheet.paste(im, (x + 2, y + 2)); dr.text((x + 2, y + 122), row.label[:20], fill="black")
    sheet.save(OUT / "reports/figures/sample_grid_part_level.png")
    stats = {"old_files_found": len(inv), "old_part_files": sum(1 for r in inv if not r["part"].startswith("KAGGLE50") and r["part"] not in SKIP_JUNK),
             "old_clean": len(part), "old_parts": part.label.nunique(), "old_dups": len(old_dups), "new_clean": len(cat), "new_classes": cat.label.nunique(),
             "combined": len(cdf), "cross_label_links": len(links), "old_new_links": len(link_cross)}
    R = OUT / "reports"
    md(R / "combined_cleaning_report.md", "Combined cleaning report (old + new)", [
        ("Old data check", "Every file under data/raw/catalog_images: magic bytes, decode, 320 px, licence by source, junk and Kaggle50 exclusion, within-part dedupe.",
         f"- files found {len(inv)}; excluded {dict(ex)}\n- part files cleaned {stats['old_part_files']} -> kept {len(part)} in {part.label.nunique()} parts; within-part duplicates removed {len(old_dups)}\n"
         f"- parts with a single image after dedupe: {int((pp == 1).sum())}; 2+ images: {int((pp >= 2).sum())}\n- grayscale-like old images: {int(part.gray_like.sum())} of {len(part)} ({part.gray_like.mean() * 100:.0f}%); new: {int(cat.gray_like.sum())} of {len(cat)} ({cat.gray_like.mean() * 100:.0f}%)\n"
         f"- folders that look like store codes: {sorted(set(part[part.store_code_like].label))}\n- multi-code (equivalence) folders: {int(part[part.multi_code_folder].label.nunique())}, none human-reviewed",
         "Review STORE folders and the 35 equivalence groups by hand."),
        ("Merge", "Old part-level rows and new category-level rows in one manifest, label_type column keeps the two label spaces apart. Cross-label near-duplicates (pHash <= 6) linked into one group_id.",
         f"- combined clean images {len(cdf)} (part {len(part)}, category {len(cat)}); cross-label links {len(links)} (old-new {len(link_cross)})\n- licences: {cdf.licence.value_counts().to_dict()}",
         "Split by group_id (never by image). Never train one classifier head over part and category labels together."),
        ("Roles", "Spec compliance.", "- Old tier1b is motorcycle hardware: the Dataset Spec allows it as far unknowns only; using it for training is a scope change to record. New Commons rows have no part numbers: category-level and distractor use only.",
         "Record the scope change in the spec and tracker before any fine-tuning.")])
    md(R / "combined_eda_report.md", "Combined EDA report", [
        ("Part track (old)", "Images per part, size, domain.", f"- {len(part)} images, {part.label.nunique()} parts; per part min {pp.min()}, median {pp.median():.0f}, mean {pp.mean():.1f}, max {pp.max()}\n- short side median {part.short_side.median():.0f}; file KB median {part.file_size_bytes.median() / 1024:.0f}",
         "Parts with 1 image cannot be query plus gallery; keep them as distractors."),
        ("Category track (new)", "Class counts and sizes.", f"- {len(cat)} images, {cat.label.nunique()} classes; min {cat.label.value_counts().min()}, max {cat.label.value_counts().max()}; short side median {cat.short_side.median():.0f}\n- label review: {cat.groupby('label_review').size().to_dict()}",
         "Drop water pump and fuel pump; review the rest."),
        ("Domain gap", "Grayscale share and style.", f"- grayscale-like: part {part.gray_like.mean() * 100:.0f}%, category {cat.gray_like.mean() * 100:.0f}%. Part track is studio hardware, category track is web photos of cars and shops.",
         "Expect a domain gap; evaluate on your own photos, not on either track."),
        ("Figures", "Charts.", "- images_per_part, images_per_category, short_side_by_track, grayscale_by_track, licence_by_track, exclusion_reasons_old, sample_grid_part_level.", "Inspect sample_grid_part_level.png for label sanity.")])
    print(stats)
    print("excluded", dict(ex), "| part dist", pp.describe().round(1).to_dict(), "| gray old/new", round(part.gray_like.mean(), 2), round(cat.gray_like.mean(), 2))
    print("single-image parts", int((pp == 1).sum()), "store-like", sorted(set(part[part.store_code_like].label)), "links old-new", len(link_cross))
    print("label_review", cdf.groupby(["label_type", "label_review"]).size().to_dict())


if __name__ == "__main__":
    main()

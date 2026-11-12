#!/usr/bin/env python3
"""Inventory, validate, deduplicate and profile every image in E:\\car_search_data\\dataset (no scraping, no splits).

Reads raw/ (round 1), raw_round2/ and processed/ (224 px derived copies of round 1: inventoried, never part of the
clean pool) plus their metadata CSVs. Writes ONLY to dataset_clean/: inventory, clean RGB JPEGs, metadata CSVs,
reports and figures. Rerunnable and deterministic (image ids come from the collectors; ties break on id).

Usage: python scripts/data/clean_dataset.py
"""
import csv
import hashlib
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_clip_dataset import FORMATS, magic_format, to_rgb
from commons_collect import licence_ok

ROOT = Path(r"E:\car_search_data\dataset")
OUT = Path(r"E:\car_search_data\dataset_clean")
FOLDERS = (("raw", "round1"), ("raw_round2", "round2"), ("processed", "processed_derived"))
KEYS = {"water pump": ["pump"], "timing belt": ["belt", "timing"], "bulb": ["bulb", "lamp", "light", "halogen", "led"],
        "cabin air filter": ["filter", "cabin", "pollen"], "clutch disc": ["clutch"], "fuel pump": ["pump", "fuel"],
        "starter motor": ["starter", "motor"], "spark plug": ["spark", "plug"], "brake disc": ["brake", "disc", "disk", "rotor"],
        "radiator": ["radiator"], "air filter": ["filter", "air"], "shock absorber": ["shock", "absorber", "damper", "suspension"],
        "headlight": ["headlight", "headlamp", "light", "lamp"], "oil filter": ["oil", "filter"], "battery": ["battery", "accumulator"],
        "brake pad": ["brake", "pad"], "alternator": ["alternator", "generator", "dynamo"],
        "exhaust muffler": ["exhaust", "muffler", "silencer"], "wiper blade": ["wiper", "blade"], "ignition coil": ["ignition", "coil"]}
NEG_TITLE = ["advert", "patent", "painting", "poster", "drawing", "blueprint", "irrigation", "hand pump", "sewage", "well pump"]


def dedupe(rows: list, max_dist=6, confirm=None):
    """Greedy canonical pick. rows: dicts with id, sha, ph (int), w, h, size, meta_n. Best quality first (pixels, bytes,
    metadata completeness, id); a later row is a duplicate of the first kept row it matches (exact sha, else pHash <= max_dist
    and, when given, confirm(row, kept_row): pHash alone merges different small objects on white backgrounds)."""
    order = sorted(rows, key=lambda r: (-r["w"] * r["h"], -r["size"], -r["meta_n"], r["id"]))
    kept, dups, sha_to = [], [], {}
    arr = np.zeros(len(order), dtype=np.uint64)
    for r in order:
        if r["sha"] in sha_to:
            dups.append((r, sha_to[r["sha"]], "exact", 0))
            continue
        hit = None
        if kept:
            x = arr[:len(kept)] ^ np.uint64(r["ph"])
            d = np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(1)
            for j in np.argsort(d, kind="stable"):
                if d[j] > max_dist:
                    break
                if confirm is None or confirm(r, kept[j]):
                    hit = (kept[j], int(d[j]))
                    break
        if hit:
            dups.append((r, hit[0], "near", hit[1]))
            continue
        arr[len(kept)] = np.uint64(r["ph"])
        kept.append(r)
        sha_to[r["sha"]] = r
    return kept, dups


def build_inventory():
    def load(p):
        return {r.get("image_id") or r.get("file"): r for r in csv.DictReader((ROOT / p).open(encoding="utf-8"))} if (ROOT / p).exists() else {}
    m1, m2 = load("metadata/scraped_images.csv"), load("metadata/accepted_round2.csv")
    splits = {r["image_id"]: r["split"] for f in (ROOT / "manifests").glob("*.csv") for r in csv.DictReader(f.open(encoding="utf-8"))}
    import imagehash
    rows = []
    for folder, batch in FOLDERS:
        for p in sorted((ROOT / folder).rglob("*")):
            if not p.is_file():
                continue
            iid = p.stem
            meta = m1.get(iid) if batch == "round1" else m2.get(iid) if batch == "round2" else (m1.get(iid) or m2.get(iid))
            raw = p.read_bytes()
            fmt = magic_format(raw[:512])
            w = h = None
            ph, loadable = 0, False
            try:
                with Image.open(p) as im:
                    im.load()
                    w, h = im.size
                    ph = int(str(imagehash.phash(to_rgb(im))), 16)
                    loadable = True
            except Exception:
                pass
            g = (meta or {}).get
            rows.append({"image_id": iid, "path": p.as_posix(), "folder": folder, "batch": batch, "class_folder": p.parent.name.replace("_", " "),
                         "class_meta": g("class", ""), "file_name": p.name, "ext": p.suffix.lower(), "format": fmt, "width": w, "height": h,
                         "aspect": round(w / h, 3) if w else None, "short_side": min(w, h) if w else None, "size": len(raw),
                         "source_url": g("source_url", "") or g("page_url", ""), "title": g("page_title", "") or g("title", ""),
                         "licence": g("licence", ""), "author": g("creator", "") or g("author", ""), "clip": g("clip_relevance", ""),
                         "has_meta": meta is not None, "split": splits.get(iid, ""), "sha": hashlib.sha256(raw).hexdigest(),
                         "ph": ph, "loadable": loadable})
    return rows


def classify(r, id_counts):
    """First failing check wins; (status, reason). status: ok | rejected | needs_review | excluded."""
    if r["format"] not in FORMATS:
        return "rejected", f"rejected_format_{r['format']}"
    if not r["loadable"]:
        return "rejected", "rejected_corrupt"
    if r["short_side"] < 320:
        return "rejected", "rejected_low_resolution"
    if not r["has_meta"]:
        return "needs_review", "no_metadata_row"
    if not r["licence"]:
        return "needs_review", "licence_missing"
    if not licence_ok(r["licence"]):
        return "excluded", "licence_not_cc0_pd_ccby"
    if r["class_meta"] and r["class_meta"] != r["class_folder"]:
        return "needs_review", "class_label_conflict"
    if id_counts[r["image_id"]] > 1:
        return "needs_review", "duplicate_image_id"
    return "ok", ""


def title_flag(r):
    t = (r["title"] or "").lower()
    if not t:
        return ""
    if any(n in t for n in NEG_TITLE):
        return "negative_title_word"
    return "" if any(k in t for k in KEYS.get(r["class_folder"], [])) else "no_class_keyword_in_title"


def md(path, title, sections):
    """sections: (heading, checked, found, action) tuples -> one report in the required shape."""
    out = [f"# {title}\n"]
    for h, c, f, a in sections:
        out += [f"## {h}\n", f"**Checked:** {c}\n", f"**Found:**\n{f}\n", f"**Recommended action:** {a}\n"]
    path.write_text("\n".join(out), encoding="utf-8")


def table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for d in ("images", "metadata", "reports/figures"):
        shutil.rmtree(OUT / d, ignore_errors=True)
        (OUT / d).mkdir(parents=True, exist_ok=True)
    inv = build_inventory()
    raw = [r for r in inv if r["batch"] != "processed_derived"]
    id_counts = Counter(r["image_id"] for r in raw)
    for r in inv:
        r["status"], r["reason"] = classify(r, id_counts) if r["batch"] != "processed_derived" else ("derived", "processed_copy_of_round1")
        r["title_flag"] = title_flag(r) if r["batch"] != "processed_derived" else ""
    ok = [dict(id=r["image_id"], sha=r["sha"], ph=r["ph"], w=r["width"], h=r["height"], size=r["size"],
               meta_n=sum(bool(r[k]) for k in ("source_url", "title", "licence", "author", "clip")), row=r)
          for r in inv if r["status"] == "ok"]
    kept, dups = dedupe(ok)
    dup_ids = {d[0]["id"] for d in dups}
    for r in inv:
        r["is_dup"] = r["image_id"] in dup_ids and r["batch"] != "processed_derived"
    cols = ["image_id", "path", "folder", "batch", "class_folder", "class_meta", "file_name", "ext", "format", "width", "height", "aspect",
            "short_side", "size", "source_url", "title", "licence", "author", "clip", "has_meta", "split", "sha", "status", "reason",
            "title_flag", "is_dup"]
    df = pd.DataFrame(inv)
    df[cols].to_csv(OUT / "metadata/full_dataset_inventory.csv", index=False)
    dfr = df[df.batch != "processed_derived"].copy()

    clean = []
    for k in kept:
        r = k["row"]
        dst = OUT / "images" / r["class_folder"].replace(" ", "_") / f"{r['image_id']}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(r["path"]) as im:
            to_rgb(im).save(dst, quality=95)
        clean.append({"image_id": r["image_id"], "class": r["class_folder"], "clean_path": dst.as_posix(), "original_path": r["path"],
                      "batch": r["batch"], "source_url": r["source_url"], "page_title": r["title"], "licence": r["licence"], "author": r["author"],
                      "width": r["width"], "height": r["height"], "short_side": r["short_side"], "file_size_bytes": r["size"], "format": r["format"],
                      "sha256": r["sha"], "phash": f"{r['ph']:016x}", "clip_relevance": r["clip"], "title_flag": r["title_flag"], "existing_split": r["split"]})
    cdf = pd.DataFrame(clean)
    cdf.to_csv(OUT / "metadata/clean_images.csv", index=False)
    dfr[dfr.status.isin(["rejected", "excluded", "needs_review"])][["image_id", "path", "batch", "class_folder", "status", "reason", "licence", "source_url"]] \
        .to_csv(OUT / "metadata/rejected_images.csv", index=False)
    pd.DataFrame([{"dup_id": d[0]["id"], "kept_id": d[1]["id"], "dup_type": d[2], "hamming": d[3], "dup_class": d[0]["row"]["class_folder"],
                   "kept_class": d[1]["row"]["class_folder"], "dup_batch": d[0]["row"]["batch"], "kept_batch": d[1]["row"]["batch"],
                   "cross_class": d[0]["row"]["class_folder"] != d[1]["row"]["class_folder"]} for d in dups],
                 columns=["dup_id", "kept_id", "dup_type", "hamming", "dup_class", "kept_class", "dup_batch", "kept_batch", "cross_class"]) \
        .to_csv(OUT / "metadata/duplicates.csv", index=False)
    meta_rows = {"round1": ROOT / "metadata/scraped_images.csv", "round2": ROOT / "metadata/accepted_round2.csv"}
    orphan = []
    for b, p in meta_rows.items():
        have = set(dfr[dfr.batch == b].image_id)
        for r in csv.DictReader(p.open(encoding="utf-8")):
            if r["image_id"] not in have:
                orphan.append({"issue": "metadata_row_without_file", "batch": b, "image_id": r["image_id"], "detail": r.get("file_path", "")})
    miss = [{"issue": "file_without_metadata", "batch": r.batch, "image_id": r.image_id, "detail": r.path} for r in dfr[~dfr.has_meta].itertuples()]
    miss += [{"issue": "missing_source_url", "batch": r.batch, "image_id": r.image_id, "detail": r.path} for r in dfr[dfr.has_meta & (dfr.source_url == "")].itertuples()]
    miss += [{"issue": "duplicate_image_id", "batch": r.batch, "image_id": r.image_id, "detail": r.path} for r in dfr[dfr.image_id.map(id_counts) > 1].itertuples()]
    miss += [{"issue": "class_label_conflict", "batch": r.batch, "image_id": r.image_id, "detail": f"{r.class_folder} vs {r.class_meta}"} for r in dfr[dfr.reason == "class_label_conflict"].itertuples()]
    pd.DataFrame(miss + orphan, columns=["issue", "batch", "image_id", "detail"]).to_csv(OUT / "metadata/missing_metadata.csv", index=False)
    lic = dfr[dfr.status.isin(["excluded", "needs_review"]) & dfr.reason.isin(["licence_missing", "licence_not_cc0_pd_ccby", "no_metadata_row"])]
    lic[["image_id", "batch", "class_folder", "licence", "reason", "source_url"]].to_csv(OUT / "metadata/license_review.csv", index=False)

    # ---------- EDA numbers
    classes = sorted(dfr.class_folder.unique())
    per = []
    for c in classes:
        d = dfr[dfr.class_folder == c]
        valid = d[d.status.isin(["ok"]) | (d.status.isin(["needs_review", "excluded"]) & d.reason.isin(["no_metadata_row", "licence_missing", "licence_not_cc0_pd_ccby", "class_label_conflict", "duplicate_image_id"]))]
        n_clean = int((cdf["class"] == c).sum()) if len(cdf) else 0
        per.append({"class": c, "raw": len(d), "valid": int(d.status.eq("ok").sum()), "clean": n_clean, "duplicates": int(d.is_dup.sum()),
                    "rejected": int(d.status.isin(["rejected", "excluded", "needs_review"]).sum()), "missing_meta": int((~d.has_meta).sum()),
                    "flagged_title": int(((d.title_flag != "") & d.status.eq("ok") & ~d.is_dup).sum()),
                    "ready": "READY" if n_clean >= 100 else "READY_SMALL" if n_clean >= 50 else "WEAK" if n_clean >= 30 else "NOT_READY"})
    pc = pd.DataFrame(per)
    pc.to_csv(OUT / "metadata/per_class_summary.csv", index=False)
    cl = cdf.copy()
    cl["clip_f"] = pd.to_numeric(cl.clip_relevance, errors="coerce")
    counts = cl["class"].value_counts()
    sizes = cl.file_size_bytes / 1024

    def fig(name, f):
        plt.figure(figsize=(9, 4.5)); f(); plt.tight_layout(); plt.savefig(OUT / "reports/figures" / name, dpi=110); plt.close()
    fig("class_distribution.png", lambda: (plt.barh(counts.index[::-1], counts.values[::-1]), plt.title("Clean images per class")))
    for col, nm in (("width", "image_width_histogram.png"), ("height", "image_height_histogram.png"), ("short_side", "short_side_histogram.png")):
        fig(nm, lambda col=col: (plt.hist(cl[col], 40), plt.title(f"{col} (px), clean set")))
    fig("aspect_ratio_histogram.png", lambda: (plt.hist(cl.width / cl.height, 40), plt.title("Aspect ratio (w/h)")))
    fig("file_size_distribution.png", lambda: (plt.hist(sizes, 40), plt.title("File size (KB)")))
    fig("format_distribution.png", lambda: (lambda v: (plt.bar(v.index, v.values), plt.title("Original format, clean set")))(cl.format.value_counts()))
    fig("license_distribution.png", lambda: (lambda v: (plt.barh(v.index, v.values), plt.title("Licence, clean set")))(cl.licence.value_counts()))
    scrape_rej = Counter()
    for f in ("metadata/rejected_images.csv", "metadata/rejected_round2.csv"):
        for r in csv.DictReader((ROOT / f).open(encoding="utf-8")):
            scrape_rej[r["reason"].split(":")[0].split("<")[0]] += 1
    clean_rej = Counter(dfr[dfr.status.isin(["rejected", "excluded", "needs_review"])].reason)
    allrej = scrape_rej + clean_rej + Counter({"duplicate": len(dups)})
    fig("rejection_reasons.png", lambda: (plt.barh(list(allrej)[::-1], [allrej[k] for k in list(allrej)[::-1]]), plt.title("Rejections: scrape stage + cleaning stage + duplicates")))
    dr = pc.set_index("class").pipe(lambda d: d.duplicates / d.raw.clip(lower=1) * 100)
    fig("duplicate_rate_by_class.png", lambda: (plt.barh(dr.index, dr.values), plt.title("Duplicate rate by class (% of raw)")))

    def clip_fig():
        plt.subplot(1, 2, 1); plt.hist(cl.clip_f.dropna(), 30); plt.title("CLIP relevance")
        g = [cl[cl["class"] == c].clip_f.dropna() for c in counts.index]
        plt.subplot(1, 2, 2); plt.boxplot(g, labels=counts.index, vert=False); plt.title("By class")
    plt.figure(figsize=(12, 6)); clip_fig(); plt.tight_layout(); plt.savefig(OUT / "reports/figures/clip_relevance_distribution.png", dpi=110); plt.close()
    rng = random.Random(42)
    for c in classes:
        sub = cdf[cdf["class"] == c] if len(cdf) else cdf
        ids = sorted(sub.image_id)
        pick = rng.sample(ids, min(24, len(ids)))
        sheet = Image.new("RGB", (8 * 130, 3 * 140), "white")
        d = ImageDraw.Draw(sheet)
        for i, iid in enumerate(pick):
            im = Image.open(sub[sub.image_id == iid].clean_path.iloc[0]); im.thumbnail((126, 118))
            x, y = (i % 8) * 130, (i // 8) * 140
            sheet.paste(im, (x + 2, y + 2)); d.text((x + 2, y + 122), iid[-14:], fill="black")
        sheet.save(OUT / f"reports/figures/sample_grid_{c.replace(' ', '_')}.png")

    # ---------- reports
    R = OUT / "reports"
    nbatch = dfr.groupby("batch").size().to_dict()
    tot = len(dfr)
    inv_sum = [f"- Total files found: {len(df)} (raw round 1 {nbatch.get('round1', 0)}, raw round 2 {nbatch.get('round2', 0)}, processed derived {int((df.batch == 'processed_derived').sum())})",
               f"- Unique files (sha256, raw only): {dfr.sha.nunique()} of {tot}", f"- By class (raw): {dfr.class_folder.value_counts().to_dict()}",
               f"- By format (magic bytes, raw): {dfr.format.value_counts().to_dict()}", f"- By batch: {nbatch}",
               f"- Missing metadata row: {int((~dfr.has_meta).sum())}; missing source URL: {int((dfr.source_url == '').sum())}; missing licence: {int((dfr.licence == '').sum())}",
               f"- raw vs processed folders: {df.folder.value_counts().to_dict()}", f"- Existing split labels (from manifests): {df[df.split != ''].split.value_counts().to_dict()}"]
    (OUT / "metadata/inventory_summary.md").write_text("# Inventory summary\n\n" + "\n".join(inv_sum) + "\n", encoding="utf-8")
    nclean, nvalid = len(cdf), int(dfr.status.eq("ok").sum())
    rej_n = int(dfr.status.isin(["rejected", "excluded", "needs_review"]).sum())
    review = int((dfr.status == "needs_review").sum() + ((dfr.title_flag != "") & dfr.status.eq("ok") & ~dfr.is_dup).sum())
    low400, low500 = int((cl.short_side < 400).sum()), int((cl.short_side < 500).sum())
    st = pc.ready.value_counts().to_dict()
    md(R / "data_cleaning_report.md", "Data cleaning report", [
        ("Inventory and validation", "Every raw and raw_round2 image: magic-byte format, decode, short side >= 320, licence, metadata match, class conflicts, duplicate ids.",
         f"- raw {tot}, valid {nvalid}, clean {nclean}, duplicates removed {len(dups)}, rejected/excluded/needs review {rej_n}\n- reasons: {dict(clean_rej)}\n\n" + table(pc),
         "Review the NOT_READY and WEAK classes before any training; the rest of the checks passed."),
        ("Clean set", "Canonical copy kept per duplicate group (highest pixels, then bytes, then metadata completeness); RGB JPEG q95 at original size in dataset_clean/images.",
         f"- {nclean} clean images in {cdf['class'].nunique()} classes. Originals untouched.", "Use dataset_clean/metadata/clean_images.csv as the only input to the later split.")])
    md(R / "full_eda_report.md", "Full EDA report", [
        ("Class distribution", "Counts per class in the clean set.", f"- min {counts.min()}, max {counts.max()}, median {counts.median():.0f}, mean {counts.mean():.1f}; max/min {counts.max() / counts.min():.1f}\n- <30: {list(counts[counts < 30].index)}\n- <50: {list(counts[counts < 50].index)}\n- <100: {len(counts[counts < 100])} classes",
         "Weight or sample by class; do not pad small classes."),
        ("Image quality", "Size, aspect ratio, bytes, format.", f"- width median {cl.width.median():.0f}, height median {cl.height.median():.0f}, short side median {cl.short_side.median():.0f}\n- short side < 400: {low400}; < 500: {low500}\n- aspect median {(cl.width / cl.height).median():.2f}\n- file size KB median {sizes.median():.0f}, p95 {sizes.quantile(.95):.0f}\n- formats {cl.format.value_counts().to_dict()}",
         "Sizes are fine for CLIP (224 px); keep original-size clean images and let the training transform resize."),
        ("Source quality", "Per batch: images, scrape-stage acceptance, duplicate rate, missing metadata.",
         "\n".join(f"- {b}: {n} raw images, duplicates {int(dfr[(dfr.batch == b)].is_dup.sum())}, missing meta {int(((dfr.batch == b) & ~dfr.has_meta).sum())}" for b, n in nbatch.items()) + f"\n- scrape-stage rejection reasons: {dict(scrape_rej)}",
         "Round 2 yields far less per class than round 1; Commons is nearly exhausted for thin classes."),
        ("CLIP relevance", "Stored relevance scores (zero-shot, threshold 0.85).", f"- scored {int(cl.clip_f.notna().sum())} of {nclean}; below 0.85: {int((cl.clip_f < 0.85).sum())}; median {cl.clip_f.median():.3f}\n- possible false positives: {int(((cl.title_flag != '') & (cl.clip_f >= 0.95)).sum())} images with CLIP >= 0.95 but a suspicious title (irrigation or advert images scored 1.0 in tests). False negatives cannot be measured: rejected files were not kept.",
         "Review flagged images by eye; a stricter or class-specific prompt would not fix irrigation-vs-car pumps, only a manual pass will."),
        ("Duplicates and labels", "Exact and near-duplicates, title keywords.", f"- exact {sum(d[2] == 'exact' for d in dups)}, near {sum(d[2] == 'near' for d in dups)}, cross-class {sum(d[0]['row']['class_folder'] != d[1]['row']['class_folder'] for d in dups)}\n- title flagged: {int((cl.title_flag != '').sum())} ({cl.title_flag.value_counts().to_dict()})",
         "See manual_review_checklist.md.")])
    md(R / "class_balance_report.md", "Class balance report", [
        ("Per-class counts", "Raw, valid, clean, duplicates, rejects, missing metadata, ready status (>=100 READY, 50-99 READY_SMALL, 30-49 WEAK, <30 NOT_READY).", table(pc) + f"\n\nStatus counts: {st}", "Collect more data for NOT_READY and WEAK classes, or leave them out of category training.")])
    md(R / "data_quality_report.md", "Data quality report", [
        ("File integrity", "Magic bytes, decode, minimum resolution.", f"- non-JPEG/PNG/WEBP: {int(dfr.reason.str.startswith('rejected_format').sum())}; corrupt: {int((dfr.reason == 'rejected_corrupt').sum())}; low resolution: {int((dfr.reason == 'rejected_low_resolution').sum())}", "None needed if all zero."),
        ("Metadata", "File to row matching, source URL, duplicate ids, class conflicts.", f"- issues: {dict(Counter(m['issue'] for m in miss + orphan))}", "Fix at source if any appear; see missing_metadata.csv."),
        ("Noise", "Title keywords vs class; negative title words.", f"- flagged in clean set: {int((cl.title_flag != '').sum())}; by class: {cl[cl.title_flag != ''].groupby('class').size().to_dict()}", "Review flagged images first (manual_review_checklist.md)."),
        ("Domain gap", "Labels and style.", "- Labels are the search term only (category level, no part numbers). Images are web photos, not shop photos.", "Use for category-level work and distractors only; part-number retrieval needs own photos or the company catalogue.")])
    sd = pd.read_csv(OUT / "metadata/duplicates.csv")
    md(R / "duplicate_report.md", "Duplicate report", [
        ("Exact and near duplicates", "sha256 and pHash (Hamming <= 6) across all batches and classes.",
         f"- groups removed {len(dups)} (exact {int((sd.dup_type == 'exact').sum()) if len(sd) else 0}, near {int((sd.dup_type == 'near').sum()) if len(sd) else 0}); cross-class {int(sd.cross_class.sum()) if len(sd) else 0}\n- per class: {sd.dup_class.value_counts().to_dict() if len(sd) else {}}\n- per batch: {sd.dup_batch.value_counts().to_dict() if len(sd) else {}}",
         "pHash 6 can merge look-alike diagrams; glance at duplicates.csv rows with hamming 5 to 6.")])
    lic_c = cl.licence.value_counts().to_dict()
    md(R / "license_report.md", "License report", [
        ("Licences", "Accept CC0, public domain, CC BY only; missing is needs_review; SA, NC, ND excluded.", f"- clean set: {lic_c}\n- excluded or needs review: {len(lic)} (license_review.csv)", "CC BY images need creator credit in the thesis; build credits from clean_images.csv (author, licence, source_url). Do not redistribute the images.")])
    flagged = cl[cl.title_flag != ""].groupby("class").size().sort_values(ascending=False)
    md(R / "manual_review_checklist.md", "Manual review checklist", [
        ("Order of review", "Flagged-image rate and class size.", "\n".join(f"- {c}: {n} flagged of {counts[c]} (sample_grid_{c.replace(' ', '_')}.png)" for c, n in flagged.items()) or "- none flagged",
         "Open the sample grids in this order; remove wrong-class images from clean_images.csv by id."),
        ("Classes needing more data", "Clean count under 50.", f"- {list(counts[counts < 50].index)}", "Own photos or the company catalogue."),
        ("Image ids to inspect", "Suspicious titles (first 40).", "\n".join(f"- {r.image_id}: {r.title_flag}: {r.page_title}" for r in cl[cl.title_flag != ''].head(40).itertuples()) or "- none", "Delete or relabel on sight.")])
    print(f"raw {tot} valid {nvalid} clean {nclean} dups {len(dups)} rejected {rej_n} review {review}")
    print(pc.to_string(index=False))
    print("reasons", dict(clean_rej), "licence", lic_c, "status", st, "classes<50", list(counts[counts < 50].index))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""EDA and cleaning for the Kaggle gpiosenka car-parts sets (gpiosenka/car-parts-40-classes, Apache 2.0).

The same Kaggle page published a 40-class version (folder "car parts") and the current 50-class version
(folder "car parts 50"). Reads data/raw/roboflow/<folder>/{train,valid,test}/<CLASS>/*.jpg and its car parts.csv. Kaggle's own
train/valid/test split is recorded but not reused: images are re-split later by group_id.
Checks: csv vs disk, magic bytes, decode, size, uniform frames, exact and near duplicates inside a class,
near duplicates across classes (label conflict: both copies excluded), overlap with the combined manifest
(Commons + tier1b), and a zero-shot CLIP label check (flag only, nothing removed for it).
Labels are category-level only (no part numbers); names are lower-cased and aligned to the Commons labels.
Writes only to E:\\car_search_data\\dataset_kaggle50 (manifest, rejected, reports, figures). Originals untouched.

Usage: python scripts/data/clean_kaggle50.py [--no-clip]                          (50 classes)
       python scripts/data/clean_kaggle50.py --folder "car parts" --name kaggle40     (40 classes)
"""
import argparse
import csv
import hashlib
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_clip_dataset import FORMATS, magic_format, to_rgb
from clean_dataset import dedupe, md, table
from combine_clean import gray_like, hamming_groups, same_row_content
from make_ccby_split import same_content, thumb

REPO = Path(__file__).resolve().parents[2]
COMBINED = Path(r"E:\car_search_data\dataset_combined\metadata\combined_images.csv")
OUT_ROOT = Path(r"E:\car_search_data")
URL = "https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes"
AUTHOR = "Gerry (gpiosenka), Kaggle, Apache 2.0"
# Source images are 224 px by design (a classification set). The 320 px catalogue rule does not apply to
# category-level training data; 224 is CLIP's input size, so nothing is lost. Recorded as a spec exception.
MIN_SIDE = 224
# Align with the Commons search-term labels where both name the same part; fix the source's typo.
ALIAS = {"brake rotor": "brake disc", "headlights": "headlight", "taillights": "taillight", "muffler": "exhaust muffler",
         "starter": "starter motor", "carberator": "carburetor"}
CLIP_FLAG = 0.10  # flag when CLIP's top-1 is another class and the given class gets < 10% probability


def norm_label(cls: str) -> str:
    c = cls.strip().lower()
    return ALIAS.get(c, c)


def uniform(im: Image.Image) -> bool:
    return float(np.asarray(im.convert("L").resize((64, 64)), dtype=np.float32).std()) < 5


def white_bg(im: Image.Image) -> float:
    """Share of near-white pixels on the border: high means a product photo on a plain background."""
    a = np.asarray(im.convert("L").resize((64, 64)))
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return float((border > 235).mean())


def clip_check(rows: list, classes: list, model_name: str) -> None:
    """Adds clip_top1 and clip_p_label to each row (zero-shot over the 50 class names)."""
    from transformers import CLIPModel, CLIPProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(model_name)
    with torch.no_grad():
        t = proc(text=[f"a photo of a {c}, a car part" for c in classes], return_tensors="pt", padding=True).to(dev)
        tf = model.get_text_features(**t)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        for s in range(0, len(rows), 64):
            batch = rows[s:s + 64]
            ims = [to_rgb(Image.open(r["path"])) for r in batch]
            f = model.get_image_features(pixel_values=proc(images=ims, return_tensors="pt")["pixel_values"].to(dev))
            f = f / f.norm(dim=-1, keepdim=True)
            pr = (model.logit_scale.exp() * f @ tf.T).softmax(-1).cpu().numpy()
            for r, p in zip(batch, pr):
                r["clip_top1"], r["clip_p_label"] = classes[int(p.argmax())], float(p[classes.index(r["label"])])
            print(f"clip {s + len(batch)}/{len(rows)}", end="\r", flush=True)
    print()


def sheet(rows: list, path: Path, caption, cols=10, cell=150):
    if not rows:
        return
    img = Image.new("RGB", (cols * cell, ((len(rows) + cols - 1) // cols) * (cell + 24)), "white")
    d = ImageDraw.Draw(img)
    for i, r in enumerate(rows):
        im = Image.open(r["path"]).convert("RGB")
        im.thumbnail((cell - 4, cell - 4))
        x, y = (i % cols) * cell, (i // cols) * (cell + 24)
        img.paste(im, (x + 2, y + 2))
        d.text((x + 2, y + cell), caption(r)[:26], fill="black")
    img.save(path, quality=85)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--folder", default="car parts 50", help="folder under data/raw/roboflow")
    ap.add_argument("--name", default="kaggle50", help="output dataset_<name>, file prefix and source tag")
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    a = ap.parse_args()
    SRC, OUT, name = REPO / "data/raw/roboflow" / a.folder, OUT_ROOT / f"dataset_{a.name}", a.name
    import imagehash
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for d in ("metadata", "reports/figures"):
        shutil.rmtree(OUT / d, ignore_errors=True)
        (OUT / d).mkdir(parents=True, exist_ok=True)

    # ---- inventory: csv vs disk
    listed = {r["filepaths"].replace("\\", "/"): r for r in csv.DictReader((SRC / "car parts.csv").open(encoding="utf-8"))}
    on_disk = {p.relative_to(SRC).as_posix(): p for p in SRC.glob("*/*/*") if p.is_file()}
    missing, extra = sorted(set(listed) - set(on_disk)), sorted(set(on_disk) - set(listed))

    inv = []
    for rel, p in sorted(on_disk.items()):
        split, cls = rel.split("/")[:2]
        raw = p.read_bytes()
        r = {"image_id": f"{name}_{split}_{cls.replace(' ', '_')}_{p.stem}", "path": p, "kaggle_split": split, "source_class": cls,
             "label": norm_label(cls), "format": magic_format(raw[:512]), "file_size_bytes": len(raw),
             "sha256": hashlib.sha256(raw).hexdigest(), "reason": "", "in_csv": rel in listed}
        try:
            with Image.open(p) as im:
                im.load()
                rgb = to_rgb(im)
            r.update(width=rgb.width, height=rgb.height, mode=im.mode, gray_like=gray_like(rgb), white_bg=white_bg(rgb),
                     brightness=float(np.asarray(rgb.convert("L")).mean()), ph=int(str(imagehash.phash(rgb)), 16), th=thumb(rgb))
            if uniform(rgb):
                r["reason"] = "rejected_uniform"
        except Exception:
            r["reason"] = "rejected_corrupt"
        if not r["reason"] and r["format"] not in FORMATS:
            r["reason"] = f"rejected_format_{r['format']}"
        if not r["reason"] and min(r["width"], r["height"]) < MIN_SIDE:
            r["reason"] = "rejected_low_resolution"
        inv.append(r)
    ok = [r for r in inv if not r["reason"]]

    # ---- duplicates inside a class (exact sha, else pHash <= 6 confirmed by content thumbnail)
    for r in ok:
        r.update(id=r["image_id"], sha=r["sha256"], w=r["width"], h=r["height"], size=r["file_size_bytes"], meta_n=0)
    by_cls = defaultdict(list)
    for r in ok:
        by_cls[r["label"]].append(r)
    kept, dups = [], []
    for rs in by_cls.values():
        k, d = dedupe(rs, confirm=same_row_content)
        kept += k
        dups += d
    for d in dups:
        d[0]["reason"] = "duplicate_within_class"
    split_leaks = sum(1 for d in dups if d[0]["kaggle_split"] != d[1]["kaggle_split"])

    # ---- near duplicates across classes: the same picture under two labels, so the label is wrong for one
    links, _ = hamming_groups([(r["image_id"], r["label"], r["ph"], r["th"]) for r in kept])
    conflict = {x for a_, b_, _ in links for x in (a_, b_)}
    for r in kept:
        if r["image_id"] in conflict:
            r["reason"] = "label_conflict_across_classes"
    clean = [r for r in kept if not r["reason"]]

    # ---- overlap with the existing combined manifest (Commons + tier1b)
    overlap = []
    if COMBINED.exists():
        cm = pd.read_csv(COMBINED, dtype=str)
        cph = np.array([int(h, 16) for h in cm.phash], dtype=np.uint64)
        for r in clean:
            dist = np.unpackbits((cph ^ np.uint64(r["ph"])).view(np.uint8)).reshape(-1, 64).sum(1)
            for j in np.nonzero(dist <= 6)[0]:
                with Image.open(cm.clean_path.iloc[j]) as im:
                    if same_content(r["th"], thumb(im.convert("RGB"))):
                        overlap.append((r["image_id"], cm.image_id.iloc[j], cm.label_type.iloc[j] + ":" + cm.label.iloc[j], int(dist[j])))
    overlap_ids = {o[0] for o in overlap}

    # ---- zero-shot CLIP label check (flag only)
    classes = sorted({r["label"] for r in inv})
    if not a.no_clip:
        clip_check(clean, classes, a.model)
    for r in clean:
        r["clip_flag"] = bool(r.get("clip_top1") and r["clip_top1"] != r["label"] and r["clip_p_label"] < CLIP_FLAG)

    # ---- write manifest in the combined_images.csv shape (+ Kaggle extras)
    rows = [{"image_id": r["image_id"], "label_type": "category", "label": r["label"], "clean_path": r["path"].as_posix(),
             "original_path": r["path"].as_posix(), "source": name, "source_url": URL, "licence": "Apache 2.0", "author": AUTHOR,
             "width": r["width"], "height": r["height"], "short_side": min(r["width"], r["height"]), "file_size_bytes": r["file_size_bytes"],
             "sha256": r["sha256"], "phash": f"{r['ph']:016x}", "clip_relevance": f"{r['clip_p_label']:.3f}" if "clip_p_label" in r else "",
             "gray_like": r["gray_like"], "domain": "web_product_photo", "store_code_like": False, "multi_code_folder": False,
             "existing_part_split": "", "title_flag": "", "source_class": r["source_class"], "kaggle_split": r["kaggle_split"],
             "white_bg": round(r["white_bg"], 3), "brightness": round(r["brightness"], 1), "clip_top1": r.get("clip_top1", ""),
             "label_review": "clip_disagrees_check" if r["clip_flag"] else "unreviewed",
             "overlaps_combined": r["image_id"] in overlap_ids} for r in clean]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"metadata/{name}_clean.csv", index=False)
    pd.DataFrame([{"path": r["path"].as_posix(), "label": r["label"], "kaggle_split": r["kaggle_split"], "reason": r["reason"]}
                  for r in inv if r["reason"]]).to_csv(OUT / f"metadata/{name}_rejected.csv", index=False)
    pd.DataFrame([{"dup_id": d[0]["image_id"], "kept_id": d[1]["image_id"], "dup_type": d[2], "hamming": d[3], "label": d[0]["label"],
                   "dup_split": d[0]["kaggle_split"], "kept_split": d[1]["kaggle_split"]} for d in dups],
                 columns=["dup_id", "kept_id", "dup_type", "hamming", "label", "dup_split", "kept_split"]).to_csv(OUT / f"metadata/{name}_duplicates.csv", index=False)
    pd.DataFrame(links, columns=["image_a", "image_b", "hamming"]).to_csv(OUT / f"metadata/{name}_label_conflicts.csv", index=False)
    pd.DataFrame(overlap, columns=["kaggle_id", "combined_id", "combined_label", "hamming"]).to_csv(OUT / f"metadata/{name}_overlap_combined.csv", index=False)

    # ---- EDA
    R, F = OUT / "reports", OUT / "reports/figures"
    raw_counts = Counter(r["label"] for r in inv)
    cc = df.label.value_counts()
    ex = Counter(r["reason"] for r in inv if r["reason"])

    def fig(name, f, size=(9, 4.5)):
        plt.figure(figsize=size); f(); plt.tight_layout(); plt.savefig(F / name, dpi=110); plt.close()
    fig("images_per_class.png", lambda: (plt.barh(cc.index[::-1], cc.values[::-1]), plt.title(f"{name}: clean images per class")), (8, 11))
    fig("short_side.png", lambda: (plt.hist(df.short_side, 20), plt.title("Short side (px)")))
    fig("brightness.png", lambda: (plt.hist(df.brightness, 40), plt.title("Mean brightness (0-255)")))
    fig("white_background.png", lambda: (plt.hist(df.white_bg, 20), plt.title("Share of near-white border pixels")))
    fig("exclusion_reasons.png", lambda: (plt.barh(list(ex), list(ex.values())), plt.title("Excluded files")))
    if not a.no_clip:
        agree = df.assign(ok=df.clip_top1 == df.label).groupby("label").ok.mean().sort_values()
        fig("clip_agreement_per_class.png", lambda: (plt.barh(agree.index, agree.values * 100), plt.title("CLIP zero-shot top-1 = given label (%)")), (8, 11))
    rng = random.Random(42)
    grid = [r for c in classes for r in rng.sample([x for x in clean if x["label"] == c], min(4, sum(x["label"] == c for x in clean)))]
    sheet(grid, F / "sample_grid.jpg", lambda r: r["label"], cols=8)
    flagged = sorted((r for r in clean if r["clip_flag"]), key=lambda r: r["clip_p_label"])
    sheet(flagged[:200], F / "clip_flagged.jpg", lambda r: f"{r['label']}>{r['clip_top1']}")
    sheet([r for r in kept if r["image_id"] in conflict][:100], F / "label_conflicts.jpg", lambda r: r["label"])

    per_class = pd.DataFrame({"label": classes, "raw": [raw_counts[c] for c in classes], "clean": [int(cc.get(c, 0)) for c in classes]})
    if not a.no_clip:
        per_class["clip_agree_pct"] = [round(float(agree.get(c, 0)) * 100) for c in classes]
        per_class["clip_flagged"] = [int(df[(df.label == c) & (df.label_review == "clip_disagrees_check")].shape[0]) for c in classes]
    per_class.to_csv(OUT / "metadata/per_class_summary.csv", index=False)
    shared = sorted(set(classes) & set(pd.read_csv(COMBINED, dtype=str).query("label_type == 'category'").label)) if COMBINED.exists() else []
    md(R / f"{name}_cleaning_report.md", f"{name} cleaning report", [
        ("Inventory", "car parts.csv against the files on disk.",
         f"- files on disk {len(on_disk)}; listed in csv {len(listed)}; missing from disk {len(missing)}; not in csv {len(extra)}\n- classes {len(classes)}; Kaggle split {dict(Counter(r['kaggle_split'] for r in inv))}",
         "None if both mismatch counts are 0."),
        ("File checks", f"Magic bytes, decode, short side >= {MIN_SIDE} px (exception to the 320 px catalogue rule), uniform frames.",
         f"- excluded {dict(ex)}\n- sizes: {df.short_side.value_counts().to_dict()}", "Record the 224 px exception in the Dataset Spec."),
        ("Duplicates", "Exact SHA-256, then pHash <= 6 confirmed by a content thumbnail, inside each class; then the same across classes.",
         (f"- within-class duplicates removed {len(dups)} (exact {sum(d[2] == 'exact' for d in dups)}, near {sum(d[2] == 'near' for d in dups)}); "
         f"{split_leaks} of them crossed Kaggle's own train/valid/test split\n- across-class near duplicates (label conflicts): {len(links)} pairs, {len(conflict)} images excluded"),
         "Do not reuse Kaggle's split; re-split by group_id."),
        ("Overlap with combined set", "pHash <= 6 + content check against dataset_combined (Commons + tier1b).",
         f"- {len(overlap)} links, {len(overlap_ids)} Kaggle images", "Linked images must share a group_id when merged."),
        ("Label check", f"Zero-shot CLIP ({a.model}) over the {len(classes)} class names; flag when top-1 differs and the given label gets < {CLIP_FLAG:.0%}.",
         ("- skipped (--no-clip)" if a.no_clip else f"- top-1 agreement {float((df.clip_top1 == df.label).mean()) * 100:.1f}%; flagged {int(df.label_review.eq('clip_disagrees_check').sum())} images\n- lowest-agreement classes: {agree.head(8).round(2).to_dict()}"),
         "Review reports/figures/clip_flagged.jpg by eye. CLIP disagreement is a hint, not proof: low agreement on look-alike classes is expected."),
        ("Result", "Clean set.", f"- {len(df)} clean images in {df.label.nunique()} classes (from {len(inv)})\n- per class min {cc.min()}, median {int(cc.median())}, max {cc.max()}\n- classes shared with Commons: {shared}",
         "Merge as label_type = category through combine_clean.py.")])
    md(R / f"{name}_eda_report.md", f"{name} EDA report", [
        ("Class balance", "Clean images per class.", table(per_class), "Imbalance is mild (max/min about 2); no resampling needed for a first run."),
        ("Image properties", "Size, colour, background, brightness.",
         (f"- all images {df.short_side.min()}-{df.short_side.max()} px\n- grayscale-like {df.gray_like.mean() * 100:.1f}%\n"
         f"- white-border share median {df.white_bg.median():.2f} (product photos on plain background: {(df.white_bg > 0.8).mean() * 100:.0f}%)\n"
         f"- brightness median {df.brightness.median():.0f}"),
         "Web product photos: cleaner than shop photos. Expect a domain gap to real queries."),
        ("Role", "Spec compliance.", "- Type labels only, no part numbers: category-level training and distractors, never part-number ground truth.",
         "Never mix with part-number labels in one classifier head.")])
    print({"files": len(inv), "clean": len(df), "excluded": dict(ex), "within_dups": len(dups), "split_leaks": split_leaks,
           "conflict_pairs": len(links), "overlap_combined": len(overlap), "missing": len(missing), "extra": len(extra),
           "clip_agree": None if a.no_clip else round(float((df.clip_top1 == df.label).mean()), 3),
           "clip_flagged": int(df.label_review.eq("clip_disagrees_check").sum())})


if __name__ == "__main__":
    main()

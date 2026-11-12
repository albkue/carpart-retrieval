#!/usr/bin/env python3
"""EDA of the merged type split (make_type_split.py output): balance, sources, image properties, leakage.

Reads E:\\car_search_data\\eda\\type_v1\\splits_type_v1_{train,eval}.json. Per image: role (train / gallery /
query seen / query held-out), type, source (from the path), size, grayscale, white border, brightness, pHash.
Leakage check: smallest pHash distance from each eval image to any training image, confirmed by the content
thumbnail (the same test used for deduplication). Writes eda/type_v1/eda/{eda_report.md, images.csv, figures/}.

Usage: python scripts/data/eda_type_split.py
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_dataset import md, table
from combine_clean import gray_like
from make_ccby_split import same_content, thumb

SPLIT = Path(r"E:\car_search_data\eda\type_v1")
OUT = SPLIT / "eda"


def source_of(path: str) -> str:
    p = path.replace("\\", "/").lower()
    return "kaggle50" if "car parts 50" in p else "kaggle40" if "/car parts/" in p else "commons" if "dataset_clean" in p else "other"


def white_border(im: Image.Image) -> float:
    a = np.asarray(im.convert("L").resize((64, 64)))
    return float((np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]) > 235).mean())


def min_dist_to(train_ph: np.ndarray, ph: int) -> np.ndarray:
    return np.unpackbits((train_ph ^ np.uint64(ph)).view(np.uint8)).reshape(-1, 64).sum(1)


def main():
    import imagehash
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    tr = json.loads((SPLIT / "splits_type_v1_train.json").read_text(encoding="utf-8"))
    ev = json.loads((SPLIT / "splits_type_v1_eval.json").read_text(encoding="utf-8"))
    entries = [(e, "train") for e in tr["catalog"]] + [(e, "gallery") for e in ev["catalog"]] + \
              [(e, "query_seen") for e in ev["queries"]["valid"]] + [(e, "query_heldout") for e in ev["queries"]["heldout"]]
    rows, ths = [], []
    for i, (e, role) in enumerate(entries):
        with Image.open(e["image"]) as im:
            rgb = im.convert("RGB")
            rows.append({"image": e["image"], "type": e["part_id"], "role": role, "source": source_of(e["image"]),
                         "width": rgb.width, "height": rgb.height, "gray_like": gray_like(rgb), "white_border": round(white_border(rgb), 3),
                         "brightness": round(float(np.asarray(rgb.convert("L")).mean()), 1), "phash": f"{int(str(imagehash.phash(rgb)), 16):016x}"})
            ths.append(thumb(rgb))
        if i % 2000 == 0:
            print(f"{i}/{len(entries)}", flush=True)
    df = pd.DataFrame(rows)
    heldout = set(tr.get("heldout_part_ids", []))
    df["heldout_type"] = df.type.isin(heldout)

    # ---- leakage: each eval image against all training images
    is_tr = (df.role == "train").to_numpy()
    tr_idx = np.nonzero(is_tr)[0]
    tr_ph = np.array([int(h, 16) for h in df.phash[is_tr]], dtype=np.uint64)
    mins, leaks = [], []
    for i in np.nonzero(~is_tr)[0]:
        d = min_dist_to(tr_ph, int(df.phash[i], 16))
        mins.append(int(d.min()))
        for j in np.nonzero(d <= 6)[0]:
            if same_content(ths[i], ths[tr_idx[j]]):
                leaks.append((df.image[i], df.image[tr_idx[j]], int(d[j])))
    df.loc[~is_tr, "min_phash_to_train"] = mins
    df.to_csv(OUT / "images.csv", index=False)
    pd.DataFrame(leaks, columns=["eval_image", "train_image", "hamming"]).to_csv(OUT / "leakage_pairs.csv", index=False)

    # ---- tables
    roles = ["train", "gallery", "query_seen", "query_heldout"]
    per_type = df.pivot_table(index="type", columns="role", values="image", aggfunc="count", fill_value=0).reindex(columns=roles, fill_value=0)
    per_type["total"] = per_type.sum(1)
    per_type["sources"] = df.groupby("type").source.agg(lambda s: ", ".join(f"{k} {v}" for k, v in Counter(s).most_common()))
    per_type["held_out"] = per_type.index.isin(heldout)
    per_type = per_type.sort_values("total", ascending=False)
    per_type.to_csv(OUT / "per_type.csv")
    by_src = df.groupby(["source", "role"]).size().unstack(fill_value=0).reindex(columns=roles, fill_value=0)

    # ---- figures
    def fig(name, f, size=(10, 4.5)):
        plt.figure(figsize=size); f(); plt.tight_layout(); plt.savefig(OUT / "figures" / name, dpi=110); plt.close()
    pt = per_type.iloc[::-1]

    def stacked():
        left = np.zeros(len(pt))
        for r in roles:
            plt.barh(pt.index, pt[r], left=left, label=r)
            left += pt[r].to_numpy()
        plt.legend(); plt.title("Images per type by role (held-out types have no train images)")
    fig("images_per_type_by_role.png", stacked, (9, 13))
    fig("source_mix.png", lambda: (by_src.plot.bar(stacked=True, ax=plt.gca()), plt.title("Images by source and role"), plt.xticks(rotation=0)))
    fig("short_side.png", lambda: ([plt.hist(df[df.source == s][["width", "height"]].min(1), 30, alpha=.6, label=s) for s in df.source.unique()], plt.legend(), plt.title("Short side (px) by source")))
    fig("brightness.png", lambda: ([plt.hist(df[df.source == s].brightness, 40, alpha=.6, label=s) for s in df.source.unique()], plt.legend(), plt.title("Mean brightness by source")))
    fig("white_border.png", lambda: ([plt.hist(df[df.source == s].white_border, 20, alpha=.6, label=s) for s in df.source.unique()], plt.legend(), plt.title("Share of near-white border pixels by source")))
    ev_d = df[~is_tr]
    fig("eval_to_train_min_phash.png", lambda: (plt.hist(ev_d.min_phash_to_train, bins=range(41)), plt.axvline(6.5, color="r"), plt.title("Eval image: smallest pHash distance to any training image (red: duplicate threshold)")))
    rng = random.Random(42)
    types = list(per_type.index)
    cell, cols = 110, 6
    sheet = Image.new("RGB", (cols * cell + 170, len(types) * cell), "white")
    dr = ImageDraw.Draw(sheet)
    for r_i, t in enumerate(types):
        imgs = df[df.type == t].image.tolist()
        dr.text((4, r_i * cell + cell // 2), t[:24], fill="red" if t in heldout else "black")
        for c_i, path in enumerate(rng.sample(imgs, min(cols, len(imgs)))):
            with Image.open(path) as im:
                im = im.convert("RGB"); im.thumbnail((cell - 4, cell - 4))
            sheet.paste(im, (170 + c_i * cell + 2, r_i * cell + 2))
    sheet.save(OUT / "figures" / "sample_grid_per_type.jpg", quality=80)

    # ---- report
    trc = per_type[~per_type.held_out]["train"]
    md(OUT / "eda_report.md", "Type split v1: EDA", [
        ("Size and balance", "Images per role and per type.",
         f"- {len(df)} images: train {int((df.role == 'train').sum())}, gallery {int((df.role == 'gallery').sum())}, queries seen {int((df.role == 'query_seen').sum())}, queries held-out {int((df.role == 'query_heldout').sum())}\n"
         f"- {df.type.nunique()} types ({len(heldout)} held out: {sorted(heldout)})\n- train images per seen type: min {trc.min()} ({trc.idxmin()}), median {int(trc.median())}, max {trc.max()} ({trc.idxmax()}); max/min ratio {trc.max() / trc.min():.1f}",
         "Imbalance is moderate; the P x K sampler draws types uniformly, so no reweighting is needed."),
        ("Sources", "Images by source and role.", table(by_src.reset_index()),
         "Commons is a small minority and only in some types: report per-source results if a type behaves oddly."),
        ("Image properties", "Size, colour, background, brightness by source.",
         "\n".join(f"- {s}: short side median {int(g[['width', 'height']].min(1).median())} px; grayscale-like {g.gray_like.mean() * 100:.0f}%; white border median {g.white_border.median():.2f}; brightness median {g.brightness.median():.0f}"
                   for s, g in df.groupby("source")),
         "Kaggle images are 224 px product photos on plain backgrounds; Commons images are larger, real-world photos. Both differ from shop photos."),
        ("Leakage", "Smallest pHash distance from every eval image to any training image; pairs <= 6 confirmed by the content thumbnail.",
         f"- confirmed near-duplicate pairs across train and eval: {len(leaks)}\n- eval images with a training image at distance <= 10 (unconfirmed, for context): {int((ev_d.min_phash_to_train <= 10).sum())} of {len(ev_d)}\n- median smallest distance: {int(ev_d.min_phash_to_train.median())}",
         "0 confirmed pairs means the split is clean at the duplicate threshold used for deduplication."),
        ("Per type", "Counts per role and source.", table(per_type.reset_index()), "See figures/sample_grid_per_type.jpg (held-out types in red)."),
    ])
    print({"images": len(df), "types": df.type.nunique(), "leaks": len(leaks), "median_min_phash": int(ev_d.min_phash_to_train.median())})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""EDA and leakage check for the CC BY spare-part set, then copy the prepared data to another drive.

Reads data/processed/splits_ccby_v0_{train,eval}.json (from make_ccby_split.py) and writes, under --out:
  eda_report.md, eda_stats.json, plots/*.png, contact_sheet.jpg
  prepared/images/<part>/<n>.jpg  the unique images used (copied, nothing modified)
  prepared/splits_ccby_v0_{train,eval}.json  same manifests with absolute image paths
  prepared/LICENCE_AND_SOURCES.txt  CC BY 4.0 credit note

Leakage check: frozen CLIP embeddings; for every validation image the most similar training image.
Usage:
    python scripts/data/eda_ccby.py --out E:/car_search_data/eda/ccby_v0
"""
import argparse
import json
import re
import shutil
import statistics as st
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PATTERNS = [("honda-style 5-3-3", r"^\d{5}-[A-Z0-9]{3}-[A-Z0-9]{2,3}"), ("two or more codes", r".+-.+"),
            ("letter prefix + digits", r"^[A-Z]{1,4}\d+"), ("digits only", r"^\d+$")]


def part_pattern(name: str) -> str:
    for label, rx in PATTERNS:
        if re.match(rx, name):
            return label
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", type=Path, default=REPO / "data/processed/splits_ccby_v0_train.json")
    ap.add_argument("--eval", type=Path, default=REPO / "data/processed/splits_ccby_v0_eval.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    a = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from torchvision import transforms as T
    from transformers import CLIPModel

    train_m = json.loads(a.train.read_text(encoding="utf-8"))
    eval_m = json.loads(a.eval.read_text(encoding="utf-8"))
    rows = ([("train", e) for e in train_m["catalog"]] + [("valid_gallery", e) for e in eval_m["catalog"]]
            + [("valid_query", e) for e in eval_m["queries"]["valid"]] + [("far_unknown", e) for e in eval_m["far_unknown"]])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "plots").mkdir(exist_ok=True)

    # ---- per-image stats
    stats = []
    for split, e in rows:
        with Image.open(REPO / e["image"]) as im:
            rgb = im.convert("RGB")
            arr = np.asarray(rgb.resize((64, 64)), dtype=np.float32)
            gray_diff = float(np.abs(arr[..., 0] - arr[..., 1]).mean() + np.abs(arr[..., 1] - arr[..., 2]).mean())
            lap = np.asarray(ImageOps.grayscale(rgb.resize((128, 128))), dtype=np.float32)
            blur = float(((lap[1:-1, 1:-1] * 4 - lap[:-2, 1:-1] - lap[2:, 1:-1] - lap[1:-1, :-2] - lap[1:-1, 2:]) ** 2).mean())
            stats.append({"split": split, "part": e["part_id"], "path": e["image"], "w": im.width, "h": im.height,
                          "kb": round((REPO / e["image"]).stat().st_size / 1024, 1), "gray": gray_diff < 4.0,
                          "mean_brightness": float(arr.mean()), "sharpness": blur, "fmt": (REPO / e["image"]).suffix.lower()})
    per_part = Counter(s["part"] for s in stats if s["split"] in ("train", "valid_gallery", "valid_query", "far_unknown"))
    train_parts = {s["part"] for s in stats if s["split"] == "train"}
    side = [min(s["w"], s["h"]) for s in stats]
    out = {
        "images": len(stats), "parts": len(per_part), "by_split": Counter(s["split"] for s in stats),
        "images_per_part": {"min": min(per_part.values()), "median": st.median(per_part.values()),
                            "mean": round(st.mean(per_part.values()), 2), "max": max(per_part.values()),
                            "hist": dict(sorted(Counter(per_part.values()).items()))},
        "short_side_px": {"min": min(side), "median": st.median(side), "max": max(side),
                          "under_320": sum(1 for x in side if x < 320)},
        "grayscale_share": round(sum(s["gray"] for s in stats) / len(stats), 3),
        "formats": dict(Counter(s["fmt"] for s in stats)),
        "median_kb": st.median(s["kb"] for s in stats),
        "blurry_share_sharpness_lt_20": round(sum(s["sharpness"] < 20 for s in stats) / len(stats), 3),
        "part_number_patterns": dict(Counter(part_pattern(p) for p in per_part)),
        "multi_code_folders": sum(1 for p in per_part if p.count("-") >= 1 and re.search(r"[A-Z]?\d{5,}-[A-Z]?\d{5,}", p)),
    }

    # ---- leakage check with frozen CLIP
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(a.model).to(dev).eval()
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758))])
    embs = []
    for i in range(0, len(stats), 32):
        x = torch.stack([tf(Image.open(REPO / s["path"]).convert("RGB")) for s in stats[i:i + 32]]).to(dev)
        with torch.no_grad():
            f = model.visual_projection(model.vision_model(pixel_values=x).pooler_output)
        embs.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
    E = torch.cat(embs).numpy()
    tr = np.array([s["split"] == "train" for s in stats])
    ev = np.array([s["split"] in ("valid_gallery", "valid_query") for s in stats])
    sim = E[ev] @ E[tr].T
    best = sim.max(axis=1)
    out["leakage_valid_to_train_max_cosine"] = {
        "median": round(float(np.median(best)), 3), "p90": round(float(np.percentile(best, 90)), 3),
        "max": round(float(best.max()), 3), "share_over_0.95": round(float((best > 0.95).mean()), 3),
        "share_over_0.90": round(float((best > 0.90).mean()), 3)}
    # same-part vs nearest other-part similarity inside the validation set
    vs = [s for s in stats if s["split"] in ("valid_gallery", "valid_query")]
    ve = E[ev]
    parts_v = np.array([s["part"] for s in vs])
    own, other = [], []
    for i in range(len(vs)):
        m = parts_v == parts_v[i]
        m[i] = False
        s_all = ve @ ve[i]
        s_all[i] = -1
        if m.any():
            own.append(float(s_all[m].max()))
        other.append(float(s_all[parts_v != parts_v[i]].max()))
    out["valid_own_part_vs_other_part_max_cosine"] = {"own_median": round(float(np.median(own)), 3),
                                                      "other_median": round(float(np.median(other)), 3)}
    (a.out / "eda_stats.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")

    # ---- plots and contact sheet
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        ax[0].bar(list(out["images_per_part"]["hist"].keys()), list(out["images_per_part"]["hist"].values()), color="#1B2B45")
        ax[0].set(title="Images per part", xlabel="images", ylabel="parts")
        ax[1].hist(side, bins=30, color="#F5A524")
        ax[1].set(title="Short side (px)", xlabel="px", ylabel="images")
        ax[2].hist(best, bins=30, color="#B23A3A")
        ax[2].set(title="Validation to nearest train image (cosine)", xlabel="cosine")
        fig.tight_layout()
        fig.savefig(a.out / "plots" / "overview.png", dpi=130)
    except Exception as e:  # plots are optional
        print("plots skipped:", e)
    import random
    rng = random.Random(0)
    sample = rng.sample(sorted(train_parts), min(12, len(train_parts)))
    sheet = Image.new("RGB", (4 * 200, 3 * 200), "white")
    for k, p in enumerate(sample):
        f = next(s["path"] for s in stats if s["part"] == p)
        sheet.paste(ImageOps.fit(Image.open(REPO / f).convert("RGB"), (200, 200)), ((k % 4) * 200, (k // 4) * 200))
    sheet.save(a.out / "contact_sheet.jpg", quality=88)

    # ---- copy prepared data
    prep = a.out / "prepared"
    (prep / "images").mkdir(parents=True, exist_ok=True)
    mapping = {}
    for s in stats:
        dst = prep / "images" / s["part"] / Path(s["path"]).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(REPO / s["path"], dst)
        mapping[s["path"]] = dst.as_posix()

    def remap(entries):
        return [{**e, "image": mapping[e["image"]]} for e in entries]

    t2 = {**train_m, "catalog": remap(train_m["catalog"])}
    e2 = {**eval_m, "catalog": remap(eval_m["catalog"]), "far_unknown": remap(eval_m["far_unknown"]),
          "queries": {"valid": remap(eval_m["queries"]["valid"])}}
    (prep / a.train.name).write_text(json.dumps(t2, indent=1) + "\n", encoding="utf-8")
    (prep / a.eval.name).write_text(json.dumps(e2, indent=1) + "\n", encoding="utf-8")
    (prep / "LICENCE_AND_SOURCES.txt").write_text(
        "Images come from Roboflow Universe spare-part datasets released under CC BY 4.0:\n"
        "- Complete Spare Parts (supply-spare-parts/complete-spare-parts)\n"
        "- Supply Spare Parts (supply-spare-parts/supply-spare-parts)\n"
        "- Supply Spare Parts Detection (supply-spare-parts/supply-spare-parts-detection)\n"
        "Credit the dataset owners in the thesis. Labels are folder names, unverified.\n", encoding="utf-8")

    L = out["leakage_valid_to_train_max_cosine"]
    report = [
        "# EDA: CC BY spare-part set (provisional split v0)", "",
        f"- Images: {out['images']} in {out['parts']} parts; by split: {dict(out['by_split'])}",
        f"- Images per part: median {out['images_per_part']['median']}, mean {out['images_per_part']['mean']}, "
        f"max {out['images_per_part']['max']}; histogram {out['images_per_part']['hist']}",
        f"- Short side px: median {out['short_side_px']['median']}, min {out['short_side_px']['min']}, "
        f"under 320: {out['short_side_px']['under_320']}",
        f"- Grayscale-like images: {out['grayscale_share']:.1%}; formats {out['formats']}; median file {out['median_kb']} KB",
        f"- Blurry-looking (low sharpness): {out['blurry_share_sharpness_lt_20']:.1%}",
        f"- Part-number patterns: {out['part_number_patterns']}",
        "", "## Leakage check (frozen CLIP, validation image to most similar train image)",
        f"- Median cosine {L['median']}, p90 {L['p90']}, max {L['max']}",
        f"- Share above 0.95: {L['share_over_0.95']:.1%}; above 0.90: {L['share_over_0.90']:.1%}",
        f"- Inside validation: same-part best match median {out['valid_own_part_vs_other_part_max_cosine']['own_median']} "
        f"vs other-part {out['valid_own_part_vs_other_part_max_cosine']['other_median']}",
        "", "High shares above 0.95 mean look-alike photos (augmented copies) sit in both train and validation.",
        "", "Plots: plots/overview.png. Sample parts: contact_sheet.jpg."]
    (a.out / "eda_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()

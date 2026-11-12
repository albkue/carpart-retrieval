#!/usr/bin/env python3
"""Hyperparameter search for A5 on the TUNING split only (never the test queries).

Stage 1 (arm B): plain fine-tune, lr_ft x temperature, short runs (--epochs, default 2).
Stage 2 (arm C): LP-FT, lr_proj in {1e-3, 1e-2}, with the best temperature and lr_lpft_ft = best lr_ft / 10.
Each run trains on the tuning-train split and is scored on the tuning gallery + tuning queries: Recall@1, then mAP.
Writes <out>/search_results.csv, <out>/best.json and prints the settings to put in configs/a5/a5.yaml.
Short runs get a short warm-up (10% of the steps): the config's 500 warm-up steps would exceed a 408-step run and
leave the learning rate rising the whole time.

Usage:
    python scripts/search_a5.py --out E:/car_search_data/results/a5_search
    python scripts/search_a5.py --score E:/car_search_data/checkpoints/x/model.pt      (internal: score one checkpoint)
"""
import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_a5 import type_metrics
from eval_checkpoint import CLIP_MEAN, CLIP_STD

REPO = Path(__file__).resolve().parent.parent
TUNE = Path(r"E:\car_search_data\eda\type_v1\tune")
CKPT = Path(r"E:\car_search_data\checkpoints\search")
LR_FT, TEMPS, LR_PROJ = [1e-5, 3e-5, 1e-4], [0.03, 0.05, 0.07], [1e-3, 1e-2]


def rank_key(row: dict) -> tuple:
    """Higher tuning Recall@1 wins; ties go to mAP."""
    return (row["tune_recall@1"], row["tune_mAP"])


def best_of(rows: list) -> dict:
    return max(rows, key=rank_key)


def score_one(ckpt: Path, model_name: str, limit: int | None = None) -> dict:
    """Recall@1 and mAP of one checkpoint on the tuning gallery and tuning queries (untouched by training)."""
    from PIL import Image
    from torchvision import transforms as T
    from transformers import CLIPVisionModelWithProjection
    ev = json.loads((TUNE / "splits_type_v1_tune_eval.json").read_text(encoding="utf-8"))
    gallery, queries = ev["catalog"], ev["queries"]["valid"]
    if limit:
        keep = sorted({r["part_id"] for r in queries})[:limit]
        gallery, queries = [r for r in gallery if r["part_id"] in keep], [r for r in queries if r["part_id"] in keep]
    dev = "cuda"
    model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(dev).eval()
    proj = torch.nn.Linear(768, 768, bias=False).to(dev)
    with torch.no_grad():
        proj.weight.copy_(torch.eye(768))
        state = torch.load(ckpt, map_location=dev, weights_only=True)
        proj.load_state_dict(state["proj"])
        if "vision_model" in state:
            model.vision_model.load_state_dict(state["vision_model"])
            model.visual_projection.load_state_dict(state["visual_projection"])
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(CLIP_MEAN, CLIP_STD)])

    def embed(rows):
        out = []
        for i in range(0, len(rows), 32):
            x = torch.stack([tf(Image.open(r["image"]).convert("RGB")) for r in rows[i:i + 32]]).to(dev)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
                f = model.visual_projection(model.vision_model(pixel_values=x).pooler_output)
                out.append(torch.nn.functional.normalize(proj(f.float()).float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    g, q = embed(gallery), embed(queries)
    m = type_metrics(q @ g.T, [r["part_id"] for r in gallery], [r["part_id"] for r in queries])
    return {"tune_recall@1": round(float(m["recall@1"].mean()), 4), "tune_mAP": round(float(m["ap"].mean()), 4), "n_queries": len(queries)}


def run_one(args, mode: str, out: Path, extra: list) -> dict:
    """Train one setting (subprocess, so the GPU is freed), then score it (second subprocess)."""
    py = sys.executable
    cmd = [py, str(REPO / "scripts" / "train_embedding.py"), "--config", str(args.config), "--mode", mode,
           "--epochs", str(args.epochs), "--warmup", str(args.warmup), "--out", str(out)] + extra
    if args.smoke:
        cmd += ["--max-types", "3", "--max-steps", "2"]
    t0 = time.time()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    sc = [py, str(Path(__file__).resolve()), "--score", str(out / "model.pt")] + (["--limit", "3"] if args.smoke else [])
    res = json.loads(subprocess.run(sc, check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1])
    res["minutes"] = round((time.time() - t0) / 60, 1)
    (out / "model.pt").unlink()  # 1.2 GB each; the search keeps only scores
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "a5" / "a5.yaml")
    ap.add_argument("--epochs", type=int, default=2, help="short budget per run")
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--smoke", action="store_true", help="3 types, 2 steps per run: checks the plumbing only")
    ap.add_argument("--score", type=Path, help="internal: print the tuning score of one checkpoint as JSON")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.score:
        print(json.dumps(score_one(args.score, args.model, args.limit)))
        return
    assert args.out, "--out is required"
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for lr, temp in itertools.product(LR_FT, TEMPS):
        tag = f"B_lr{lr:g}_t{temp:g}"
        r = run_one(args, "ft", CKPT / tag, ["--lr-ft", str(lr), "--temperature", str(temp)])
        rows.append({"stage": "B", "lr_ft": lr, "temperature": temp, "lr_proj": "", **r})
        print(tag, r, flush=True)
    bestB = best_of([r for r in rows if r["stage"] == "B"])
    lr_lpft = bestB["lr_ft"] / 10
    for lp in LR_PROJ:
        tag = f"C_proj{lp:g}"
        r = run_one(args, "lpft", CKPT / tag, ["--lr-proj", str(lp), "--lr-lpft-ft", str(lr_lpft), "--temperature", str(bestB["temperature"])])
        rows.append({"stage": "C", "lr_ft": lr_lpft, "temperature": bestB["temperature"], "lr_proj": lp, **r})
        print(tag, r, flush=True)
    bestC = best_of([r for r in rows if r["stage"] == "C"])
    with (args.out / "search_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    best = {"selected_on": "tuning queries only (Recall@1, then mAP)", "epochs_per_run": args.epochs,
            "B": {"lr_ft": bestB["lr_ft"], "temperature": bestB["temperature"], "tune_recall@1": bestB["tune_recall@1"]},
            "C": {"lr_proj": bestC["lr_proj"], "lr_lpft_ft": lr_lpft, "temperature": bestB["temperature"], "tune_recall@1": bestC["tune_recall@1"]}}
    (args.out / "best.json").write_text(json.dumps(best, indent=1) + "\n", encoding="utf-8")
    print("\nPut in configs/a5/a5.yaml (train section):")
    print(f"  temperature: {bestB['temperature']}\n  lr_ft: {bestB['lr_ft']:g}\n  lr_proj: {bestC['lr_proj']:g}\n  lr_lpft_ft: {lr_lpft:g}")


if __name__ == "__main__":
    main()

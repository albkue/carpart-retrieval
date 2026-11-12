#!/usr/bin/env python3
"""A5 hyperparameter search on the TUNING split only (short runs, one variable at a time).

Stage 1: B (full FT) temperature {0.03, 0.05, 0.07} at lr_ft 3e-5.
Stage 2: B lr_ft {1e-5, 1e-4} at the best temperature (3e-5 reused from stage 1).
Stage 3: C (LP-FT) lr_proj {1e-3, 1e-2} at the best temperature; stage-2 lr = best lr_ft / 10.
Ranking uses tune Recall@1, then tune mAP. Held-out and seen metrics are never printed or used.
Each run trains short_budget_epochs (config) on the tuning-train split; finished runs are skipped on re-run.

Usage: python scripts/hp_search_a5.py [--epochs 2] [--out E:/car_search_data/results/a5hp]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CKPT = Path("E:/car_search_data/checkpoints/a5hp")


def run(name, mode, epochs, out_root, **kw):
    ckpt, res = CKPT / name, out_root / name
    metrics = res / "metrics_a1.00.json"
    if not metrics.exists():
        warm = max(10, (204 * epochs) // 10)  # config warm-up (500 steps) would outlast a short run
        cmd = [sys.executable, str(REPO / "scripts/train_embedding.py"), "--config", str(REPO / "configs/a5/a5.yaml"),
               "--mode", mode, "--epochs", str(epochs), "--warmup", str(warm), "--out", str(ckpt)]
        for k, v in kw.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        for attempt in (1, 2):  # a native crash (exit 0xC0000005) hit one run once, 2026-10-29; retry from scratch
            if subprocess.run(cmd, cwd=REPO).returncode == 0:
                break
        else:
            raise SystemExit(f"{name}: training failed twice")
        subprocess.run([sys.executable, str(REPO / "scripts/eval_a5.py"), "--checkpoint", str(ckpt / "model.pt"),
                        "--alphas", "1.0", "--limit-far", "20", "--out", str(res)], check=True, cwd=REPO)
        (ckpt / "model.pt").unlink()  # 1.2 GB each; the metrics are what the search keeps
    t = json.loads(metrics.read_text())["sets"]["tune"]
    return t["recall@1"]["value"], t["mAP"]["value"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("E:/car_search_data/results/a5hp"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("USE_TF", "0")
    rows = []

    def trial(name, mode, **kw):
        r1, mp = run(name, mode, a.epochs, a.out, **kw)
        rows.append({"run": name, "mode": mode, **kw, "tune_recall@1": r1, "tune_mAP": mp})
        print(f"== {name}: tune R@1 {r1:.4f} mAP {mp:.4f}", flush=True)
        with open(a.out / "hp_search.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        return r1, mp

    def best(names):
        return max(names, key=lambda n: next((r["tune_recall@1"], r["tune_mAP"]) for r in rows if r["run"] == n))

    temps = {t: f"B_t{t}" for t in (0.03, 0.05, 0.07)}
    for t, n in temps.items():
        trial(n, "ft", temperature=t, lr_ft=3e-5)
    bt = next(t for t, n in temps.items() if n == best(list(temps.values())))

    lrs = {3e-5: temps[bt]}
    for lr in (1e-5, 1e-4):
        lrs[lr] = f"B_lr{lr:g}"
        trial(lrs[lr], "ft", temperature=bt, lr_ft=lr)
    bl = next(lr for lr, n in lrs.items() if n == best(list(lrs.values())))

    cs = {}
    for lp in (1e-3, 1e-2):
        cs[lp] = f"C_proj{lp:g}"
        trial(cs[lp], "lpft", temperature=bt, lr_proj=lp, lr_lpft_ft=bl / 10)
    bp = next(lp for lp, n in cs.items() if n == best(list(cs.values())))
    print(f"\nWINNERS: temperature {bt}, lr_ft {bl:g}, lr_proj {bp:g}, lr_lpft_ft {bl / 10:g}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""A5 seeds: wait for hp_search_a5.py, then train and evaluate B (ft) and C (lpft) for seeds 42, 43, 44.

Reads the `WINNERS:` line from the search log, trains 10 epochs on the tuning-train split with those settings,
and evaluates each checkpoint on every set with the WiSE-FT alpha sweep. If the winners equal the config defaults,
seed 42 is not re-run: the existing a5v2 checkpoints and results are reused. Finished runs are skipped.

Usage: python scripts/run_a5_seeds.py
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG = Path("E:/car_search_data/results/hp_search.log")
CKPT, RES = Path("E:/car_search_data/checkpoints/a5seeds"), Path("E:/car_search_data/results/a5seeds")
DEFAULTS = (0.05, 3e-5, 1e-3)


def winners():
    while True:
        text = LOG.read_text(errors="replace") if LOG.exists() else ""
        m = re.search(r"WINNERS: temperature ([\d.e-]+), lr_ft ([\d.e-]+), lr_proj ([\d.e-]+)", text)
        if m:
            return tuple(float(x) for x in m.groups())
        if "Traceback" in text:
            sys.exit("hp search failed, see " + str(LOG))
        time.sleep(60)


def main():
    os.environ.setdefault("USE_TF", "0")
    temp, lr_ft, lr_proj = winners()
    print("winners", temp, lr_ft, lr_proj, flush=True)
    for arm, mode in (("B", "ft"), ("C", "lpft")):
        for seed in (42, 43, 44):
            if seed == 42 and (temp, lr_ft, lr_proj) == DEFAULTS:
                print(f"{arm} seed 42: reusing a5v2", flush=True)
                continue
            name = f"{arm}_s{seed}"
            ckpt, res = CKPT / name, RES / name
            if not (res / "metrics_a1.00.json").exists():
                cmd = [sys.executable, str(REPO / "scripts/train_embedding.py"), "--config", str(REPO / "configs/a5/a5.yaml"),
                       "--mode", mode, "--epochs", "10", "--seed", str(seed), "--temperature", str(temp),
                       "--lr-ft", str(lr_ft), "--lr-proj", str(lr_proj), "--lr-lpft-ft", str(lr_ft / 10), "--out", str(ckpt)]
                subprocess.run(cmd, check=True, cwd=REPO)
                subprocess.run([sys.executable, str(REPO / "scripts/eval_a5.py"), "--checkpoint", str(ckpt / "model.pt"),
                                "--alphas", "0,0.25,0.5,0.75,1", "--out", str(res)], check=True, cwd=REPO)
            print("done", name, flush=True)


if __name__ == "__main__":
    main()

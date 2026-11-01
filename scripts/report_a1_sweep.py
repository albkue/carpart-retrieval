#!/usr/bin/env python3
"""A1 crop sweep report: overall and per-source Recall@K / mAP per region, paired bootstrap against full_image.

Reads results/<hash>/{metrics.json,per_query.csv} for every run whose config name is --name, joins queries to the
split's `src` tag, prints a markdown table. Usage: python scripts/report_a1_sweep.py [--name a1_type_v2]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
ORDER = ["full_image", "centre_70", "crop_00", "crop_10", "crop_15", "crop_20"]


def paired_ci(a: np.ndarray, b: np.ndarray, n=2000, seed=0):
    """Mean difference b - a with a 95% percentile bootstrap CI over queries."""
    rng = np.random.default_rng(seed)
    d = b - a
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="a1_type_v2")
    a = ap.parse_args()
    runs = {}
    for mp in (REPO / "results").glob("*/metrics.json"):
        m = json.loads(mp.read_text())
        if m.get("name") == a.name:
            cfg = yaml.safe_load((mp.parent / "config.resolved.yaml").read_text())
            pq = pd.read_csv(mp.parent / "per_query.csv")
            runs[m["region"]] = (m, pq, cfg)
    cfg = next(iter(runs.values()))[2]
    split = json.loads(Path(cfg["data"]["split_manifest"]).read_text())
    src = {q["image"]: q["src"] for q in split["queries"][cfg["data"]["eval_split"]]}
    rows = []
    base = runs["full_image"][1].assign(src=lambda d: d.image.map(src))
    for r in ORDER:
        m, pq, _ = runs[r]
        pq = pq.assign(src=pq.image.map(src))
        row = {"region": r, "detected": m["detection_rate"]}
        for name, sel in (("all", slice(None)), ("kaggle50", pq.src == "kaggle50"), ("hitl", pq.src == "hitl"), ("commons", pq.src == "commons")):
            q = pq if isinstance(sel, slice) else pq[sel]
            row[f"R@1 {name}"], row[f"mAP {name}"] = q["recall@1"].mean(), q["ap"].mean()
            row[f"n {name}"] = len(q)
        d, lo, hi = paired_ci(base["recall@1"].to_numpy(), pq["recall@1"].to_numpy())
        row["dR@1 vs full"], row["lo"], row["hi"] = d, lo, hi
        det = pq[pq.detected.astype(str) == "True"] if "detected" in pq else pq.iloc[:0]
        if len(det):
            bd = base.loc[det.index]
            row["R@1 detected-only"], row["R@1 same, full_image"] = det["recall@1"].mean(), bd["recall@1"].mean()
            row["n detected"] = len(det)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("region")
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(df.round(3).T.to_string())
    df.round(4).to_csv(REPO / "results" / f"{a.name}_report.csv")


if __name__ == "__main__":
    main()

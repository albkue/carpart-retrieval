#!/usr/bin/env python3
"""A5 summary across arms: A frozen, B FT, C LP-FT, D WiSE(A,B), E WiSE(A,C).

Reads eval_a5.py outputs: <results>/A/metrics_a0.00.json, <results>/B_D/alpha_table.csv, <results>/C_E/alpha_table.csv
(+ selected_alpha.json). Writes summary.csv and summary.md with, per arm: Recall@1 / mAP on seen, held-out and
Commons queries; trade-off = (seen gain) - (held-out gain) relative to frozen; feature distortion; abstention.
D and E are reported at the pre-registered alpha 0.5 and at the tuning-selected alpha. For each WiSE-FT curve it
checks whether any interior alpha falls below the straight line between alpha 0 and alpha 1.

Usage: python scripts/report_a5.py --results E:/car_search_data/results/a5
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def below_line(alphas: list, values: list) -> list:
    """Interior alphas whose value is below the linear interpolation of the two endpoints."""
    (a0, v0), (a1, v1) = (alphas[0], values[0]), (alphas[-1], values[-1])
    return [a for a, v in zip(alphas[1:-1], values[1:-1]) if v < v0 + (v1 - v0) * (a - a0) / (a1 - a0) - 1e-12]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, required=True)
    a = ap.parse_args()
    R = a.results
    A = json.loads((R / "A" / "metrics_a0.00.json").read_text(encoding="utf-8"))
    rowA = {"arm": "A frozen", "alpha": 0.0}
    for s, v in A["sets"].items():
        rowA[f"{s}_recall@1"], rowA[f"{s}_mAP"] = v["recall@1"]["value"], v["mAP"]["value"]
    for k, v in A.get("abstention", {}).items():
        if not isinstance(v, str):
            rowA[f"abst_{k}"] = v
    rows, curves = [rowA], []
    for folder, ft_name, wise_name in (("B_D", "B FT", "D WiSE(A,B)"), ("C_E", "C LP-FT", "E WiSE(A,C)")):
        t = pd.read_csv(R / folder / "alpha_table.csv").sort_values("alpha")
        sel = json.loads((R / folder / "selected_alpha.json").read_text(encoding="utf-8"))["selected_alpha"]
        pick = lambda al: t.iloc[(t.alpha - al).abs().argmin()].to_dict()
        rows += [{"arm": ft_name, **pick(1.0)}, {"arm": f"{wise_name} a=0.5 (primary)", **pick(0.5)},
                 {"arm": f"{wise_name} a={sel} (tuning-selected)", **pick(sel)}]
        for col in ("seen_recall@1", "heldout_recall@1", "seen_mAP", "heldout_mAP"):
            if col in t:
                curves.append({"curve": wise_name, "metric": col, "below_line_at": below_line(t.alpha.tolist(), t[col].tolist())})
    df = pd.DataFrame(rows)
    for m in ("recall@1", "mAP"):
        if f"seen_{m}" in df and f"heldout_{m}" in df:
            df[f"tradeoff_{m}"] = (df[f"seen_{m}"] - rowA[f"seen_{m}"]) - (df[f"heldout_{m}"] - rowA[f"heldout_{m}"])
    keep = [c for c in ["arm", "alpha", "seen_recall@1", "seen_mAP", "heldout_recall@1", "heldout_mAP", "commons_recall@1",
                        "tradeoff_recall@1", "tradeoff_mAP", "distortion_seen", "distortion_heldout", "distortion_commons",
                        "distortion_far_unknown", "abst_false_abstention_seen", "abst_correct_rejection_far",
                        "abst_auroc_known_vs_far"] if c in df]
    df[keep].to_csv(R / "summary.csv", index=False)
    md = ["# A5 summary (type-level retrieval)\n", df[keep].round(4).to_markdown(index=False), "\n## WiSE-FT curves below the straight line\n"]
    md += [f"- {c['curve']} {c['metric']}: {c['below_line_at'] or 'none'}" for c in curves]
    (R / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

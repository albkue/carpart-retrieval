#!/usr/bin/env python3
"""A5 figures averaged over seeds: alpha curves (R@1, mAP), abstention, distortion trade-off.

Reads <seeds>/<arm>_s<seed>/alpha_table.csv for B (WiSE-FT A,B) and C (WiSE-FT A,C); alpha 1 is B / C itself.
Usage: python scripts/plot_a5_seeds.py [--seeds E:/car_search_data/results/a5seeds] [--out DIR]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ARMS = {"B": ("full FT / WiSE(A,B)", "#c0392b"), "C": ("LP-FT / WiSE(A,C)", "#1f77b4")}


def load(root, arm):
    df = pd.concat([pd.read_csv(root / f"{arm}_s{s}" / "alpha_table.csv").assign(seed=s) for s in (42, 43, 44)])
    g = df.groupby("alpha")
    return g.mean(numeric_only=True), g.std(numeric_only=True)


def line(ax, data, col, title, ylabel):
    for arm, (label, colour) in ARMS.items():
        m, s = data[arm]
        ax.plot(m.index, m[col], "o-", color=colour, label=label)
        ax.fill_between(m.index, m[col] - s[col], m[col] + s[col], color=colour, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("alpha (0 = frozen, 1 = fine-tuned)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=Path, default=Path("E:/car_search_data/results/a5seeds"))
    ap.add_argument("--out", type=Path, default=Path("C:/Documents/Obsidian/Sothy Knowledge/01-Thesis/Research/images"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = {arm: load(a.seeds, arm) for arm in ARMS}

    for metric, name in (("recall@1", "recall1"), ("mAP", "map")):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        line(axs[0], data, f"seen_{metric}", f"Seen types: {metric}", metric)
        line(axs[1], data, f"heldout_{metric}", f"Held-out types: {metric}", "")
        axs[0].legend()
        fig.suptitle("A5 alpha curve, mean ± sd over seeds 42/43/44")
        fig.tight_layout()
        fig.savefig(a.out / f"a5_alpha_{name}.png", dpi=160)
        plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    line(axs[0], data, "abst_false_abstention_seen", "False abstention, seen", "rate")
    line(axs[1], data, "abst_false_abstention_heldout", "False abstention, held-out", "rate")
    line(axs[2], data, "abst_auroc_known_vs_far", "AUROC known vs far unknown (motorcycle)", "AUROC")
    axs[0].legend()
    fig.suptitle("A5 abstention (threshold = 5th percentile on tuning queries)")
    fig.tight_layout()
    fig.savefig(a.out / "a5_abstention.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for arm, (label, colour) in ARMS.items():
        m, _ = data[arm]
        ax.plot(m["distortion_seen"], m["heldout_recall@1"], "o-", color=colour, label=label)
        for al, x, y in zip(m.index, m["distortion_seen"], m["heldout_recall@1"]):
            ax.annotate(f"{al:g}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("feature distortion on seen queries")
    ax.set_ylabel("held-out recall@1")
    ax.set_title("Distortion vs held-out recall (labels = alpha)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.out / "a5_distortion_tradeoff.png", dpi=160)

    for arm in ARMS:
        m, _ = data[arm]
        print(arm)
        print(m[["seen_recall@1", "heldout_recall@1", "seen_mAP", "heldout_mAP", "distortion_seen",
                 "abst_false_abstention_seen", "abst_false_abstention_heldout", "abst_correct_rejection_far",
                 "abst_auroc_known_vs_far"]].round(3).to_string())


if __name__ == "__main__":
    main()

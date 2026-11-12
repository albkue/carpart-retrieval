#!/usr/bin/env python3
"""A5 evaluation: frozen, fine-tuned or WiSE-FT-blended CLIP on the type split, with abstention and distortion.

Architecture for every arm: CLIP image encoder -> 768x768 projection (identity when frozen) -> L2 norm.
For a fine-tuned checkpoint, --alphas sweeps WiSE-FT blends with frozen CLIP (alpha 1.0 = the checkpoint itself).

Per model it reports, on seen-type queries, held-out-type queries, the Commons-only subset of both, and the
tuning queries: Recall@1/5/10 and MRR (types ranked by their best gallery image), mAP (image level: every
gallery image of the query's type is relevant), top-1 type accuracy, per-type accuracy and a confusion matrix.
Feature distortion = 1 - cos(frozen embedding, new embedding) per query set, including motorcycle far unknowns.
Abstention: threshold = 5th percentile of the max gallery similarity on the TUNING queries (5% false abstention
there, pre-registered); applied unchanged to test queries and to the motorcycle far unknowns.
Alpha is selected on tuning-query Recall@1 only (then mAP); held-out, Commons and far-unknown results never pick it.
Search is exact cosine in memory (same ranking as an exact Qdrant cosine search).

Usage:
    python scripts/eval_a5.py --checkpoint frozen --out E:/car_search_data/results/a5/A_frozen
    python scripts/eval_a5.py --checkpoint E:/car_search_data/checkpoints/a5_B/model.pt --alphas 0,0.25,0.5,0.75,1 --out E:/car_search_data/results/a5/B_D
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_checkpoint import CLIP_MEAN, CLIP_STD, bootstrap_ci
from wise_ft import blend, frozen_state

TYPE = Path(r"E:\car_search_data\eda\type_v1")
FALSE_ABSTAIN = 0.05


def type_metrics(sim: np.ndarray, g_types: list, q_types: list, ks=(1, 5, 10)) -> dict:
    """Per-query arrays: recall@k, rr (reciprocal rank of the true type), ap (image-level AP), top1 predicted type."""
    from sklearn.metrics import average_precision_score
    g = np.array(g_types)
    types = sorted(set(g_types))
    best = np.stack([sim[:, g == t].max(1) for t in types], 1)  # (queries, types): best image per type
    order = np.argsort(-best, 1)
    out = {f"recall@{k}": [] for k in ks}
    out.update(rr=[], ap=[], top1=[])
    for i, truth in enumerate(q_types):
        ranked = [types[j] for j in order[i]]
        rank = ranked.index(truth) + 1 if truth in ranked else None
        for k in ks:
            out[f"recall@{k}"].append(float(rank is not None and rank <= k))
        out["rr"].append(1.0 / rank if rank else 0.0)
        rel = (g == truth).astype(int)
        out["ap"].append(float(average_precision_score(rel, sim[i])) if rel.any() else 0.0)
        out["top1"].append(ranked[0])
    return {k: (np.array(v) if k != "top1" else v) for k, v in out.items()}


def tuning_leak(checkpoint: Path, tune_rows: list) -> int:
    """How many tuning images appear in the training split recorded in the checkpoint's train_log.json (0 if unknown)."""
    log = checkpoint.parent / "train_log.json"
    if not log.exists():
        return 0
    split = Path(json.loads(log.read_text(encoding="utf-8"))["args"]["split"])
    if not split.exists():
        return 0
    seen = {r["image"] for r in json.loads(split.read_text(encoding="utf-8"))["catalog"]}
    return sum(r["image"] in seen for r in tune_rows)


def abstain_threshold(max_sim_known: np.ndarray, false_abstain=FALSE_ABSTAIN) -> float:
    """Threshold below which a query is 'not confidently in catalogue': the given quantile of known max-sims."""
    return float(np.quantile(max_sim_known, false_abstain))


def summarise(m: dict, ci=True) -> dict:
    s = {}
    for k in ("recall@1", "recall@5", "recall@10", "rr", "ap"):
        name = {"rr": "MRR", "ap": "mAP"}.get(k, k)
        v = m[k]
        s[name] = {"value": round(float(v.mean()), 4)} if len(v) else {"value": None}
        if ci and len(v):
            lo, hi = bootstrap_ci(v)
            s[name].update(ci_low=round(lo, 4), ci_high=round(hi, 4))
    s["top1_type_accuracy"] = s["recall@1"]["value"]
    s["n"] = len(m["rr"])
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default="frozen")
    ap.add_argument("--alphas", default="1.0", help="WiSE-FT alphas for a fine-tuned checkpoint, e.g. 0,0.25,0.5,0.75,1")
    ap.add_argument("--eval-split", type=Path, default=TYPE / "splits_type_v1_eval.json")
    ap.add_argument("--train-split", type=Path, default=TYPE / "splits_type_v1_train.json")
    ap.add_argument("--tune-split", type=Path, default=TYPE / "tune" / "splits_type_v1_tune_eval.json")
    ap.add_argument("--far-unknowns", type=Path, default=Path(r"E:\car_search_data\dataset_combined\metadata\combined_images.csv"))
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--limit-types", type=int, default=None, help="smoke test: keep only N seen and N held-out types")
    ap.add_argument("--limit-far", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import pandas as pd
    from PIL import Image
    from sklearn.metrics import confusion_matrix, roc_auc_score
    from torchvision import transforms as T
    from transformers import CLIPVisionModelWithProjection

    ev = json.loads(a.eval_split.read_text(encoding="utf-8"))
    held = set(json.loads(a.train_split.read_text(encoding="utf-8")).get("heldout_part_ids", []))
    tune = json.loads(a.tune_split.read_text(encoding="utf-8"))
    gallery, q_seen, q_held = ev["catalog"], ev["queries"]["valid"], ev["queries"]["heldout"]
    t_gallery, t_queries = tune["catalog"], tune["queries"]["valid"]
    if a.limit_types:
        keep = set(sorted({r["part_id"] for r in q_seen})[:a.limit_types]) | set(sorted(held)[:a.limit_types])
        f = lambda rows: [r for r in rows if r["part_id"] in keep]
        gallery, q_seen, q_held, t_gallery, t_queries = map(f, (gallery, q_seen, q_held, t_gallery, t_queries))
    cm = pd.read_csv(a.far_unknowns, dtype=str)
    far = (cm[cm.label_type == "part"].sort_values("image_id").groupby("label").head(1).clean_path.tolist())
    if a.limit_far:
        far = far[:a.limit_far]
    is_commons = lambda r: "dataset_clean" in r["image"].replace("\\", "/")
    sets = {"seen": q_seen, "heldout": q_held, "commons": [r for r in q_seen + q_held if is_commons(r)], "tune": t_queries}
    print({k: len(v) for k, v in sets.items()}, "gallery", len(gallery), "tune gallery", len(t_gallery), "far", len(far), flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPVisionModelWithProjection.from_pretrained(a.model).to(dev).eval()
    proj = torch.nn.Linear(768, 768, bias=False).to(dev)
    # kept on the GPU: blending runs there (CPU tensor maths crashed with an access violation on Windows when
    # scikit-learn was loaded in the same process, 2026-10-23); tuned tensors move over one at a time in blend()
    frozen = {g: {k: v.to(dev) for k, v in sd.items()} for g, sd in frozen_state(model).items()}
    tuned = None if a.checkpoint == "frozen" else torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    leaked = tuning_leak(Path(a.checkpoint), t_gallery + t_queries) if tuned is not None else 0
    if leaked:  # alpha and the abstention threshold are chosen on the tuning set: it must be unseen in training
        print(f"WARNING: {leaked} tuning images were in this checkpoint's training split; alpha selection and the "
              "abstention threshold are not valid (train on the tuning-train split)", flush=True)
    alphas = [0.0] if tuned is None else [float(x) for x in a.alphas.split(",")]

    # decode each image once (uint8, 224 px centre crop); normalise on the GPU per batch
    prep = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.PILToTensor()])
    paths = sorted({r["image"] for r in gallery + t_gallery + q_seen + q_held + t_queries} | set(far))
    cache = {}
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            cache[p] = prep(im.convert("RGB"))
        if i % 1000 == 0:
            print(f"decoded {i}/{len(paths)}", flush=True)
    mean = torch.tensor(CLIP_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=dev).view(1, 3, 1, 1)

    def load_state(state):
        with torch.no_grad():
            model.vision_model.load_state_dict({k: v.to(dev) for k, v in state["vision_model"].items()})
            model.visual_projection.load_state_dict({k: v.to(dev) for k, v in state["visual_projection"].items()})
            proj.load_state_dict({k: v.to(dev) for k, v in state["proj"].items()})

    def embed(ps: list) -> dict:
        out = {}
        for i in range(0, len(ps), a.batch):
            chunk = ps[i:i + a.batch]
            x = (torch.stack([cache[p] for p in chunk]).to(dev).float() / 255 - mean) / std
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                f = model.visual_projection(model.vision_model(pixel_values=x).pooler_output)
                e = torch.nn.functional.normalize(proj(f.float()).float(), dim=-1).cpu().numpy()
            out.update(zip(chunk, e))
        return out

    load_state(frozen)
    E0 = embed(paths)  # frozen embeddings, for distortion
    a.out.mkdir(parents=True, exist_ok=True)
    rows, per_alpha = [], {}
    for alpha in alphas:
        if tuned is None or alpha == 0.0:
            E = E0
        else:
            load_state(blend(frozen, tuned, alpha))
            E = embed(paths)
        G = np.stack([E[r["image"]] for r in gallery]); g_types = [r["part_id"] for r in gallery]
        TG = np.stack([E[r["image"]] for r in t_gallery]); tg_types = [r["part_id"] for r in t_gallery]
        res = {"checkpoint": a.checkpoint, "alpha": alpha, "sets": {}, "distortion": {}}
        maxsim = {}
        for name, qs in sets.items():
            if not qs:
                continue
            Q = np.stack([E[r["image"]] for r in qs])
            gq, gt = (TG, tg_types) if name == "tune" else (G, g_types)
            sim = Q @ gq.T
            m = type_metrics(sim, gt, [r["part_id"] for r in qs])
            res["sets"][name] = summarise(m)
            maxsim[name] = sim.max(1)
            d = 1 - np.sum(Q * np.stack([E0[r["image"]] for r in qs]), 1)
            res["distortion"][name] = {"mean": round(float(d.mean()), 5), "median": round(float(np.median(d)), 5)}
            if name in ("seen", "heldout"):
                truth = [r["part_id"] for r in qs]
                labels = sorted(set(gt))
                acc = defaultdict(list)
                for t, p in zip(truth, m["top1"]):
                    acc[t].append(float(t == p))
                pd.DataFrame([{"type": t, "n": len(v), "top1_accuracy": round(float(np.mean(v)), 4)} for t, v in sorted(acc.items())]) \
                    .to_csv(a.out / f"per_type_{name}_a{alpha:.2f}.csv", index=False)
                pd.DataFrame(confusion_matrix(truth, m["top1"], labels=labels), index=labels, columns=labels) \
                    .to_csv(a.out / f"confusion_{name}_a{alpha:.2f}.csv")
        F = np.stack([E[p] for p in far]) if far else np.zeros((0, 768))
        far_max = (F @ G.T).max(1) if len(F) else np.array([])
        res["distortion"]["far_unknown"] = {"mean": round(float((1 - np.sum(F * np.stack([E0[p] for p in far]), 1)).mean()), 5)} if len(F) else {}
        thr = abstain_threshold(maxsim["tune"]) if "tune" in maxsim else None
        if thr is not None:
            res["abstention"] = {"threshold": round(thr, 4), "chosen_on": f"tuning queries, {FALSE_ABSTAIN:.0%} false abstention",
                                 "false_abstention_seen": round(float((maxsim["seen"] < thr).mean()), 4) if "seen" in maxsim else None,
                                 "false_abstention_heldout": round(float((maxsim["heldout"] < thr).mean()), 4) if "heldout" in maxsim else None,
                                 "correct_rejection_far": round(float((far_max < thr).mean()), 4) if len(far_max) else None,
                                 "n_far": len(far_max)}
            known = np.concatenate([maxsim.get("seen", []), maxsim.get("heldout", [])])
            if len(far_max) and len(known):
                res["abstention"]["auroc_known_vs_far"] = round(float(roc_auc_score(np.r_[np.ones(len(known)), np.zeros(len(far_max))], np.r_[known, far_max])), 4)
        per_alpha[alpha] = res
        (a.out / f"metrics_a{alpha:.2f}.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        flat = {"alpha": alpha}
        for name, s in res["sets"].items():
            for k in ("recall@1", "recall@5", "recall@10", "MRR", "mAP"):
                flat[f"{name}_{k}"] = s[k]["value"]
        flat.update({f"distortion_{k}": v.get("mean") for k, v in res["distortion"].items()})
        flat.update({f"abst_{k}": v for k, v in res.get("abstention", {}).items() if not isinstance(v, str)})
        rows.append(flat)
        print(f"alpha {alpha}: " + ", ".join(f"{n} R@1 {s['recall@1']['value']}" for n, s in res["sets"].items()), flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(a.out / "alpha_table.csv", index=False)
    if len(alphas) > 1 and "tune_recall@1" in table:
        best = table.sort_values(["tune_recall@1", "tune_mAP"], ascending=False).iloc[0]
        (a.out / "selected_alpha.json").write_text(json.dumps(
            {"selected_alpha": float(best.alpha), "rule": "max tuning-query Recall@1, then mAP", "preregistered_primary_alpha": 0.5}, indent=1), encoding="utf-8")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for metric in ("recall@1", "mAP"):
            plt.figure(figsize=(6, 4))
            for name in ("seen", "heldout"):
                col = f"{name}_{metric}"
                if col in table:
                    plt.plot(table.alpha, table[col], "o-", label=name)
                    plt.plot([table.alpha.iloc[0], table.alpha.iloc[-1]], [table[col].iloc[0], table[col].iloc[-1]], "--", alpha=.5)
            plt.xlabel("alpha (0 = frozen, 1 = fine-tuned)"); plt.ylabel(metric); plt.legend(); plt.title(f"WiSE-FT curve: {metric}")
            plt.tight_layout(); plt.savefig(a.out / f"alpha_curve_{metric.replace('@', '')}.png", dpi=110); plt.close()
    print(f"done -> {a.out}")


if __name__ == "__main__":
    main()

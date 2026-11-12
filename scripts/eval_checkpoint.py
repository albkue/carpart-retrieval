#!/usr/bin/env python3
"""Score a frozen or fine-tuned CLIP image encoder on an eval manifest (Recall@K, MRR, bootstrap CI).

Matches the harness metrics: parts ranked by their best image, one relevant part per query,
AP = reciprocal rank, percentile bootstrap over queries. Exact cosine search in memory.

Usage:
    python scripts/eval_checkpoint.py --manifest data/processed/splits_ccby_v0_eval.json --checkpoint frozen
    python scripts/eval_checkpoint.py --manifest ... --checkpoint checkpoints/run1/model.pt --out checkpoints/run1/eval.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CLIP_MEAN, CLIP_STD = (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)


def score(sim: np.ndarray, gallery_parts: list, query_parts: list, ks=(1, 5, 10)) -> dict:
    """sim is (queries, gallery). Returns per-query recall@k and reciprocal rank arrays."""
    rows = {f"recall@{k}": [] for k in ks}
    rows["mAP"] = []
    for i, truth in enumerate(query_parts):
        order = np.argsort(-sim[i])
        ranked = list(dict.fromkeys(gallery_parts[j] for j in order))
        rank = ranked.index(truth) + 1 if truth in ranked else None
        rows["mAP"].append(1.0 / rank if rank else 0.0)
        for k in ks:
            rows[f"recall@{k}"].append(float(rank is not None and rank <= k))
    return {k: np.array(v) for k, v in rows.items()}


def bootstrap_ci(values: np.ndarray, resamples=1000, ci=0.95, seed=42):
    idx = np.random.default_rng(seed).integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    tail = (1 - ci) / 2 * 100
    return float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--checkpoint", default="frozen", help="'frozen' or a model.pt from train_embedding.py")
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    import torch
    from PIL import Image
    from torchvision import transforms as T
    from transformers import CLIPModel

    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
    gallery, queries = manifest["catalog"], manifest["queries"][a.split]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(a.model).to(dev).eval()
    proj = torch.nn.Linear(768, 768, bias=False).to(dev)
    with torch.no_grad():
        proj.weight.copy_(torch.eye(768))
        if a.checkpoint != "frozen":
            state = torch.load(a.checkpoint, map_location=dev, weights_only=True)  # tensors only
            proj.load_state_dict(state["proj"])
            if "vision_model" in state:
                model.vision_model.load_state_dict(state["vision_model"])
                model.visual_projection.load_state_dict(state["visual_projection"])
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                    T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])

    def embed(items):
        out = []
        for i in range(0, len(items), 32):
            x = torch.stack([tf(Image.open(REPO / e["image"]).convert("RGB")) for e in items[i:i + 32]]).to(dev)
            with torch.no_grad():
                f = model.visual_projection(model.vision_model(pixel_values=x).pooler_output)
                out.append(torch.nn.functional.normalize(proj(f).float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    g, q = embed(gallery), embed(queries)
    rows = score(q @ g.T, [e["part_id"] for e in gallery], [e["part_id"] for e in queries])
    result = {"checkpoint": a.checkpoint, "manifest": str(a.manifest), "n_queries": len(queries),
              "n_gallery": len(gallery), "metrics": {}}
    for k, v in rows.items():
        lo, hi = bootstrap_ci(v)
        result["metrics"][k] = {"value": round(float(v.mean()), 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}
        print(f"{k:10s} {v.mean():.3f}  [{lo:.3f}, {hi:.3f}]")
    if a.out:
        a.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

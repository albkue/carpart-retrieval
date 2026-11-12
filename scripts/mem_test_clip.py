#!/usr/bin/env python3
"""GPU memory test for A5 training: ViT-L/14 + 768x768 projection, same settings as train_embedding.py.

For each batch size, runs a few optimiser steps of full fine-tuning (AdamW over vision encoder + projections,
bf16 autocast, gradient checkpointing, gradient clipping) and of projection-only training, on random pixels
with P x K labels, and reports peak allocated / reserved memory and seconds per step. Catches OOM per setting.

Usage: python scripts/mem_test_clip.py --batches 8,16,32 [--no-checkpointing]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_embedding import contrastive_loss


def run(model, proj, mode: str, batch: int, steps: int, k=4) -> dict:
    for q in model.parameters():
        q.requires_grad_(mode == "ft")
    params = list(proj.parameters()) + (list(model.vision_model.parameters()) + list(model.visual_projection.parameters()) if mode == "ft" else [])
    opt = torch.optim.AdamW(params, lr=1e-6, weight_decay=0.1, fused=True)
    labels = [f"t{i // k}" for i in range(batch)]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = []
    try:
        for _ in range(steps):
            x = torch.randn(batch, 3, 224, 224, device="cuda")
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = model.visual_projection(model.vision_model(pixel_values=x).pooler_output)
                emb = torch.nn.functional.normalize(proj(f).float(), dim=-1)
            loss = contrastive_loss(emb, labels, 0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        res = {"ok": True, "sec_per_step": round(sum(times[1:]) / max(1, len(times) - 1), 3)}
    except torch.OutOfMemoryError:
        res = {"ok": False, "error": "OOM"}
    res.update(mode=mode, batch=batch, peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2))
    del opt
    for q in model.parameters():
        q.grad = None
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batches", default="8,16,32")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--no-checkpointing", action="store_true")
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    from transformers import CLIPVisionModelWithProjection
    model = CLIPVisionModelWithProjection.from_pretrained(a.model).cuda().train()
    if not a.no_checkpointing:
        model.gradient_checkpointing_enable()
    proj = torch.nn.Linear(768, 768, bias=False).cuda()
    with torch.no_grad():
        proj.weight.copy_(torch.eye(768))
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    results = []
    for b in [int(x) for x in a.batches.split(",")]:
        for mode in ("lp", "ft"):
            r = run(model, proj, mode, b, a.steps)
            results.append(r)
            print(json.dumps(r), flush=True)
    summary = {"gpu": torch.cuda.get_device_name(0), "total_gb": round(total, 2), "gradient_checkpointing": not a.no_checkpointing,
               "precision": "bf16 autocast, fp32 master weights, fused AdamW, vision tower only", "results": results}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))


if __name__ == "__main__":
    main()

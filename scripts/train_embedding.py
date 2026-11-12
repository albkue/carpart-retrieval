#!/usr/bin/env python3
"""Fine-tune the CLIP image encoder for part-number retrieval (full FT, LP, or LP-FT).

Model: CLIP ViT-L/14 (same weights as the service, ADR 001) + an identity-initialised
768->768 projection. At step 0 the embedding equals frozen CLIP's; the script asserts it.
Loss: in-batch supervised contrastive on L2-normalised embeddings; positives are other
images of the same part. Batches are P parts x K images. bf16 autocast + gradient
checkpointing + fused AdamW + vision tower only, because full FT overflows 8 GB without them (measured 2026-10-21
and 2026-10-23: foreach AdamW ran out of memory at batch 32).

Only train on data whose licence allows it. The eBay set is NOT cleared (API licence
section 9, item 10); the manifest's `heldout_part_ids`, if present, are asserted out.

Labels can be part numbers or part types (A5 type experiment): the sampler only needs a part_id per image.
With --epoch-unit images an epoch is one pass over the training images (ceil(images / (P*K)) batches) instead of
one pass over the part ids; use it when there are few ids with many images each (types).

Usage:
    python scripts/train_embedding.py --split data/processed/splits_toy.json --mode lp --epochs 2
    python scripts/train_embedding.py --split <manifest> --mode lpft --epochs 4 --out checkpoints/run1
    python scripts/train_embedding.py --config configs/a5/a5.yaml --mode ft --out E:/car_search_data/checkpoints/a5_B
"""
import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLIP_MEAN, CLIP_STD = (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)


def parts_from_manifest(manifest: dict) -> dict[str, list[str]]:
    """part_id -> training image paths (catalogue entries), parts with >= 2 images only.

    Asserts no held-out part is in the training set: one leak invalidates the open-set result.
    """
    parts = defaultdict(list)
    for row in manifest["catalog"]:
        parts[row["part_id"]].append(row["image"])
    leaked = set(parts) & set(manifest.get("heldout_part_ids", []))
    assert not leaked, f"held-out parts in the training set: {sorted(leaked)[:5]}"
    return {p: imgs for p, imgs in parts.items() if len(imgs) >= 2}


def batches(parts: dict[str, list[str]], p: int, k: int, rng: random.Random):
    """One epoch of P x K batches: (image paths, part labels). Images repeat if a part has < K."""
    ids = sorted(parts)
    rng.shuffle(ids)
    for i in range(0, len(ids) - p + 1, p):
        paths, labels = [], []
        for pid in ids[i:i + p]:
            imgs = parts[pid]
            picks = rng.sample(imgs, k) if len(imgs) >= k else [rng.choice(imgs) for _ in range(k)]
            paths += picks
            labels += [pid] * k
        yield paths, labels


def lpft_ft_lr(explicit, lr_ft) -> float:
    """LP-FT phase-2 learning rate: an explicit value, else plain FT's rate / 10 (3e-5 default -> 3e-6)."""
    return explicit if explicit is not None else (lr_ft or 3e-5) / 10


def epoch_batches(parts: dict[str, list[str]], p: int, k: int, rng: random.Random, steps: int):
    """Exactly `steps` P x K batches, cycling through shuffled passes over the part ids."""
    done = 0
    while done < steps:
        for b in batches(parts, p, k, rng):
            yield b
            done += 1
            if done == steps:
                return


def steps_per_epoch(parts: dict[str, list[str]], p: int, k: int, unit: str) -> int:
    """'parts': one pass over the ids; 'images': enough batches to see every training image once on average."""
    if unit == "images":
        return max(1, math.ceil(sum(len(v) for v in parts.values()) / (p * k)))
    return len(parts) // p


def contrastive_loss(emb, labels, temperature=0.05):
    """Supervised in-batch contrastive loss; emb is (n, d) L2-normalised, labels a list of n ids."""
    import torch
    ids = {l: i for i, l in enumerate(dict.fromkeys(labels))}
    lab = torch.tensor([ids[l] for l in labels], device=emb.device)
    n = len(labels)
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    logits = (emb @ emb.T / temperature).masked_fill(eye, -1e4)
    pos = (lab[:, None] == lab[None, :]) & ~eye
    logp = torch.log_softmax(logits, dim=1)
    return -(logp * pos).sum(1).div(pos.sum(1).clamp(min=1)).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--mode", choices=["ft", "lp", "lpft"], default="lp")
    ap.add_argument("--epochs", type=int, default=2, help="lpft: half projection-only, half full")
    ap.add_argument("--parts-per-batch", type=int, default=8)
    ap.add_argument("--images-per-part", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None, help="default: ft 3e-5, lp 1e-3, lpft second phase 3e-6")
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--out", type=Path, default=REPO / "checkpoints" / "run")
    ap.add_argument("--config", type=Path, help="YAML whose 'train' section sets defaults (CLI flags still win)")
    ap.add_argument("--epoch-unit", choices=["parts", "images"], default="parts")
    ap.add_argument("--warmup", type=int, default=None, help="warm-up steps per phase (default: min(500, total // 10))")
    ap.add_argument("--lr-proj", type=float, default=None, help="projection-only phase LR (lp, first half of lpft)")
    ap.add_argument("--lr-ft", type=float, default=None, help="full fine-tune LR for --mode ft (plain FT)")
    ap.add_argument("--lr-lpft-ft", type=float, default=None,
                    help="lpft second-phase LR (default: lr_ft / 10, Kumar et al. 2022); never shares lr_ft, so a config's plain-FT rate cannot leak into LP-FT")
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--max-types", type=int, default=None, help="smoke test: keep only the first N part ids")
    ap.add_argument("--max-steps", type=int, default=None, help="smoke test: stop each phase after N steps")
    for action in ap._actions:  # --split may come from --config instead
        if action.dest == "split":
            action.required = False
    pre, _ = ap.parse_known_args()
    if pre.config:
        import yaml
        cfg = yaml.safe_load(pre.config.read_text(encoding="utf-8")).get("train", {})
        ap.set_defaults(**{k.replace("-", "_"): (Path(v) if k in ("split", "out") else v) for k, v in cfg.items()})
    args = ap.parse_args()
    assert args.split, "--split (or train.split in --config) is required"

    import torch
    from PIL import Image
    from torchvision import transforms as T
    from transformers import CLIPVisionModelWithProjection

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    parts = parts_from_manifest(json.loads(Path(args.split).read_text(encoding="utf-8")))
    if args.max_types:
        parts = {pid: parts[pid] for pid in sorted(parts)[:args.max_types]}
    p, k = min(args.parts_per_batch, len(parts)), args.images_per_part
    assert len(parts) >= 2, f"need at least 2 parts with 2+ images, found {len(parts)}"
    n_steps = steps_per_epoch(parts, p, k, args.epoch_unit)

    dev = "cuda"
    model = CLIPVisionModelWithProjection.from_pretrained(args.model).to(dev)  # vision tower only: same weights and embeddings, no unused text tower on the GPU
    model.gradient_checkpointing_enable()
    proj = torch.nn.Linear(768, 768, bias=False).to(dev)
    with torch.no_grad():
        proj.weight.copy_(torch.eye(768))

    tf = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.9, 1.1),
                                        interpolation=T.InterpolationMode.BICUBIC),  # no horizontal flip: left/right parts differ
                    T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])
    load = lambda paths: torch.stack([tf(Image.open(REPO / q).convert("RGB")) for q in paths]).to(dev)

    def embed(x):
        feats = model.visual_projection(model.vision_model(pixel_values=x).pooler_output)
        return torch.nn.functional.normalize(proj(feats).float(), dim=-1)

    # identity check: at step 0 the embedding must equal frozen CLIP's
    with torch.no_grad():
        x0 = load(next(batches(parts, 2, 1, random.Random(0)))[0])
        ref = torch.nn.functional.normalize(model(pixel_values=x0).image_embeds.float(), dim=-1)
        cos = (embed(x0) * ref).sum(-1).min().item()
    assert cos > 0.9999, f"projection is not identity at init (cos {cos})"

    if args.mode == "lpft":
        phases = [("lp", args.epochs // 2, args.lr_proj or 1e-3), ("ft", args.epochs - args.epochs // 2, lpft_ft_lr(args.lr_lpft_ft, args.lr_ft))]
    elif args.mode == "lp":
        phases = [("lp", args.epochs, args.lr_proj or args.lr or 1e-3)]
    else:
        phases = [("ft", args.epochs, args.lr_ft or args.lr or 3e-5)]
    log, t_start = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    step = 0
    for phase, epochs, lr in phases:
        for q in model.parameters():
            q.requires_grad_(phase == "ft")
        params = list(proj.parameters()) + (list(model.vision_model.parameters()) + list(model.visual_projection.parameters())
                                           if phase == "ft" else [])
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.1, fused=True)  # fused: no foreach intermediates (~ one extra copy of the params)
        total = max(1, epochs * n_steps)
        if args.max_steps:
            total = min(total, args.max_steps)
        warm = max(1, min(args.warmup, total) if args.warmup is not None else min(500, total // 10))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s, warm=warm, total=total: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
        model.train()
        phase_step = 0
        for ep in range(epochs):
            if args.max_steps and phase_step >= args.max_steps:
                break
            for paths, labels in epoch_batches(parts, p, k, rng, n_steps):
                if args.max_steps and phase_step >= args.max_steps:
                    break
                t0 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    emb = embed(load(paths))
                loss = contrastive_loss(emb, labels, args.temperature)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
                opt.step()
                sched.step()
                torch.cuda.synchronize()
                step += 1
                phase_step += 1
                log.append({"step": step, "phase": phase, "epoch": ep, "loss": round(loss.item(), 4),
                            "sec": round(time.time() - t0, 2)})
                print(f"{phase} ep{ep} step{step} loss {loss.item():.4f} {log[-1]['sec']}s", flush=True)
        if args.mode == "lpft" and phase == "lp":  # LP-FT stage 1, kept so stage 2's starting point is on record
            args.out.mkdir(parents=True, exist_ok=True)
            torch.save({"proj": proj.state_dict()}, args.out / "model_stage1.pt")

    args.out.mkdir(parents=True, exist_ok=True)
    state = {"proj": proj.state_dict()}
    if args.mode != "lp":
        state["vision_model"] = model.vision_model.state_dict()
        state["visual_projection"] = model.visual_projection.state_dict()
    torch.save(state, args.out / "model.pt")
    summary = {"args": {a: str(v) for a, v in vars(args).items()}, "train_parts": len(parts),
               "batch": p * k, "steps": step, "identity_cos_at_init": round(cos, 6),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "total_sec": round(time.time() - t_start, 1), "log": log}
    (args.out / "train_log.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"done: {step} steps, peak {summary['peak_gpu_gb']} GB, {summary['total_sec']} s -> {args.out}")


if __name__ == "__main__":
    main()

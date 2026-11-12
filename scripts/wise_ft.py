#!/usr/bin/env python3
"""WiSE-FT (Wortsman et al., 2022): blend frozen CLIP with a fine-tuned checkpoint in weight space.

theta_alpha = (1 - alpha) * theta_frozen + alpha * theta_finetuned, over EVERY trained tensor: the CLIP vision
encoder, CLIP's visual_projection and the 768x768 projection head (frozen side = identity). A checkpoint group
that was never trained (e.g. no vision weights in a projection-only run) stays frozen. Non-float buffers
(e.g. position ids) are copied, and must be equal on both sides.

Usage (writes a blended checkpoint usable by eval_a5.py / eval_checkpoint.py):
    python scripts/wise_ft.py --checkpoint E:/car_search_data/checkpoints/a5_B/model.pt --alpha 0.5 --out E:/car_search_data/checkpoints/a5_D_a050/model.pt
"""
import argparse
from pathlib import Path

GROUPS = ("vision_model", "visual_projection", "proj")


def frozen_state(model) -> dict:
    """Checkpoint-shaped state of frozen CLIP with the identity projection (CPU copies)."""
    import torch
    return {"vision_model": {k: v.detach().cpu().clone() for k, v in model.vision_model.state_dict().items()},
            "visual_projection": {k: v.detach().cpu().clone() for k, v in model.visual_projection.state_dict().items()},
            "proj": {"weight": torch.eye(768)}}


def blend(frozen: dict, tuned: dict, alpha: float) -> dict:
    """Linear interpolation of two checkpoint-shaped states. alpha=0 -> frozen, alpha=1 -> tuned."""
    assert 0.0 <= alpha <= 1.0, alpha
    out = {}
    for g in GROUPS:
        f, t = frozen[g], tuned.get(g)
        if t is None:  # group not trained in this checkpoint
            out[g] = {k: v.clone() for k, v in f.items()}
            continue
        assert f.keys() == t.keys(), f"{g}: keys differ ({len(f.keys() ^ t.keys())} mismatched)"
        out[g] = {}
        for k, fv in f.items():
            tv = t[k].to(fv.device)
            assert fv.shape == tv.shape, f"{g}.{k}: shape {tuple(fv.shape)} vs {tuple(tv.shape)}"
            if fv.is_floating_point():
                out[g][k] = ((1 - alpha) * fv.float() + alpha * tv.float()).to(fv.dtype)
            else:
                assert bool((fv == tv).all()), f"{g}.{k}: non-float tensor differs"
                out[g][k] = fv.clone()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    from transformers import CLIPVisionModelWithProjection
    frozen = frozen_state(CLIPVisionModelWithProjection.from_pretrained(a.model))
    tuned = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blend(frozen, tuned, a.alpha), a.out)
    print(f"alpha {a.alpha}: {a.checkpoint} -> {a.out}")


if __name__ == "__main__":
    main()

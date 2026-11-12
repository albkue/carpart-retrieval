import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))
from eval_a5 import abstain_threshold, type_metrics
from make_type_tune_split import carve
from train_embedding import epoch_batches, steps_per_epoch
from validate_type_split import check_split
from wise_ft import blend


def _state(v):
    return {"vision_model": {"w": torch.full((2, 2), float(v)), "pos": torch.arange(3)},
            "visual_projection": {"weight": torch.full((2,), float(v))},
            "proj": {"weight": torch.full((2, 2), float(v))}}


def test_blend_interpolates_every_group_and_keeps_buffers():
    f, t = _state(0), _state(4)
    assert torch.equal(blend(f, t, 0.0)["vision_model"]["w"], f["vision_model"]["w"])
    assert torch.equal(blend(f, t, 1.0)["proj"]["weight"], t["proj"]["weight"])
    mid = blend(f, t, 0.25)
    for g in ("vision_model", "visual_projection", "proj"):
        key = "w" if g == "vision_model" else "weight"
        assert torch.allclose(mid[g][key], torch.full_like(f[g][key], 1.0))
    assert torch.equal(mid["vision_model"]["pos"], torch.arange(3))


def test_blend_untrained_group_stays_frozen():
    f, t = _state(0), {"proj": {"weight": torch.full((2, 2), 4.0)}}  # projection-only checkpoint
    out = blend(f, t, 0.5)
    assert torch.equal(out["vision_model"]["w"], f["vision_model"]["w"])
    assert torch.allclose(out["proj"]["weight"], torch.full((2, 2), 2.0))


def test_type_metrics_ranks_types_by_best_image_and_image_level_ap():
    g_types = ["a", "a", "b"]
    sim = np.array([[0.9, 0.1, 0.5],    # query a: best a 0.9 > b 0.5 -> rank 1; AP over images: a@1, b@2, a@3 -> (1 + 2/3) / 2
                    [0.2, 0.3, 0.8]])   # query a: best b 0.8 > a 0.3 -> rank 2
    m = type_metrics(sim, g_types, ["a", "a"])
    assert m["recall@1"].tolist() == [1.0, 0.0] and m["recall@5"].tolist() == [1.0, 1.0]
    assert m["rr"].tolist() == [1.0, 0.5] and m["top1"] == ["a", "b"]
    assert abs(m["ap"][0] - (1 + 2 / 3) / 2) < 1e-9


def test_abstain_threshold_gives_requested_false_abstention():
    known = np.linspace(0.5, 1.0, 101)
    thr = abstain_threshold(known, 0.05)
    assert abs((known < thr).mean() - 0.05) < 0.02


def test_epoch_length_in_images_and_exact_batch_count():
    parts = {f"t{i}": [f"t{i}_{j}" for j in range(40)] for i in range(10)}  # 400 images
    assert steps_per_epoch(parts, 8, 4, "images") == 13 and steps_per_epoch(parts, 8, 4, "parts") == 1
    got = list(epoch_batches(parts, 8, 4, random.Random(0), 13))
    assert len(got) == 13 and all(len(paths) == 32 for paths, _ in got)


def test_check_split_catches_leak_heldout_and_missing_gallery():
    ok_train = {"catalog": [{"image": "a1", "part_id": "a"}], "heldout_part_ids": ["h"]}
    ok_eval = {"catalog": [{"image": "a2", "part_id": "a"}, {"image": "h1", "part_id": "h"}],
               "queries": {"valid": [{"image": "a3", "part_id": "a"}], "heldout": [{"image": "h2", "part_id": "h"}]}}
    assert check_split(ok_train, ok_eval) == []
    leak = {**ok_train, "catalog": ok_train["catalog"] + [{"image": "a2", "part_id": "a"}, {"image": "x", "part_id": "h"}]}
    probs = " ".join(check_split(leak, ok_eval))
    assert "both train and eval" in probs and "held-out types in train" in probs
    no_gal = {**ok_eval, "catalog": [ok_eval["catalog"][0]]}
    assert any("without gallery" in p for p in check_split(ok_train, no_gal))


def test_tune_carve_is_disjoint_and_per_type():
    rows = [{"image": f"{t}{i}", "part_id": t} for t in "ab" for i in range(50)]
    tr, ga, qu = carve(rows)
    assert len(tr) + len(ga) + len(qu) == 100
    assert not ({r["image"] for r in tr} & {r["image"] for r in ga + qu})
    assert {r["part_id"] for r in qu} == {"a", "b"} and len(qu) == 4 and len(ga) == 6


def test_below_line_flags_interior_dips_only():
    from report_a5 import below_line
    assert below_line([0, 0.5, 1], [0.8, 0.7, 0.9]) == [0.5]
    assert below_line([0, 0.25, 0.5, 1], [0.8, 0.85, 0.9, 1.0]) == []


def test_lpft_stage2_lr_is_tenth_of_ft_unless_given():
    from train_embedding import lpft_ft_lr
    assert abs(lpft_ft_lr(None, 3e-5) - 3e-6) < 1e-12
    assert abs(lpft_ft_lr(None, None) - 3e-6) < 1e-12
    assert lpft_ft_lr(1e-6, 3e-5) == 1e-6


def test_tuning_leak_counts_tuning_images_in_training_split(tmp_path):
    import json

    from eval_a5 import tuning_leak
    split = tmp_path / "train.json"
    split.write_text(json.dumps({"catalog": [{"image": "a", "part_id": "x"}, {"image": "b", "part_id": "x"}]}))
    (tmp_path / "train_log.json").write_text(json.dumps({"args": {"split": str(split)}}))
    ck = tmp_path / "model.pt"
    assert tuning_leak(ck, [{"image": "a"}, {"image": "c"}]) == 1
    assert tuning_leak(tmp_path / "none" / "model.pt", [{"image": "a"}]) == 0


def test_search_picks_highest_recall_then_map():
    from search_a5 import best_of
    rows = [{"id": 1, "tune_recall@1": 0.90, "tune_mAP": 0.80},
            {"id": 2, "tune_recall@1": 0.93, "tune_mAP": 0.70},
            {"id": 3, "tune_recall@1": 0.93, "tune_mAP": 0.75}]
    assert best_of(rows)["id"] == 3

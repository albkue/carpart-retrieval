"""Checks for scripts/train_embedding.py that need no GPU or model download."""
import importlib.util
import random
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "train_embedding", Path(__file__).resolve().parent.parent / "scripts" / "train_embedding.py")
te = importlib.util.module_from_spec(spec)
spec.loader.exec_module(te)

MANIFEST = {"catalog": [{"image": f"a/{i}.jpg", "part_id": pid} for pid in "ABCDEFGHIJ" for i in range(3)]
            + [{"image": "z/0.jpg", "part_id": "SOLO"}]}


def test_single_image_parts_dropped():
    assert "SOLO" not in te.parts_from_manifest(MANIFEST)


def test_heldout_part_in_training_set_is_rejected():
    with pytest.raises(AssertionError):
        te.parts_from_manifest({**MANIFEST, "heldout_part_ids": ["C"]})


def test_batches_are_p_by_k_with_distinct_parts():
    parts = te.parts_from_manifest(MANIFEST)
    for paths, labels in te.batches(parts, 4, 2, random.Random(0)):
        assert len(paths) == len(labels) == 8
        assert len(set(labels)) == 4 and all(labels.count(l) == 2 for l in set(labels))


def test_loss_is_lower_for_clustered_embeddings():
    labels = ["a", "a", "b", "b"]
    clustered = torch.nn.functional.normalize(torch.tensor([[1., 0], [1, .01], [0, 1], [.01, 1]]), dim=-1)
    scattered = torch.nn.functional.normalize(torch.tensor([[1., 0], [0, 1], [1, .01], [.01, 1]]), dim=-1)
    assert te.contrastive_loss(clustered, labels) < te.contrastive_loss(scattered, labels)

"""The experiment harness: metrics maths, config guards, and one full run
against an in-process Qdrant with a fake embedder."""
import json

import numpy as np
import pytest
import yaml

from scripts import run_experiment as rx

CONFIG = rx.REPO / "configs" / "baseline_mvb.yaml"


def test_rank_parts_pools_by_best_image():
    hits = [("a", 0.9), ("b", 0.8), ("a", 0.7), ("c", 0.6), ("b", 0.5)]
    assert rx.rank_parts(hits) == ["a", "b", "c"]


def test_score_query():
    row = rx.score_query(["a", "b", "c"], "b", [1, 5])
    assert row == {"rank": 2, "ap": 0.5, "recall@1": 0.0, "recall@5": 1.0}
    assert rx.score_query(["a"], "z", [1])["ap"] == 0.0


def test_bootstrap_ci_brackets_the_mean_and_is_seeded():
    values = np.array([1.0] * 70 + [0.0] * 30)
    lo, hi = rx.bootstrap_ci(values, 1000, 0.95, np.random.default_rng(42))
    assert lo < 0.7 < hi
    assert (lo, hi) == rx.bootstrap_ci(values, 1000, 0.95, np.random.default_rng(42))


def _base_config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path, value", [
    (("index", "exact"), False),
    (("data", "eval_split"), "test"),
    (("query", "region"), "crop_99"),
    (("query", "use_ocr"), True),
    (("query", "preprocess"), None),
])
def test_config_guards(path, value):
    cfg = _base_config()
    cfg[path[0]][path[1]] = value
    with pytest.raises(SystemExit):
        rx.check_config(cfg)


def test_crop_region_needs_a_detector():
    cfg = _base_config()
    cfg["query"]["region"] = "crop_15"
    rx.check_config(cfg)
    del cfg["query"]["detector"]
    with pytest.raises(SystemExit, match="detector"):
        rx.check_config(cfg)


def test_query_region_uses_the_service_crop(monkeypatch):
    from PIL import Image

    class FakeDetector:
        def __init__(self, bbox):
            self.bbox = bbox

        def detect(self, image):
            assert image.shape == (200, 400, 3)  # numpy RGB, as the service passes it
            return type("R", (), {"bbox": self.bbox})() if self.bbox else None

    cfg = _base_config()
    im = Image.new("RGB", (400, 200))
    assert rx.query_region(im, "full_image", cfg) == (im, None)
    assert rx.query_region(im, "centre_70", cfg)[0].size == (280, 140)

    monkeypatch.setattr(rx, "_detector", lambda model, conf: FakeDetector([150, 75, 250, 125]))
    crop, detected = rx.query_region(im, "crop_15", cfg)
    assert (crop.size, detected) == ((130, 130), True)
    assert rx.query_region(im, "crop_00", cfg)[0].size == (100, 100)

    # No box: falls back to centre_70, flagged so A1 can report the miss rate.
    monkeypatch.setattr(rx, "_detector", lambda model, conf: FakeDetector(None))
    crop, detected = rx.query_region(im, "crop_15", cfg)
    assert (crop.size, detected) == ((280, 140), False)


def test_config_hash_tracks_split_content(tmp_path, monkeypatch):
    monkeypatch.setattr(rx, "REPO", tmp_path)
    (tmp_path / "split.json").write_text("{}")
    first = rx.config_hash(rx.load_config(CONFIG, "split.json"))
    (tmp_path / "split.json").write_text('{"changed": 1}')
    assert rx.config_hash(rx.load_config(CONFIG, "split.json")) != first


def test_full_run_writes_immutable_results(tmp_path, monkeypatch):
    parts = ["p1", "p2", "p3"]
    split = {
        "toy": True,
        "catalog": [{"image": f"{p}_{i}.jpg", "part_id": p, "category": "c"} for p in parts for i in range(2)],
        "queries": {"valid": [{"image": f"{p}_q.jpg", "part_id": p} for p in parts]},
    }
    (tmp_path / "split.json").write_text(json.dumps(split))
    monkeypatch.setattr(rx, "REPO", tmp_path)

    def fake_embed(paths, cfg, preprocess=False, region="full_image", detected=None):
        # One direction per part plus small per-image noise: every query's
        # nearest part is its own.
        rng = np.random.default_rng(0)
        base = np.eye(cfg["embed"]["dim"])[: len(parts)]
        vecs = np.array([base[parts.index(p.split("_")[0])] for p in paths])
        vecs = vecs + rng.normal(0, 0.01, vecs.shape)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    monkeypatch.setattr(rx, "embed_images", fake_embed)
    cfg = rx.load_config(CONFIG, "split.json")
    out = rx.run(cfg, tmp_path / "results", ":memory:")

    metrics = json.loads((out / "metrics.json").read_text())
    assert out.name == rx.config_hash(cfg)
    assert metrics["toy"] is True
    assert metrics["metrics"]["recall@1"]["value"] == 1.0
    assert metrics["metrics"]["mAP"]["value"] == 1.0
    for name in ("per_query.csv", "config.resolved.yaml", "git_sha", "env.json"):
        assert (out / name).exists()

    with pytest.raises(SystemExit, match="never overwritten"):
        rx.run(rx.load_config(CONFIG, "split.json"), tmp_path / "results", ":memory:")


def test_unverified_catalogue_entry_is_refused(tmp_path, monkeypatch):
    split = {"catalog": [{"image": "a.jpg", "part_id": "a", "verified": False}],
             "queries": {"valid": [{"image": "q.jpg", "part_id": "a"}]}}
    (tmp_path / "split.json").write_text(json.dumps(split))
    monkeypatch.setattr(rx, "REPO", tmp_path)
    with pytest.raises(SystemExit, match="verified=false"):
        rx.load_split(rx.load_config(CONFIG, "split.json"))


def test_preprocess_applies_the_service_preprocessor(tmp_path, monkeypatch):
    from PIL import Image

    Image.new("RGB", (300, 200), "white").save(tmp_path / "q.jpg")
    monkeypatch.setattr(rx, "REPO", tmp_path)
    seen = []

    class FakeEmbedder:
        def encode_images(self, images):
            seen.extend(im.size for im in images)
            return np.zeros((len(images), 768))

    monkeypatch.setattr(rx, "_embedder", lambda model: FakeEmbedder())
    cfg = _base_config()
    rx.embed_images(["q.jpg"], cfg)
    rx.embed_images(["q.jpg"], cfg, preprocess=True)
    # Raw image passes through; preprocessed one is the service's 640x640 letterbox.
    assert seen == [(300, 200), (640, 640)]


def test_crop_run_records_detection(tmp_path, monkeypatch):
    split = {"catalog": [{"image": f"{p}.jpg", "part_id": p} for p in ("a", "b")],
             "queries": {"valid": [{"image": f"{p}_q.jpg", "part_id": p} for p in ("a", "b")]}}
    (tmp_path / "split.json").write_text(json.dumps(split))
    monkeypatch.setattr(rx, "REPO", tmp_path)

    def fake_embed(paths, cfg, preprocess=False, region="full_image", detected=None):
        if detected is not None:
            detected.extend([True, False][: len(paths)])
        vecs = np.eye(cfg["embed"]["dim"])[[0 if p.startswith("a") else 1 for p in paths]]
        return vecs.astype(float)

    monkeypatch.setattr(rx, "embed_images", fake_embed)
    cfg = rx.load_config(CONFIG, "split.json")
    cfg["query"]["region"] = "crop_15"
    out = rx.run(cfg, tmp_path / "results", ":memory:")

    metrics = json.loads((out / "metrics.json").read_text())
    assert (metrics["region"], metrics["detection_rate"]) == ("crop_15", 0.5)
    assert "detected" in (out / "per_query.csv").read_text().splitlines()[0]

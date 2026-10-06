#!/usr/bin/env python3
"""Run one experiment: config -> embed -> index -> query -> metrics.

The only way a number reaches the thesis (README rule 5). Reads everything
that affects the result from the config, including the seed, and writes an
immutable results directory keyed by the hash of the resolved config:

    results/<config-hash>/
        metrics.json          Recall@K and mAP, each with a bootstrap CI
        per_query.csv         every query's rank, for post-hoc analysis
        config.resolved.yaml  the exact configuration, split checksum included
        git_sha               the code that produced it
        env.json              Python, torch, CUDA, Qdrant versions

A directory that already exists is never overwritten; change the config, or
delete the directory by hand if the run it holds is known to be wrong.

The split manifest is JSON, image paths relative to the repo root:

    {"catalog": [{"image": "...", "part_id": "...", "category": ..., ...}],
     "queries": {"valid": [{"image": "...", "part_id": "..."}], ...}}

``part_id`` is the equivalence-group id (DATASET_SPEC §4): cross-referenced
part numbers are merged before the manifest is written, so a query has exactly
one correct part. Extra catalogue fields become Qdrant payload.

Usage:
    docker compose up -d qdrant
    python scripts/run_experiment.py --config configs/baseline_mvb.yaml
    python scripts/run_experiment.py --config configs/baseline_mvb.yaml --split data/processed/splits_toy.json
"""
import argparse
import csv
import functools
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from search.qdrant_index import QdrantIndex

REPO = Path(__file__).resolve().parent.parent

# A1 query regions. crop_NN = YOLO box with NN% padding, centre_70 when YOLO
# finds nothing (the service's rule); crop_15 is the deployed path.
CROP_PADDING = {"crop_00": 0.0, "crop_10": 0.10, "crop_15": 0.15, "crop_20": 0.20}
REGIONS = ("full_image", "centre_70", *CROP_PADDING)

logger = logging.getLogger("run_experiment")


# --- config ------------------------------------------------------------------

def load_config(path: Path, split: str | None = None) -> dict:
    """Read a config, apply the --split override, pin the split's checksum.

    The checksum goes into the resolved config, so editing the manifest in
    place changes the hash exactly like editing the config does.
    """
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if split:
        cfg["data"]["split_manifest"] = split
    manifest = REPO / cfg["data"]["split_manifest"]
    if not manifest.exists():
        raise SystemExit(f"Split manifest not found: {manifest}")
    cfg["data"]["split_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def check_config(cfg: dict):
    """Refuse configs the harness would silently mis-measure."""
    if cfg["data"]["eval_split"] == "test":
        # RESEARCH_PROTOCOL §7: read once, in April 2027. Remove this guard
        # deliberately, in its own commit, on that day.
        raise SystemExit("eval_split 'test' is sealed until April 2027")
    if cfg["index"]["backend"] != "qdrant":
        raise SystemExit(f"Unsupported index backend: {cfg['index']['backend']}")
    if cfg["index"].get("exact") is not True:
        # An ANN run understates Recall@K without any error. A8 lifts this.
        raise SystemExit("index.exact must be true for research runs")
    if cfg["index"]["metric"] != "cosine":
        raise SystemExit(f"Unsupported metric: {cfg['index']['metric']}")
    if cfg["embed"]["backend"] != "service":
        raise SystemExit(f"Unsupported embed backend: {cfg['embed']['backend']}")
    if cfg["eval"]["pool_images_to_part"] != "max":
        raise SystemExit("Only pool_images_to_part: max is implemented")
    query = cfg["query"]
    if query["region"] not in REGIONS:
        raise SystemExit(f"query.region must be one of {REGIONS}, got {query['region']}")
    if query["region"] in CROP_PADDING and not query.get("detector", {}).get("model"):
        raise SystemExit(f"query.region {query['region']} needs query.detector.model")
    if not isinstance(query.get("preprocess"), bool):
        raise SystemExit("query.preprocess must be set to true or false")
    for flag in ("use_ocr", "use_metadata", "use_fusion"):
        if query.get(flag):
            raise SystemExit(f"query.{flag} is not implemented in the harness yet")


# --- data --------------------------------------------------------------------

def load_split(cfg: dict) -> tuple[list[dict], list[dict], bool]:
    manifest = json.loads((REPO / cfg["data"]["split_manifest"]).read_text(encoding="utf-8"))
    catalog = manifest["catalog"]
    queries = manifest["queries"][cfg["data"]["eval_split"]]
    unverified = [e["image"] for e in catalog if e.get("verified") is False]
    if unverified:
        # README rule 1: nothing enters the index with verified = false.
        raise SystemExit(f"{len(unverified)} catalogue entries have verified=false, e.g. {unverified[0]}")
    if not catalog or not queries:
        raise SystemExit("Split has an empty catalogue or query set")
    known = {e["part_id"] for e in catalog}
    orphans = [q["image"] for q in queries if q["part_id"] not in known]
    if orphans:
        raise SystemExit(f"{len(orphans)} queries have no catalogue part, e.g. {orphans[0]}")
    return catalog, queries, bool(manifest.get("toy"))


@functools.lru_cache(maxsize=1)
def _embedder(model: str):
    import torch

    from pipeline.embedding import CLIPEmbedding
    return CLIPEmbedding(model, use_gpu=torch.cuda.is_available())


@functools.lru_cache(maxsize=1)
def _detector(model: str, confidence: float):
    import torch

    from pipeline.yolo_detector import YOLOPartDetector
    return YOLOPartDetector(model, confidence, torch.cuda.is_available())


def query_region(image, region: str, cfg: dict):
    """Apply an A1 region with the service's crop code. Returns (image, detected).

    ``detected`` is None for regions that don't run YOLO.
    """
    from pipeline.query_crop import centre_crop, crop_for_embedding

    if region == "full_image":
        return image, None
    if region == "centre_70":
        return centre_crop(image), None
    det_cfg = cfg["query"]["detector"]
    result = _detector(det_cfg["model"], det_cfg["confidence"]).detect(np.array(image))
    bbox = result.bbox if result else None
    return crop_for_embedding(image, bbox, CROP_PADDING[region]), bbox is not None


def embed_images(paths: list[str], cfg: dict, preprocess: bool = False,
                 region: str = "full_image", detected: list | None = None) -> np.ndarray:
    """Embed with the service's own CLIPEmbedding (ADR 001): evaluated == deployed.

    ``preprocess`` runs the service's AdaptivePreprocessor first, then
    ``region`` crops, in the order /search/image does for queries. Catalogue
    images are indexed raw and whole by the service, so the caller passes both
    for queries only. Per-image YOLO outcomes are appended to ``detected``.
    """
    from PIL import Image

    embedder = _embedder(cfg["embed"]["model"])
    if preprocess:
        from pipeline.adaptive_preprocessor import AdaptivePreprocessor
        preprocessor = AdaptivePreprocessor()
    batch = cfg["embed"]["batch_size"]
    out = []
    for start in range(0, len(paths), batch):
        images = [Image.open(REPO / p).convert("RGB") for p in paths[start:start + batch]]
        if preprocess:
            images = [Image.fromarray(preprocessor.preprocess(im)) for im in images]
        if region != "full_image":
            cropped = [query_region(im, region, cfg) for im in images]
            images = [im for im, _ in cropped]
            if detected is not None:
                detected.extend(d for _, d in cropped)
        out.append(embedder.encode_images(images))
        logger.info(f"embedded {min(start + batch, len(paths))}/{len(paths)}")
    return np.concatenate(out)


# --- metrics -----------------------------------------------------------------

def rank_parts(hits: list[tuple]) -> list:
    """Point hits (best first) -> parts ranked by their best image (max pooling)."""
    return list(dict.fromkeys(part for part, _ in hits))


def score_query(ranked: list, truth, ks: list[int]) -> dict:
    rank = ranked.index(truth) + 1 if truth in ranked else None
    row = {"rank": rank, "ap": 1.0 / rank if rank else 0.0}
    # One relevant part per query, so AP is the reciprocal rank.
    for k in ks:
        row[f"recall@{k}"] = float(rank is not None and rank <= k)
    return row


def bootstrap_ci(values: np.ndarray, resamples: int, ci: float, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile bootstrap over queries, the independent unit (RESEARCH_PROTOCOL §6)."""
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    tail = (1 - ci) / 2 * 100
    return float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


# --- provenance --------------------------------------------------------------

def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    if dirty:
        logger.warning("Working tree is dirty: this result is not attributable to a commit")
        return sha + "-dirty"
    return sha


def environment(index: QdrantIndex) -> dict:
    import torch

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "transformers": importlib.metadata.version("transformers"),
        "qdrant_client": importlib.metadata.version("qdrant-client"),
    }
    try:
        env["qdrant_server"] = index.client.info().version
    except Exception:
        env["qdrant_server"] = "in-process"
    return env


# --- the run -----------------------------------------------------------------

def run(cfg: dict, results_root: Path, qdrant_url: str) -> Path:
    check_config(cfg)
    run_hash = config_hash(cfg)
    out = results_root / run_hash
    if out.exists():
        raise SystemExit(f"{out} already exists; results are never overwritten")

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    catalog, queries, toy = load_split(cfg)
    logger.info(f"{len(catalog)} catalogue images, {len(queries)} queries ({cfg['data']['eval_split']})")

    cat_emb = embed_images([e["image"] for e in catalog], cfg)
    region = cfg["query"]["region"]
    detected = []
    query_emb = embed_images([q["image"] for q in queries], cfg, preprocess=cfg["query"]["preprocess"],
                             region=region, detected=detected)
    for name, emb in (("catalogue", cat_emb), ("query", query_emb)):
        if emb.shape[1] != cfg["embed"]["dim"]:
            raise SystemExit(f"{name} embeddings are {emb.shape[1]}d, config says {cfg['embed']['dim']}")
        if cfg["embed"]["normalize"] and not np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3):
            raise SystemExit(f"{name} embeddings are not L2-normalised")

    hnsw = cfg["index"]["hnsw"]
    index = QdrantIndex(
        dimension=cfg["embed"]["dim"],
        collection=f"{cfg['index']['collection']}_{run_hash}",
        url=qdrant_url,
        metric="cosine",
        hnsw_m=hnsw["m"],
        hnsw_ef_construct=hnsw["ef_construct"],
        hnsw_ef_search=hnsw["ef_search"],
    )
    index.clear()
    index.add_embeddings(
        cat_emb,
        [e["part_id"] for e in catalog],
        [{k: v for k, v in e.items() if k != "part_id"} for e in catalog],
    )

    ks = cfg["eval"]["k"]
    rows = []
    # The whole catalogue per query, so every relevant part gets a rank and
    # mAP is exact rather than truncated.
    for start in range(0, len(queries), 64):
        batch = index.search_batch(query_emb[start:start + 64], k=len(catalog), exact=True)
        for i, (q, hits) in enumerate(zip(queries[start:start + 64], batch), start):
            ranked = rank_parts(hits)
            row = {"image": q["image"], "part_id": q["part_id"],
                   "top1": ranked[0] if ranked else None, **score_query(ranked, q["part_id"], ks)}
            if region in CROP_PADDING:
                # False = YOLO found nothing and the query fell back to centre_70.
                row["detected"] = detected[i]
            rows.append(row)
    index.client.delete_collection(index.collection)

    rng = np.random.default_rng(seed)
    metrics = {}
    for key in [f"recall@{k}" for k in ks] + ["ap"]:
        values = np.array([r[key] for r in rows])
        lo, hi = bootstrap_ci(values, cfg["eval"]["bootstrap_resamples"], cfg["eval"]["ci"], rng)
        metrics["mAP" if key == "ap" else key] = {"value": float(values.mean()), "ci_low": lo, "ci_high": hi}

    sha = git_sha()
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps({
        "name": cfg["name"],
        "config_hash": run_hash,
        "git_sha": sha,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "toy": toy,
        "eval_split": cfg["data"]["eval_split"],
        "n_queries": len(queries),
        "n_catalog_images": len(catalog),
        "n_catalog_parts": len({e["part_id"] for e in catalog}),
        "region": region,
        "detection_rate": float(np.mean(detected)) if region in CROP_PADDING else None,
        "metrics": metrics,
    }, indent=2) + "\n", encoding="utf-8")
    with open(out / "per_query.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (out / "git_sha").write_text(sha + "\n", encoding="utf-8")
    (out / "env.json").write_text(json.dumps(environment(index), indent=2) + "\n", encoding="utf-8")
    logger.info(f"wrote {out}: " + ", ".join(f"{k}={v['value']:.3f}" for k, v in metrics.items()))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", help="override data.split_manifest (becomes part of the hash)")
    parser.add_argument("--results", type=Path, default=REPO / "results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, args.split)
    run(cfg, args.results, os.environ.get("QDRANT_URL", "http://localhost:6333"))


if __name__ == "__main__":
    main()

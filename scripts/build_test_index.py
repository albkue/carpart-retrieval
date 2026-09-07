#!/usr/bin/env python
"""
Build a test FAISS index from local dataset images.
This lets you test the /search-by-image endpoint without the catalog API.
"""
import os
# Fix OpenMP runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from pipeline.embedding import CLIPEmbedding
from search.faiss_index import FAISSIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_test_index(image_dir: str = "ml_datasets/images/test", max_images: int = 50):
    """Index local images as mock products for testing."""
    image_path = Path(image_dir)
    if not image_path.exists():
        logger.error(f"Directory not found: {image_dir}")
        return

    image_files = list(image_path.glob("*.jpg"))[:max_images]
    if not image_files:
        logger.error(f"No .jpg images found in {image_dir}")
        return

    logger.info(f"Indexing {len(image_files)} images from {image_dir}...")

    embedder = CLIPEmbedding(settings.CLIP_MODEL, use_gpu=False)
    index = FAISSIndex(
        dimension=settings.EMBEDDING_DIMENSION,
        index_path=str(Path(settings.FAISS_INDEX_PATH) / "image"),
        index_type="hnsw",
        metric="l2",
    )
    index.create_index()

    embeddings = []
    product_ids = []
    metadata = {}

    for i, img_path in enumerate(image_files):
        try:
            img = Image.open(img_path).convert("RGB")
            emb = embedder.encode_image(img)
            embeddings.append(emb)

            pid = i
            product_ids.append(pid)
            metadata[pid] = {"name": img_path.name, "category": "test", "price": 0}

            logger.info(f"[{i+1}/{len(image_files)}] Indexed: {img_path.name}")
        except Exception as e:
            logger.warning(f"Failed to index {img_path.name}: {e}")

    if embeddings:
        index.add_embeddings(np.array(embeddings), product_ids)
        index.product_metadata = metadata
        index.save_index()
        logger.info(f"Test index saved! {len(embeddings)} images indexed.")
    else:
        logger.warning("No images were indexed.")


if __name__ == "__main__":
    build_test_index()

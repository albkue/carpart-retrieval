#!/usr/bin/env python
"""
Hybrid Product Index Builder (Image + Text)

This script builds dual FAISS indices for the Hybrid Search Pipeline:
1. Image Index (CLIP ViT-L/14, 768-dim)
2. Text Index (BGE-M3, 1024-dim)

Usage:
    python scripts/build_product_index.py
"""

import sys
from pathlib import Path
import logging
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from pipeline.embedding import CLIPEmbedding
from pipeline.text_embedding import TextEmbedding
from search.faiss_index import FAISSIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridIndexBuilder:
    def __init__(self):
        # Paths
        self.image_index_path = Path(settings.FAISS_INDEX_PATH) / "image"
        self.text_index_path = Path(settings.FAISS_INDEX_PATH) / "text"
        
        # Initialize Models
        logger.info("Initializing ML Models (this may take a while)...")
        self.clip_embedder = CLIPEmbedding(settings.CLIP_MODEL, settings.USE_GPU)
        self.text_embedder = TextEmbedding(settings.TEXT_MODEL, settings.USE_GPU)
        
        # Initialize Indices
        self.image_index = FAISSIndex(
            dimension=settings.EMBEDDING_DIMENSION,
            index_path=str(self.image_index_path),
            index_type="hnsw",
            metric="l2"
        )
        self.text_index = FAISSIndex(
            dimension=settings.TEXT_EMBEDDING_DIMENSION,
            index_path=str(self.text_index_path),
            index_type="hnsw",
            metric="inner_product"
        )

    def fetch_products(self):
        """Fetch products from the main API."""
        logger.info(f"Fetching products from {settings.MAIN_API_URL}...")
        try:
            # Note: Update this URL to match your main API's products endpoint
            response = requests.get(f"{settings.MAIN_API_URL}/api/v1/products", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch products: {e}")
            return []

    def download_image(self, url):
        try:
            resp = requests.get(url, timeout=10)
            return Image.open(BytesIO(resp.content)).convert('RGB')
        except Exception:
            return None

    def build(self):
        products = self.fetch_products()
        if not products:
            logger.warning("No products to index. Check if your main API is running.")
            return

        image_embeddings = []
        text_embeddings = []
        product_ids = []
        metadata = {}

        for i, p in enumerate(products):
            pid = p['product_id']
            name = p.get('name', '')
            desc = p.get('description', '')
            img_url = p.get('image_url')

            logger.info(f"[{i+1}/{len(products)}] Processing Product ID: {pid}")

            # 1. Image Embedding
            image = self.download_image(img_url) if img_url else None
            if image:
                try:
                    img_emb = self.clip_embedder.encode_image(image)
                    image_embeddings.append(img_emb)
                    
                    # 2. Text Embedding (Name + Description)
                    text_input = f"{name} {desc}".strip()
                    txt_emb = self.text_embedder.encode_text(text_input)
                    text_embeddings.append(txt_emb)
                    
                    product_ids.append(pid)
                    metadata[pid] = {
                        "name": name,
                        "category": p.get("category_name"),
                        "price": p.get("selling_price")
                    }
                except Exception as e:
                    logger.error(f"Error embedding product {pid}: {e}")

        # Save Image Index
        if image_embeddings:
            logger.info("Saving Image FAISS Index...")
            self.image_index.add_embeddings(np.array(image_embeddings), product_ids)
            self.image_index.product_metadata = metadata
            self.image_index.save_index()

        # Save Text Index
        if text_embeddings:
            logger.info("Saving Text FAISS Index...")
            self.text_index.add_embeddings(np.array(text_embeddings), product_ids)
            self.text_index.product_metadata = metadata
            self.text_index.save_index()

        logger.info("Hybrid Indexing Complete!")

if __name__ == "__main__":
    builder = HybridIndexBuilder()
    builder.build()

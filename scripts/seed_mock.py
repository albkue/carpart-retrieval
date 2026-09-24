import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from pipeline.text_embedding import TextEmbedding
from search.qdrant_index import QdrantIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def seed_mock_data():
    # 1. Define Sample Products
    products = [
        {
            "id": 101,
            "name": "Bosch QuietCast Premium Brake Pad",
            "desc": "Ceramic brake pads for European vehicles. Part number BP1234.",
            "brand": "Bosch",
            "category": "brake"
        },
        {
            "id": 102,
            "name": "NGK Iridium IX Spark Plug",
            "desc": "High performance spark plug for improved ignition. Model 6619 BKR6EIX.",
            "brand": "NGK",
            "category": "spark_plug"
        },
        {
            "id": 103,
            "name": "Mann-Filter Oil Filter",
            "desc": "High quality oil filter for engine protection. Fits most German cars. Part W712/80.",
            "brand": "Mann",
            "category": "filter"
        }
    ]

    # 2. Initialize Text Embedder (Real BGE-M3)
    logger.info("Initializing BGE-M3 for realistic text seeding...")
    text_model = TextEmbedding(settings.TEXT_MODEL, use_gpu=False)
    
    product_ids = [p['id'] for p in products]
    text_inputs = [f"{p['name']} {p['desc']}" for p in products]
    
    logger.info("Generating text embeddings...")
    text_embeddings = text_model.encode_batch(text_inputs)

    # 3. Generate Mock Image Embeddings (Random for now)
    logger.info("Generating mock image embeddings (768-dim)...")
    image_embeddings = np.random.randn(len(products), 768).astype('float32')
    image_embeddings /= np.linalg.norm(image_embeddings, axis=1, keepdims=True)

    payloads = [{"name": p['name'], "brand": p['brand'], "category": p['category']} for p in products]

    # 4. Write Text Collection
    logger.info("Writing text collection...")
    txt_idx = QdrantIndex(dimension=1024, collection=settings.QDRANT_TEXT_COLLECTION,
                          url=settings.QDRANT_URL, metric="inner_product")
    txt_idx.clear()
    txt_idx.add_embeddings(text_embeddings, product_ids, payloads)

    # 5. Write Image Collection
    logger.info("Writing image collection...")
    img_idx = QdrantIndex(dimension=768, collection=settings.QDRANT_IMAGE_COLLECTION,
                          url=settings.QDRANT_URL, metric="l2")
    img_idx.clear()
    img_idx.add_embeddings(image_embeddings, product_ids, payloads)

    logger.info("="*50)
    logger.info("✅ SUCCESS: Mock data seeded!")
    logger.info(f"Products added: {len(products)}")
    logger.info("You can now test search with terms like 'Bosch', 'Spark Plug', or 'W712'")
    logger.info("="*50)

if __name__ == "__main__":
    seed_mock_data()

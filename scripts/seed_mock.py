import sys
from pathlib import Path
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.text_embedding import TextEmbedding
from search.faiss_index import FAISSIndex
from app.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def seed_mock_data():
    # 1. Setup Directories
    image_dir = Path(settings.FAISS_INDEX_PATH) / "image"
    text_dir = Path(settings.FAISS_INDEX_PATH) / "text"
    image_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    # 2. Define Sample Products
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

    # 3. Initialize Text Embedder (Real BGE-M3)
    logger.info("Initializing BGE-M3 for realistic text seeding...")
    text_model = TextEmbedding(settings.TEXT_MODEL, use_gpu=False)
    
    product_ids = [p['id'] for p in products]
    text_inputs = [f"{p['name']} {p['desc']}" for p in products]
    
    logger.info("Generating text embeddings...")
    text_embeddings = text_model.encode_batch(text_inputs)

    # 4. Generate Mock Image Embeddings (Random for now)
    logger.info("Generating mock image embeddings (768-dim)...")
    image_embeddings = np.random.randn(len(products), 768).astype('float32')

    # 5. Create and Save Text Index
    logger.info("Saving Text Index...")
    txt_idx = FAISSIndex(dimension=1024, index_path=str(text_dir), metric="inner_product")
    txt_idx.create_index()
    txt_idx.add_embeddings(text_embeddings, product_ids)
    
    # Add metadata
    metadata = {p['id']: {"name": p['name'], "brand": p['brand'], "category": p['category']} for p in products}
    txt_idx.product_metadata = metadata
    txt_idx.save_index()

    # 6. Create and Save Image Index
    logger.info("Saving Image Index...")
    img_idx = FAISSIndex(dimension=768, index_path=str(image_dir), metric="l2")
    img_idx.create_index()
    img_idx.add_embeddings(image_embeddings, product_ids)
    img_idx.product_metadata = metadata
    img_idx.save_index()

    logger.info("="*50)
    logger.info("✅ SUCCESS: Mock data seeded!")
    logger.info(f"Products added: {len(products)}")
    logger.info("You can now test search with terms like 'Bosch', 'Spark Plug', or 'W712'")
    logger.info("="*50)

if __name__ == "__main__":
    seed_mock_data()

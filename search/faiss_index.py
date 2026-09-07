"""FAISS Vector Index Module.

This module provides FAISS-based vector similarity search for image embeddings.
"""
import faiss
import numpy as np
from typing import List, Tuple, Optional
import json
import os
import logging

logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS-based vector index for image and text embeddings."""
    
    def __init__(
        self,
        dimension: int = 512,
        index_path: str = "./data/faiss_index",
        index_type: str = "hnsw",
        metric: str = "l2"
    ):
        """Initialize the FAISS index.
        
        Args:
            dimension: Dimension of the embedding vectors
            index_path: Path to save/load the index
            index_type: Type of index ('flat', 'ivf', 'hnsw')
            metric: Distance metric ('l2' or 'inner_product')
        """
        self.dimension = dimension
        self.index_path = index_path
        self.index_type = index_type
        self.metric = metric
        self.index: Optional[faiss.Index] = None
        self.id_to_product: dict = {}
        self.product_metadata: dict = {}
        self.is_trained = False
        
        logger.info(f"Initializing FAISS index (dim={dimension}, type={index_type}, metric={metric})")
    
    def create_index(self):
        """Create a new FAISS index based on configuration."""
        # Select metric
        faiss_metric = faiss.METRIC_L2
        if self.metric == "inner_product":
            faiss_metric = faiss.METRIC_INNER_PRODUCT

        if self.index_type == "flat":
            if self.metric == "inner_product":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
            
        elif self.index_type == "hnsw":
            # HNSW is high-performance approximate nearest neighbor search
            self.index = faiss.IndexHNSWFlat(self.dimension, 32, faiss_metric)
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 16
            
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.dimension)
            if self.metric == "inner_product":
                quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100, faiss_metric)
            
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        logger.info(f"Created new {self.index_type} FAISS index with {self.metric}")
    
    def load_index(self) -> bool:
        """Load existing index from disk.
            
        Checks multiple possible index locations:
        1. self.index_path (default)
        2. ./data/product_index (from build_product_index.py)
            
        Returns:
            True if index loaded successfully, False otherwise
        """
        # Try multiple index locations
        possible_paths = [
            self.index_path,
            "./data/product_index",
            "./data/faiss_index"
        ]
            
        for path in possible_paths:
            index_file = f"{path}/product_embeddings.faiss"
            ids_file = f"{path}/product_ids.json"
            meta_file = f"{path}/product_metadata.json"
                
            # Also check old format
            old_index_file = f"{path}/index.faiss"
            old_meta_file = f"{path}/metadata.json"
                
            # Try new format first (from build_product_index.py)
            if os.path.exists(index_file) and os.path.exists(ids_file):
                try:
                    logger.info(f"Loading product index from {path}...")
                    self.index = faiss.read_index(index_file)
                        
                    with open(ids_file, 'r') as f:
                        product_ids = json.load(f)
                        
                    # Build id_to_product mapping (index -> product_id)
                    self.id_to_product = {i: pid for i, pid in enumerate(product_ids)}
                        
                    # Load metadata if available
                    if os.path.exists(meta_file):
                        with open(meta_file, 'r') as f:
                            self.product_metadata = json.load(f)
                    else:
                        self.product_metadata = {}
                        
                    self.is_trained = True
                    logger.info(f"Loaded index with {self.index.ntotal} products")
                    return True
                        
                except Exception as e:
                    logger.error(f"Error loading index from {path}: {e}")
                    continue
                
            # Try old format
            elif os.path.exists(old_index_file) and os.path.exists(old_meta_file):
                try:
                    logger.info(f"Loading legacy index from {path}...")
                    self.index = faiss.read_index(old_index_file)
                        
                    with open(old_meta_file, 'r') as f:
                        self.id_to_product = json.load(f)
                        self.id_to_product = {int(k): v for k, v in self.id_to_product.items()}
                        
                    self.product_metadata = {}
                    self.is_trained = True
                    logger.info(f"Loaded legacy index with {self.index.ntotal} vectors")
                    return True
                        
                except Exception as e:
                    logger.error(f"Error loading legacy index: {e}")
                    continue
            
        logger.info("No existing index found. Will need to build one.")
        return False
    
    def save_index(self):
        """Save index to disk."""
        os.makedirs(self.index_path, exist_ok=True)
        
        try:
            faiss.write_index(self.index, f"{self.index_path}/index.faiss")
            
            with open(f"{self.index_path}/metadata.json", 'w') as f:
                json.dump(self.id_to_product, f)
            
            logger.info(f"Saved index to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
    
    def train(self, embeddings: np.ndarray):
        """Train the index (required for IVF).
        
        Args:
            embeddings: Training embeddings (n_vectors x dimension)
        """
        if self.index is None:
            self.create_index()
        
        if isinstance(self.index, faiss.IndexIVFFlat):
            logger.info(f"Training IVF index with {len(embeddings)} vectors...")
            self.index.train(embeddings.astype('float32'))
            self.is_trained = True
            logger.info("Index trained successfully")
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        product_ids: List[int]
    ) -> int:
        """Add embeddings to index.
        
        Args:
            embeddings: Embedding vectors (n_vectors x dimension)
            product_ids: Corresponding product IDs
            
        Returns:
            Number of embeddings added
        """
        if self.index is None:
            self.create_index()
        
        # Ensure float32
        embeddings = embeddings.astype('float32')
        
        # Train if needed (for IVF)
        if isinstance(self.index, faiss.IndexIVFFlat):
            if not self.index.is_trained:
                logger.warning("IVF index not trained. Training now...")
                self.index.train(embeddings)
        
        start_id = self.index.ntotal
        self.index.add(embeddings)
        
        # Map index IDs to product IDs
        for i, product_id in enumerate(product_ids):
            self.id_to_product[start_id + i] = product_id
        
        added = len(product_ids)
        logger.info(f"Added {added} embeddings to index (total: {self.index.ntotal})")
        
        return added
    
    def _distance_to_similarity(self, distance: float) -> float:
        """Convert FAISS distance to similarity score based on metric type.

        For L2: smaller distance = more similar → map to (0, 1] via 1/(1+dist).
        For inner_product: higher value = more similar → map [-1, 1] to [0, 1].

        Args:
            distance: Raw distance returned by FAISS.

        Returns:
            Similarity score in [0, 1] range.
        """
        if self.metric == "inner_product":
            # Normalized embeddings: IP == cosine similarity in [-1, 1]
            return float(max(0.0, (distance + 1.0) / 2.0))
        # L2 default
        return float(1.0 / (1.0 + distance))

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (product_id, similarity) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty, cannot search")
            return []
        
        # Ensure query is 2D float32
        query = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query, k)
        
        # Convert to results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx in self.id_to_product:
                product_id = self.id_to_product[idx]
                similarity = self._distance_to_similarity(dist)
                results.append((product_id, float(similarity)))
        
        logger.debug(f"Found {len(results)} results")
        return results
    
    def search_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 10
    ) -> List[List[Tuple[int, float]]]:
        """Search for similar embeddings in batch.
        
        Args:
            query_embeddings: Query embedding matrix (n_queries x dimension)
            k: Number of results per query
            
        Returns:
            List of result lists
        """
        if self.index is None or self.index.ntotal == 0:
            return [[] for _ in range(len(query_embeddings))]
        
        query = query_embeddings.astype('float32')
        distances, indices = self.index.search(query, k)
        
        all_results = []
        for i in range(len(query_embeddings)):
            results = []
            for dist, idx in zip(distances[i], indices[i]):
                if idx >= 0 and idx in self.id_to_product:
                    product_id = self.id_to_product[idx]
                    similarity = self._distance_to_similarity(dist)
                    results.append((product_id, float(similarity)))
            all_results.append(results)
        
        return all_results
    
    def remove_product(self, product_id: int) -> bool:
        """Remove a product from the index.
        
        Note: FAISS doesn't support direct removal, so this removes
        the mapping. For full removal, rebuild the index.
        
        Args:
            product_id: Product ID to remove
            
        Returns:
            True if removed, False if not found
        """
        # Find and remove from mapping
        keys_to_remove = [
            k for k, v in self.id_to_product.items()
            if v == product_id
        ]
        
        for key in keys_to_remove:
            del self.id_to_product[key]
        
        if keys_to_remove:
            logger.info(f"Removed product {product_id} from mapping")
            return True
        
        return False
    
    def search_with_metadata(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[dict]:
        """Search for similar products with full metadata.
            
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
                
        Returns:
            List of product dictionaries with similarity scores
        """
        results = self.search(query_embedding, k)
            
        products = []
        for product_id, similarity in results:
            product_info = {
                'product_id': product_id,
                'similarity': similarity
            }
                
            # Add metadata if available
            if str(product_id) in self.product_metadata:
                product_info.update(self.product_metadata[str(product_id)])
                
            products.append(product_info)
            
            
        return products
    
    def get_stats(self) -> dict:
        """Get index statistics.
            
        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {
                "status": "empty",
                "total_vectors": 0,
                "unique_products": 0
            }
    
        return {
            "status": "ready",
            "total_vectors": self.index.ntotal,
            "unique_products": len(set(self.id_to_product.values())),
            "dimension": self.dimension,
            "index_type": self.index_type,
            "products_with_metadata": len(self.product_metadata)
        }
    
    def clear(self):
        """Clear the index."""
        self.index = None
        self.id_to_product = {}
        self.is_trained = False
        logger.info("Index cleared")

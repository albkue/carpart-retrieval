"""BGE-M3 Text Embedding Module.

This module provides BGE-M3 based text embedding generation for semantic search.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import logging
import torch

logger = logging.getLogger(__name__)


class TextEmbedding:
    """BGE-M3 based text embedding generator."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_gpu: bool = False
    ):
        """Initialize the BGE-M3 text embedding generator.
        
        Args:
            model_name: Name of the BGE-M3 model to use
            use_gpu: Whether to use GPU for inference
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self._model = None
        self._device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing Text embedder (model={model_name}, device={self._device})")
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the BGE-M3 model."""
        if self._model is None:
            logger.info(f"Loading BGE-M3 model from {self.model_name}...")
            model = SentenceTransformer(self.model_name, device=self._device)
            
            # Optimization: Use torch.compile for PyTorch 2.0+ speed boost
            try:
                if hasattr(torch, "compile"):
                    logger.info("Compiling BGE-M3 model for faster inference...")
                    self._model = torch.compile(model)
                else:
                    self._model = model
            except Exception as e:
                logger.warning(f"Could not compile text model: {e}. Falling back to standard.")
                self._model = model
                
        return self._model
    
    def encode_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single string.
        
        Args:
            text: Input text
            
        Returns:
            Normalized 1024-dim embedding vector
        """
        if not text:
            return np.zeros(1024)
            
        try:
            # BGE-M3 produces 1024-dim vectors by default
            embedding = self.model.encode(
                text, 
                normalize_embeddings=True, 
                show_progress_bar=False
            )
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            raise
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of strings.
        
        Args:
            texts: List of input strings
            
        Returns:
            Normalized embedding matrix
        """
        try:
            embeddings = self.model.encode(
                texts, 
                normalize_embeddings=True, 
                show_progress_bar=False,
                batch_size=32
            )
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating text batch embeddings: {e}")
            raise

    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension (1024 for BGE-M3)
        """
        return 1024

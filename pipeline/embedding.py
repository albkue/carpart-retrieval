"""CLIP Embedding Generator Module.

This module provides CLIP-based image embedding generation for similarity search.
"""
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import List, Union
import logging
import torch

logger = logging.getLogger(__name__)


class CLIPEmbedding:
    """CLIP-based image embedding generator."""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        use_gpu: bool = False
    ):
        """Initialize the CLIP embedding generator.
        
        Args:
            model_name: Name of the CLIP model to use
            use_gpu: Whether to use GPU for inference
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self._model = None
        self._processor = None
        self._device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing CLIP embedder (model={model_name}, device={self._device})")
    
    @property
    def model(self) -> CLIPModel:
        """Lazy-load the CLIP model."""
        if self._model is None:
            logger.info(f"Loading CLIP model from {self.model_name}...")
            model = CLIPModel.from_pretrained(self.model_name)
            model.to(self._device)
            model.eval()  # Set to evaluation mode
            
            # Optimization: Use torch.compile for PyTorch 2.0+ speed boost
            try:
                if hasattr(torch, "compile"):
                    logger.info("Compiling CLIP model for faster inference...")
                    self._model = torch.compile(model)
                else:
                    self._model = model
            except Exception as e:
                logger.warning(f"Could not compile model: {e}. Falling back to standard model.")
                self._model = model
                
        return self._model
    
    @property
    def processor(self) -> CLIPProcessor:
        """Lazy-load the CLIP processor."""
        if self._processor is None:
            logger.info(f"Loading CLIP processor from {self.model_name}...")
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
        return self._processor
    
    def encode_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Generate embedding for a single image.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            Normalized embedding vector (numpy array)
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            
            # Normalize embedding for cosine similarity
            embedding = outputs.cpu().numpy()
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            
            return embedding[0]
            
        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            raise
    
    def encode_images(self, images: List[Union[Image.Image, np.ndarray]]) -> np.ndarray:
        """Generate embeddings for a batch of images.
        
        Args:
            images: List of input images (PIL Images or numpy arrays)
            
        Returns:
            Normalized embedding matrix (numpy array)
        """
        # Convert all to PIL Images if needed
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img))
            else:
                pil_images.append(img)
        
        try:
            inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            
            # Normalize embeddings
            embeddings = outputs.cpu().numpy()
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating image embeddings: {e}")
            raise
    
    def encode_text(self, text: str) -> np.ndarray:
        """Generate embedding for text (for text-to-image search).
        
        Args:
            text: Input text string
            
        Returns:
            Normalized embedding vector (numpy array)
        """
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
            
            # Normalize embedding
            embedding = outputs.cpu().numpy()
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            
            return embedding[0]
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            raise
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of input text strings
            
        Returns:
            Normalized embedding matrix (numpy array)
        """
        try:
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
            
            # Normalize embeddings
            embeddings = outputs.cpu().numpy()
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating text embeddings: {e}")
            raise
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1)
        """
        # Assure normalized
        e1 = embedding1 / np.linalg.norm(embedding1)
        e2 = embedding2 / np.linalg.norm(embedding2)
        
        return float(np.dot(e1, e2))
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension
        """
        # Different CLIP models have different dimensions
        # clip-vit-base-patch32: 512
        # clip-vit-large-patch14: 768 (default)
        return self.model.config.projection_dim


def get_clip_model_dimension(model_name: str) -> int:
    """Get embedding dimension for a CLIP model without loading it.
    
    Args:
        model_name: Name of the CLIP model
        
    Returns:
        Embedding dimension
    """
    # Known dimensions for common CLIP models
    dimensions = {
        "openai/clip-vit-base-patch32": 512,
        "openai/clip-vit-base-patch16": 512,
        "openai/clip-vit-large-patch14": 768,
        "openai/clip-vit-large-patch14-336": 768,
    }
    return dimensions.get(model_name, 768)

"""ML Search Service Configuration."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings for ML Search Service."""

    # Service
    SERVICE_NAME: str = "ml-search-service"
    SERVICE_PORT: int = 8001

    # Main API
    MAIN_API_URL: str = "http://localhost:8085"

    # Model paths
    YOLO_MODEL: str = "hf://albkue/car-parts-yolov8/best.pt"
    CLIP_MODEL: str = "openai/clip-vit-large-patch14"

    # Confidence thresholds
    YOLO_CONFIDENCE_THRESHOLD: float = 0.5
    OCR_CONFIDENCE_THRESHOLD: float = 0.6

    # Vector store (Qdrant container, THESIS_TRACKER §3.6)
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_IMAGE_COLLECTION: str = "products_image"
    QDRANT_TEXT_COLLECTION: str = "products_text"
    EMBEDDING_DIMENSION: int = 768

    # Text embedding (BGE-M3)
    TEXT_MODEL: str = "BAAI/bge-m3"
    TEXT_EMBEDDING_DIMENSION: int = 1024

    # GPU settings
    USE_GPU: bool = False

    # Feature flags (disable heavy models for low-RAM environments)
    ENABLE_OCR: bool = True
    ENABLE_TEXT_SEARCH: bool = True

    # CORS
    CORS_ORIGINS: str = "*"

    # Security
    API_KEY: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

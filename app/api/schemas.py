"""Request and response schemas for ML Search Service."""
from typing import List, Optional
from pydantic import BaseModel


class ImageSearchQuery(BaseModel):
    """Query parameters extracted from image."""
    part_type: Optional[str] = None
    brand_name: Optional[str] = None
    part_number: Optional[str] = None
    confidence: float = 0.0


class SearchResult(BaseModel):
    """Single search result."""
    product_id: int
    name: Optional[str] = None
    score: float
    match_type: str  # 'image', 'text', 'metadata', 'hybrid'


class ImageSearchResponse(BaseModel):
    """Response from image search endpoint."""
    query: ImageSearchQuery
    results: List[SearchResult]
    message: Optional[str] = None  # Warning or info message for user


class IndexProductRequest(BaseModel):
    """Request to index a product image."""
    product_id: int
    image_url: str


class IndexProductResponse(BaseModel):
    """Response from index product endpoint."""
    status: str
    product_id: int


class RebuildIndexResponse(BaseModel):
    """Response from rebuild index endpoint."""
    status: str
    message: Optional[str] = None

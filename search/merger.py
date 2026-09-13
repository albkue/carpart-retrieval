"""Result Merger Module.

This module provides functionality to merge and rank results from
multiple search sources (catalog DB and FAISS vector search).
"""
import logging
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result with metadata."""
    product_id: int
    score: float
    match_type: str  # 'image', 'text', 'metadata', 'hybrid'
    confidence: float
    source_scores: dict[str, float] | None = None


class ResultMerger:
    """Merges and ranks results from multiple search sources."""
    
    def __init__(
        self,
        catalog_weight: float = 0.6,
        vector_weight: float = 0.4,
        confidence_threshold: float = 0.5,
        low_confidence_catalog_weight: float = 0.3,
        low_confidence_vector_weight: float = 0.7
    ):
        """Initialize the result merger.
        
        Args:
            catalog_weight: Weight for catalog matches (high confidence)
            vector_weight: Weight for vector matches (high confidence)
            confidence_threshold: Threshold to switch weight strategies
            low_confidence_catalog_weight: Catalog weight when confidence is low
            low_confidence_vector_weight: Vector weight when confidence is low
        """
        self.catalog_weight = catalog_weight
        self.vector_weight = vector_weight
        self.confidence_threshold = confidence_threshold
        self.low_confidence_catalog_weight = low_confidence_catalog_weight
        self.low_confidence_vector_weight = low_confidence_vector_weight
        
        logger.info(
            f"ResultMerger initialized (catalog_weight={catalog_weight}, "
            f"vector_weight={vector_weight}, threshold={confidence_threshold})"
        )
    
    def merge(
        self,
        catalog_results: list[dict],
        image_results: list[tuple],
        detection_confidence: float,
        max_results: int = 20,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        text_results: list[tuple] | None = None
    ) -> list[SearchResult]:
        """Merge image, text, and catalog search results.

        Implements  α·image_score + β·text_score + γ·meta  re-ranking.

        Args:
            catalog_results: Results from catalog metadata search
            image_results: Results from FAISS image (CLIP) index
            detection_confidence: Confidence from YOLO/OCR detection
            max_results: Maximum number of results to return
            alpha: Dynamic image weight. Overrides internal logic if set.
            beta: Dynamic text weight. Overrides internal logic if set.
            gamma: Dynamic catalog/meta weight. Overrides internal logic if set.
            text_results: Results from FAISS text (BGE-M3) index

        Returns:
            Merged and ranked list of SearchResult objects
        """
        text_results = text_results or []

        # Use dynamic weights from caller if provided (diagram spec)
        # alpha = image_score weight
        # beta  = text_score weight
        # gamma = meta (catalog) weight
        if alpha is not None and beta is not None:
            image_weight = alpha
            text_weight = beta
            catalog_weight = gamma if gamma is not None else max(0.0, 1.0 - alpha - beta)
            logger.debug(
                f"Dynamic weights applied: alpha(image)={alpha}, "
                f"beta(text)={beta}, gamma(meta)={catalog_weight}"
            )
        elif detection_confidence < self.confidence_threshold:
            # Low confidence: rely more on visual similarity
            image_weight = self.low_confidence_vector_weight
            text_weight = 0.2
            catalog_weight = self.low_confidence_catalog_weight
            logger.debug(
                f"Low confidence ({detection_confidence:.2f}), "
                f"using weights: catalog={catalog_weight}, image={image_weight}"
            )
        else:
            image_weight = self.vector_weight
            text_weight = 0.2
            catalog_weight = self.catalog_weight

        # Aggregate scores by product ID
        product_scores = defaultdict(lambda: {"image": 0.0, "text": 0.0, "catalog": 0.0})

        # Add catalog (metadata) scores
        for result in catalog_results:
            product_id = result.get('product_id')
            if product_id is None:
                continue
            score = result.get('score', 1.0)
            product_scores[product_id]["catalog"] = catalog_weight * score

        # Add image (vector) scores
        for product_id, similarity in image_results:
            product_scores[product_id]["image"] = image_weight * similarity

        # Add text scores
        for product_id, similarity in text_results:
            product_scores[product_id]["text"] = text_weight * similarity

        # Calculate combined scores and determine match types
        results = []
        catalog_ids = {r.get('product_id') for r in catalog_results if r.get('product_id')}
        image_ids = {p[0] for p in image_results}
        text_ids = {p[0] for p in text_results}

        for product_id, scores in product_scores.items():
            combined_score = scores["image"] + scores["text"] + scores["catalog"]

            # Determine match type per diagram: image | text | metadata | hybrid
            sources = []
            if product_id in image_ids:
                sources.append('image')
            if product_id in text_ids:
                sources.append('text')
            if product_id in catalog_ids:
                sources.append('metadata')

            if len(sources) >= 2:
                match_type = 'hybrid'
            elif sources:
                match_type = sources[0]
            else:
                match_type = 'unknown'

            results.append(SearchResult(
                product_id=product_id,
                score=combined_score,
                match_type=match_type,
                confidence=detection_confidence,
                source_scores={
                    "image": scores["image"],
                    "text": scores["text"],
                    "catalog": scores["catalog"]
                }
            ))

        # Sort by combined score
        results.sort(key=lambda x: x.score, reverse=True)

        # Log top results
        if results:
            top_result = results[0]
            logger.info(
                f"Top result: product_id={top_result.product_id}, "
                f"score={top_result.score:.3f}, match_type={top_result.match_type}"
            )

        return results[:max_results]
    
    def merge_with_diversity(
        self,
        catalog_results: list[dict],
        image_results: list[tuple],
        detection_confidence: float,
        max_results: int = 20,
        diversity_threshold: float = 0.8,
        text_results: list[tuple] | None = None
    ) -> list[SearchResult]:
        """Merge results with diversity to avoid similar products.
        
        Args:
            catalog_results: Results from catalog search
            image_results: Results from FAISS image index
            detection_confidence: Confidence from YOLO/OCR detection
            max_results: Maximum number of results to return
            diversity_threshold: Minimum score difference to include
            text_results: Results from FAISS text index
            
        Returns:
            Diversified list of SearchResult objects
        """
        # Get initial merged results
        all_results = self.merge(
            catalog_results,
            image_results,
            detection_confidence,
            max_results=max_results * 2,  # Get more to allow filtering
            text_results=text_results
        )
        
        if not all_results:
            return []
        
        # Apply diversity filtering
        diversified = [all_results[0]]  # Always include top result
        last_score = all_results[0].score
        
        for result in all_results[1:]:
            # Include if score is significantly different or different match type
            if (last_score - result.score) > (1 - diversity_threshold) * last_score:
                diversified.append(result)
                last_score = result.score
            
            if len(diversified) >= max_results:
                break
        
        return diversified
    
    def rerank_by_match_type(
        self,
        results: list[SearchResult],
        type_weights: dict[str, float] | None = None
    ) -> list[SearchResult]:
        """Re-rank results based on match type preferences.
        
        Args:
            results: List of search results
            type_weights: Weights for each match type
            
        Returns:
            Re-ranked list of results
        """
        if type_weights is None:
            type_weights = {
                'combined': 1.2,  # Boost combined matches
                'catalog': 1.0,
                'vector': 0.9
            }
        
        # Apply type weights
        for result in results:
            weight = type_weights.get(result.match_type, 1.0)
            result.score = result.score * weight
        
        # Re-sort
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results


def format_results_for_response(
    results: list[SearchResult],
    products_data: dict[int, dict] | None = None
) -> list[dict]:
    """Format search results for API response.
    
    Args:
        results: List of SearchResult objects
        products_data: Optional product details by ID
        
    Returns:
        List of formatted result dictionaries
    """
    formatted = []
    
    for result in results:
        item = {
            "product_id": result.product_id,
            "score": round(result.score, 3),
            "match_type": result.match_type,
            "confidence": round(result.confidence, 3)
        }
        
        # Add product details if available
        if products_data and result.product_id in products_data:
            item["product"] = products_data[result.product_id]
        
        formatted.append(item)
    
    return formatted

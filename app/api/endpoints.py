"""ML Search Service API Endpoints."""
import asyncio
import logging
import re

import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from PIL import Image

from app.config import settings
from pipeline.adaptive_preprocessor import AdaptivePreprocessor
from pipeline.brand_matcher import get_brand_matcher
from pipeline.embedding import CLIPEmbedding
from pipeline.ocr_extractor import OCRExtractor
from pipeline.part_number import normalize_part_number
from pipeline.preprocessor import validate_image, validate_image_full
from pipeline.text_embedding import TextEmbedding
from pipeline.yolo_detector import YOLOPartDetector
from search.catalog_client import CatalogClient
from search.faiss_index import FAISSIndex
from search.merger import ResultMerger

from .schemas import (
    ImageSearchQuery,
    ImageSearchResponse,
    IndexProductResponse,
    RebuildIndexResponse,
)
from .schemas import (
    SearchResult as SearchResultSchema,
)

# Text-based part category keywords for OCR fallback
# When YOLO doesn't detect a part, scan OCR text for these keywords
TEXT_PART_KEYWORDS = {
    'engine_oil': ['engine oil', 'motor oil', '10w', '5w', '0w', 'synthetic oil', 'castrol', 'mobil 1', 'shell helix'],
    'air_filter': ['air filter', 'air cleaner', 'intake filter'],
    'cabin_filter': ['cabin filter', 'pollen filter', 'ac filter'],
    'fuel_filter': ['fuel filter', 'gasoline filter'],
    'brake_fluid': ['brake fluid', 'dot 3', 'dot 4', 'dot 5', 'brake oil'],
    'coolant': ['coolant', 'antifreeze', 'radiator fluid'],
    'transmission_fluid': ['transmission fluid', 'atf', 'gear oil', 'cvt fluid'],
    'power_steering_fluid': ['power steering fluid', 'steering oil'],
    'wiper_blade': ['wiper blade', 'windshield wiper', 'wiper rubber'],
    'battery': ['battery', 'accumulator', '12v battery'],
    'spark_plug': ['spark plug', 'ignition plug'],
    'brake_pad': ['brake pad', 'disc pad'],
    'oil_filter': ['oil filter'],
    'tire': ['tire', 'tyre', 'pneumatic'],
    'serpentine_belt': ['serpentine belt', 'drive belt', 'fan belt'],
    'timing_belt': ['timing belt', 'cam belt'],
    'shock_absorber': ['shock absorber', 'shock', 'strut', 'damper'],
    'headlight': ['headlight', 'headlamp', 'head light'],
    'taillight': ['taillight', 'tail lamp', 'rear light'],
    'muffler': ['muffler', 'silencer', 'exhaust'],
    'radiator': ['radiator'],
    'alternator': ['alternator', 'generator'],
    'starter': ['starter', 'starter motor'],
}


def extract_part_type_from_text(text: str) -> str | None:
    """Extract part category from OCR text using keyword matching.
    
    Args:
        text: OCR extracted text
        
    Returns:
        Part category string if matched, None otherwise
    """
    if not text:
        return None
    
    text_lower = text.lower()
    for category, keywords in TEXT_PART_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return None


def extract_part_number(text: str) -> str | None:
    """Extract part number from OCR text.
    
    Part numbers are typically alphanumeric codes like:
    - BP1234, BP-1234
    - W712/80, W712-80
    - 12345ABC
    
    Args:
        text: OCR extracted text
        
    Returns:
        Part number string if found, None otherwise
    """
    if not text:
        return None
    
    # Common part number patterns
    patterns = [
        # Pattern: 2-4 letters + numbers (e.g., BP1234, W712)
        r'\b[A-Z]{2,4}\d{2,5}[A-Z]?\b',
        # Pattern: letters + numbers + / or - + numbers (e.g., W712/80, BP-1234)
        r'\b[A-Z]{1,4}\d{2,4}[/-]\d{1,4}\b',
        # Pattern: numbers + letters (e.g., 12345ABC)
        r'\b\d{4,6}[A-Z]{1,4}\b',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.upper())
        if matches:
            # Return the longest match (most likely to be complete part number)
            return max(matches, key=len)
    
    return None

logger = logging.getLogger(__name__)

router = APIRouter()


# ... (rest of imports)

# Initialize components (lazy-loaded via properties)
class Components:
    """Lazy-loaded components."""
    _preprocessor = None
    _adaptive_preprocessor = None
    _detector = None
    _ocr = None
    _embedder = None
    _text_embedder = None  # NEW: BGE-M3
    _faiss_index = None
    _text_faiss_index = None  # NEW: Text Index
    _catalog_client = None
    _merger = None
    _brand_matcher = None
    
    @property
    def brand_matcher(self):
        if self._brand_matcher is None:
            self._brand_matcher = get_brand_matcher()
        return self._brand_matcher

    @property
    def preprocessor(self):
        if self._preprocessor is None:
            from pipeline.preprocessor import ImagePreprocessor
            self._preprocessor = ImagePreprocessor()
        return self._preprocessor

    @property
    def adaptive_preprocessor(self):
        if self._adaptive_preprocessor is None:
            self._adaptive_preprocessor = AdaptivePreprocessor()
        return self._adaptive_preprocessor

    @property
    def detector(self):
        if self._detector is None:
            self._detector = YOLOPartDetector(
                settings.YOLO_MODEL,
                settings.YOLO_CONFIDENCE_THRESHOLD,
                settings.USE_GPU
            )
        return self._detector

    @property
    def ocr(self):
        if self._ocr is None:
            self._ocr = OCRExtractor(
                settings.OCR_CONFIDENCE_THRESHOLD,
                settings.USE_GPU
            )
        return self._ocr

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = CLIPEmbedding(
                settings.CLIP_MODEL,
                settings.USE_GPU
            )
        return self._embedder

    @property
    def text_embedder(self):
        if self._text_embedder is None:
            self._text_embedder = TextEmbedding(
                settings.TEXT_MODEL,
                settings.USE_GPU
            )
        return self._text_embedder
    
    @property
    def faiss_index(self):
        if self._faiss_index is None:
            self._faiss_index = FAISSIndex(
                dimension=settings.EMBEDDING_DIMENSION,
                index_path=f"{settings.FAISS_INDEX_PATH}/image",
                index_type="hnsw",
                metric="l2"
            )
            self._faiss_index.load_index()
        return self._faiss_index

    @property
    def text_faiss_index(self):
        if self._text_faiss_index is None:
            self._text_faiss_index = FAISSIndex(
                dimension=settings.TEXT_EMBEDDING_DIMENSION,
                index_path=f"{settings.FAISS_INDEX_PATH}/text",
                index_type="hnsw",
                metric="inner_product"
            )
            self._text_faiss_index.load_index()
        return self._text_faiss_index
    
    @property
    def catalog_client(self):
        if self._catalog_client is None:
            self._catalog_client = CatalogClient(settings.MAIN_API_URL)
        return self._catalog_client
    
    @property
    def merger(self):
        if self._merger is None:
            self._merger = ResultMerger()
        return self._merger


components = Components()


@router.post("/search-by-image", response_model=ImageSearchResponse)
async def search_by_image(
    file: UploadFile = File(..., description="Image file to search"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results to return")
):
    """Search products by image using ML.
    
    This endpoint:
    1. Preprocesses the uploaded image
    2. Runs YOLOv8 detection for part type
    3. Runs OCR for brand/text extraction
    4. Generates CLIP embedding for similarity search
    5. Searches FAISS index and catalog DB
    6. Merges and ranks results
    
    Returns:
        ImageSearchResponse with query details and matching products
    """
    # 1. Full validation pipeline (magic bytes, size, EXIF strip, quality checks)
    validation = validate_image_full(await file.read())
    if not validation.valid:
        raise HTTPException(
            status_code=validation.http_status,
            detail=validation.error
        )
    image = validation.image
    validation_warning = validation.warning
    
    # 2. Preprocess image (using adaptive preprocessor)
    try:
        processed = components.adaptive_preprocessor.preprocess(image)
        analysis = components.adaptive_preprocessor.get_last_analysis()
        logger.info(f"Image analysis: brightness={analysis.get('brightness', 0):.1f}, "
                   f"contrast={analysis.get('contrast', 0):.1f}, "
                   f"dark={analysis.get('is_dark', False)}, "
                   f"bright={analysis.get('is_bright', False)}")
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(500, "Image preprocessing failed")
    
    # 3. Run detection (always) and OCR (only if ENABLE_OCR=true)
    loop = asyncio.get_event_loop()

    detection_result = await loop.run_in_executor(
        None,
        components.detector.detect,
        processed
    )

    ocr_results = []
    if settings.ENABLE_OCR:
        try:
            ocr_pil = Image.fromarray(processed)
            ocr_pil = ocr_pil.resize((1280, 1280), Image.Resampling.LANCZOS)
            from PIL import ImageFilter
            ocr_pil = ocr_pil.filter(ImageFilter.SHARPEN)
            ocr_image = np.array(ocr_pil)
        except Exception:
            ocr_image = processed
        ocr_results = await loop.run_in_executor(
            None,
            components.ocr.extract_all,
            ocr_image
        )
    else:
        logger.debug("OCR disabled (ENABLE_OCR=false)")
    
    # 4. Build query parameters
    part_type = detection_result.part_type if detection_result else None
    
    # Collect ALL OCR text lines (not just the best one)
    all_ocr_texts = [r.text for r in ocr_results if r.text] if ocr_results else []
    ocr_text = ' '.join(all_ocr_texts) if all_ocr_texts else None
    
    # Fallback: if YOLO didn't detect a known part, scan OCR text for product keywords
    if not part_type or part_type == 'unknown':
        text_part_type = extract_part_type_from_text(ocr_text)
        if text_part_type:
            part_type = text_part_type
            logger.info(f"OCR fallback part_type: {part_type} from text: {ocr_text}")
    
    # Find best brand from all OCR results
    best_brand_result = None
    if ocr_results:
        for r in ocr_results:
            if r.is_brand:
                best_brand_result = r
                break
        if not best_brand_result and ocr_results:
            best_brand_result = max(ocr_results, key=lambda x: x.confidence)
    brand_name = None
    brand_confidence = 0.0
    part_number = None
    
    if ocr_text:
        # Use brand from OCR if already matched, otherwise run brand matcher
        if best_brand_result and best_brand_result.is_brand:
            brand_name = best_brand_result.text
            brand_confidence = best_brand_result.confidence
            logger.info(f"OCR detected brand: {brand_name} (confidence: {brand_confidence:.2f})")
        else:
            # Match brand from all text
            brand_name, brand_confidence = components.brand_matcher.match_with_confidence(ocr_text)
            if brand_name:
                logger.info(f"Matched brand: {brand_name} (confidence: {brand_confidence:.2f}) from OCR: {ocr_text}")
        
        # Extract part number (alphanumeric code like BP1234, W712/80), then
        # normalise through the shared function (pipeline/part_number.py) so
        # this matches the same canonical form the catalogue stores.
        part_number = extract_part_number(ocr_text)
        if part_number:
            part_number = normalize_part_number(part_number)
            logger.info(f"Extracted part number: {part_number} from OCR: {ocr_text}")
    
    confidence = 0.0
    # Calculate overall confidence from best available source
    ocr_conf = max(r.confidence for r in ocr_results) if ocr_results else 0.0
    if detection_result or ocr_results:
        confidence = max(
            detection_result.confidence if detection_result else 0,
            ocr_conf,
            brand_confidence
        )
    
    logger.info(f"Query: part_type={part_type}, brand={brand_name}, part_number={part_number}, confidence={confidence:.2f}")
    
    # 5. Generate embedding for vector search
    try:
        processed_pil = Image.fromarray(processed)
        w, h = processed_pil.size
        
        if detection_result and detection_result.bbox:
            # OPTIMIZED: Square Crop with Padding (Anti-Distortion)
            logger.debug(f"Applying square crop to detection bbox: {detection_result.bbox}")
            x1, y1, x2, y2 = detection_result.bbox
            bw, bh = x2 - x1, y2 - y1
            
            # 1. Add 15% padding
            pad_w = int(bw * 0.15)
            pad_h = int(bh * 0.15)
            
            # 2. Find center and side length for square
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            side = max(bw + 2*pad_w, bh + 2*pad_h)
            
            # 3. Calculate square coordinates
            nx1 = max(0, cx - side/2)
            ny1 = max(0, cy - side/2)
            nx2 = min(w, cx + side/2)
            ny2 = min(h, cy + side/2)
            
            # 4. Crop and ensure it is a perfect square (handles image edges)
            image_for_clip = processed_pil.crop((nx1, ny1, nx2, ny2))
            from PIL import ImageOps
            image_for_clip = ImageOps.pad(image_for_clip, (int(side), int(side)), color=(0,0,0))
            logger.debug(f"Square padded crop created: {image_for_clip.size}")
            
        else:
            # OPTIMIZED: 70% Center Crop Fallback (Noise Reduction)
            logger.debug("No detection, applying 70% center crop for CLIP")
            left = (w - w * 0.7) / 2
            top = (h - h * 0.7) / 2
            right = (w + w * 0.7) / 2
            bottom = (h + h * 0.7) / 2
            image_for_clip = processed_pil.crop((left, top, right, bottom))
        
        # CLIP expects 224x224, embedder handles resize internally
        embedding = await loop.run_in_executor(
            None,
            components.embedder.encode_image,
            image_for_clip
        )
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(500, "Embedding generation failed")

    # 6. Search FAISS image index
    image_results = components.faiss_index.search(embedding, k=top_k * 2)

    # 6b. Search FAISS text index using OCR text (only if both OCR and text search enabled)
    text_results = []
    if settings.ENABLE_TEXT_SEARCH and ocr_text:
        try:
            text_embedding = await loop.run_in_executor(
                None,
                components.text_embedder.encode_text,
                ocr_text
            )
            text_results = components.text_faiss_index.search(text_embedding, k=top_k * 2)
            logger.info(f"Text FAISS search found {len(text_results)} results")
        except Exception as e:
            logger.warning(f"Text FAISS search failed: {e}")

    # --- Dynamic weight calculation (diagram spec) ---
    yolo_conf = detection_result.confidence if detection_result else 0.0
    ocr_conf  = max(r.confidence for r in ocr_results) if ocr_results else 0.0

    # OCR confidence gate: if OCR unreliable, zero out text weight
    if ocr_conf < 0.5:
        alpha, beta, gamma = 0.75, 0.0, 0.25   # image-only scenario
    else:
        alpha, beta, gamma = 0.4, 0.4, 0.2     # balanced scenario

    # YOLO gate: if YOLO low confidence, don't filter by category
    if yolo_conf < 0.4:
        part_type = None
        logger.info('YOLO confidence low - skipping category filter')

    # No-match threshold
    NO_MATCH_THRESHOLD = 0.35
    if not image_results:
        logger.warning('No image search results found')
    elif max(r[1] for r in image_results) < NO_MATCH_THRESHOLD:
        logger.warning(f'Low image similarity scores (max: {max(r[1] for r in image_results):.3f})')

    # 7. Search catalog metadata (tier 1: exact part_number, tier 2: fuzzy brand, tier 3: category)
    catalog_results = []
    search_params = {"limit": top_k * 2}

    try:
        # Resolve YOLO category name → category_id using backend cache
        if part_type:
            await components.catalog_client._load_categories()
            category_id = components.catalog_client.get_category_id(part_type)
            if category_id:
                search_params["category_id"] = category_id
                logger.info(f"Resolved category '{part_type}' → ID {category_id}")
            else:
                logger.warning(f"Unknown category '{part_type}' — skipping category filter")

        if brand_name:
            search_params["brand"] = brand_name
        if part_number:
            search_params["name"] = part_number

        if search_params:
            catalog_results = await components.catalog_client.search_by_params(**search_params)
            logger.info(f"Catalog search found {len(catalog_results)} results")
    except Exception as e:
        logger.warning(f"Catalog search failed (API may be down): {e}")
        catalog_results = []

    # 8. Merge results with dynamic weights: α·image + β·text + γ·meta
    merged_results = components.merger.merge(
        catalog_results,
        image_results,
        confidence,
        max_results=top_k,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        text_results=text_results
    )
    
    # 9. Format response — backend enriches product details; ML returns IDs + scores only
    results = [
        SearchResultSchema(
            product_id=r.product_id,
            score=round(r.score, 3),
            match_type=r.match_type
        )
        for r in merged_results
    ]
    
    # Add warning messages
    warning_message = None
    if validation_warning:
        warning_message = f'Image quality warning: {validation_warning}'
    elif not results:
        warning_message = 'No matching products found. Try a clearer photo or different angle.'
    elif results and results[0].score < NO_MATCH_THRESHOLD:
        warning_message = 'No confident match found. Please verify the product or try another photo.'
    elif results and results[0].score < 0.6:
        warning_message = 'Low confidence match. Please verify the product or try another photo.'
    
    response_data = {
        "query": ImageSearchQuery(
            part_type=part_type,
            brand_name=brand_name,
            part_number=part_number,
            confidence=round(confidence, 3)
        ),
        "results": results
    }
    
    if warning_message:
        response_data["message"] = warning_message
    
    return ImageSearchResponse(**response_data)


@router.post("/index-product", response_model=IndexProductResponse)
async def index_product(
    product_id: int = Query(..., description="Product ID to index"),
    image_url: str = Query(..., description="URL of product image")
):
    """Add a product image to the FAISS index.
    
    This endpoint downloads the product image, generates an embedding,
    and adds it to the FAISS index for future searches.
    
    Returns:
        IndexProductResponse with indexing status
    """
    import httpx
    
    # Download image
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(image_url)
            if response.status_code != 200:
                raise HTTPException(400, f"Could not download image from {image_url}")
            
            image_data = response.content
        except httpx.RequestError as e:
            logger.error(f"Error downloading image: {e}")
            raise HTTPException(400, f"Could not download image: {e!s}")
    
    # Validate image
    image = validate_image(image_data)
    if image is None:
        raise HTTPException(400, "Invalid image file")
    
    # Generate embedding
    try:
        embedding = components.embedder.encode_image(image)
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(500, "Embedding generation failed")
    
    # Add to FAISS index
    try:
        components.faiss_index.add_embeddings(
            embedding.reshape(1, -1),
            [product_id]
        )
        components.faiss_index.save_index()
    except Exception as e:
        logger.error(f"Error adding to index: {e}")
        raise HTTPException(500, "Failed to add to index")
    
    return IndexProductResponse(
        status="indexed",
        product_id=product_id
    )


@router.post("/rebuild-index", response_model=RebuildIndexResponse)
async def rebuild_index(
    background_tasks: BackgroundTasks,
    batch_size: int = Query(100, ge=10, le=500, description="Products per batch")
):
    """Rebuild the FAISS index from all product images.
    
    This is a background task that:
    1. Fetches all products with images from the main API
    2. Generates embeddings for each image
    3. Creates a new FAISS index
    
    Returns:
        RebuildIndexResponse with status
    """
    async def rebuild_task():
        logger.info("Starting index rebuild...")
        
        try:
            # Clear existing index
            components.faiss_index.clear()
            
            # Fetch products in batches
            skip = 0
            total_indexed = 0
            
            while True:
                products = await components.catalog_client.get_all_products_with_images(
                    skip=skip,
                    limit=batch_size
                )
                
                if not products:
                    break
                
                # Process batch
                for product in products:
                    try:
                        # Download and index
                        import httpx
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            response = await client.get(product["image_url"])
                            if response.status_code != 200:
                                continue
                            
                            image = validate_image(response.content)
                            if image is None:
                                continue
                            
                            embedding = components.embedder.encode_image(image)
                            components.faiss_index.add_embeddings(
                                embedding.reshape(1, -1),
                                [product["product_id"]]
                            )
                            total_indexed += 1
                            
                    except Exception as e:
                        logger.warning(f"Error indexing product {product.get('product_id')}: {e}")
                        continue
                
                skip += batch_size
            
            # Save the new index
            components.faiss_index.save_index()
            logger.info(f"Index rebuild complete. Indexed {total_indexed} products.")
            
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}")
    
    # Add to background tasks
    background_tasks.add_task(rebuild_task)
    
    return RebuildIndexResponse(
        status="started",
        message="Index rebuild started in background"
    )


@router.get("/index/stats")
async def get_index_stats():
    """Get statistics about the FAISS index.
    
    Returns:
        Dictionary with index statistics
    """
    return components.faiss_index.get_stats()


@router.get("/health/catalog")
async def check_catalog_health():
    """Check if main API catalog is accessible.
    
    Returns:
        Health status of catalog connection
    """
    is_healthy = await components.catalog_client.health_check()
    
    return {
        "catalog_healthy": is_healthy,
        "catalog_url": settings.MAIN_API_URL
    }


@router.get("/brands")
async def get_known_brands():
    """Get list of known auto parts brands.
    
    Returns:
        List of brand names recognized by OCR
    """
    return {
        "brands": components.brand_matcher.get_all_brands(),
        "count": len(components.brand_matcher.get_all_brands())
    }


@router.get("/categories")
async def get_supported_categories():
    """Get list of supported auto part categories.
    
    Returns:
        List of category names for detection
    """
    return {
        "categories": components.detector.get_supported_categories()
    }

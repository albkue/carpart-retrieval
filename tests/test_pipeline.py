"""Unit tests for ML Search Service Pipeline."""
import pytest
from PIL import Image
import numpy as np
import tempfile
import os

# Test preprocessor
class TestImagePreprocessor:
    """Tests for ImagePreprocessor class."""
    
    def test_preprocess_rgb_image(self):
        """Test preprocessing of RGB image."""
        from pipeline.preprocessor import ImagePreprocessor
        
        preprocessor = ImagePreprocessor()
        image = Image.new('RGB', (800, 600), color='red')
        result = preprocessor.preprocess(image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (640, 640, 3)
    
    def test_preprocess_grayscale_image(self):
        """Test preprocessing of grayscale image (should convert to RGB)."""
        from pipeline.preprocessor import ImagePreprocessor
        
        preprocessor = ImagePreprocessor()
        image = Image.new('L', (800, 600), color=128)
        result = preprocessor.preprocess(image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (640, 640, 3)
    
    def test_preprocess_custom_size(self):
        """Test preprocessing with custom target size."""
        from pipeline.preprocessor import ImagePreprocessor
        
        preprocessor = ImagePreprocessor(target_size=(320, 320))
        image = Image.new('RGB', (800, 600), color='blue')
        result = preprocessor.preprocess(image)
        
        assert result.shape == (320, 320, 3)
    
    def test_preprocess_small_image(self):
        """Test preprocessing of image smaller than target size."""
        from pipeline.preprocessor import ImagePreprocessor
        
        preprocessor = ImagePreprocessor()
        image = Image.new('RGB', (100, 100), color='green')
        result = preprocessor.preprocess(image)
        
        assert result.shape == (640, 640, 3)
    
    def test_preprocess_pil_returns_pil(self):
        """Test preprocess_pil returns PIL Image."""
        from pipeline.preprocessor import ImagePreprocessor
        
        preprocessor = ImagePreprocessor()
        image = Image.new('RGB', (800, 600), color='red')
        result = preprocessor.preprocess_pil(image)
        
        assert isinstance(result, Image.Image)
        assert result.size == (640, 640)
    
    def test_extract_roi(self):
        """Test region of interest extraction."""
        from pipeline.preprocessor import ImagePreprocessor
        
        preprocessor = ImagePreprocessor()
        image = Image.new('RGB', (800, 600), color='white')
        bbox = (100, 100, 300, 300)
        
        result = preprocessor.extract_roi(image, bbox)
        
        assert isinstance(result, Image.Image)
        assert result.size == (200, 200)


class TestImageValidation:
    """Tests for image validation."""
    
    def test_validate_valid_image(self):
        """Test validation of valid image bytes."""
        from pipeline.preprocessor import validate_image
        import io
        
        # Create a non-uniform image (noise) to pass the quality gate
        data = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        image = Image.fromarray(data)
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        result = validate_image(image_bytes)
        
        assert result is not None
        assert isinstance(result, Image.Image)
    
    def test_validate_invalid_image(self):
        """Test validation of invalid bytes."""
        from pipeline.preprocessor import validate_image
        
        invalid_bytes = b'this is not an image'
        result = validate_image(invalid_bytes)
        
        assert result is None


class TestYOLODetector:
    """Tests for YOLOPartDetector class."""
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        from pipeline.yolo_detector import YOLOPartDetector
        
        detector = YOLOPartDetector(
            model_path="yolov8n.pt",
            confidence_threshold=0.5
        )
        
        assert detector.confidence_threshold == 0.5
        assert detector.model_path == "yolov8n.pt"
    
    def test_get_supported_categories(self):
        """Test getting supported categories."""
        from pipeline.yolo_detector import YOLOPartDetector
        
        detector = YOLOPartDetector()
        categories = detector.get_supported_categories()
        
        assert isinstance(categories, list)
        assert 'brake' in categories
        assert 'filter' in categories
    
    def test_map_to_part_type(self):
        """Test class name to part type mapping."""
        from pipeline.yolo_detector import YOLOPartDetector
        
        detector = YOLOPartDetector()
        
        assert detector._map_to_part_type('brake_pad') == 'brake'
        assert detector._map_to_part_type('oil_filter') == 'filter'
        assert detector._map_to_part_type('car_battery') == 'battery'
        assert detector._map_to_part_type('unknown_object') == 'unknown'
    
    @pytest.mark.integration
    def test_detect_with_model(self):
        """Integration test: actual detection with loaded model."""
        from pipeline.yolo_detector import YOLOPartDetector
        
        detector = YOLOPartDetector(model_path="yolov8n.pt", confidence_threshold=0.3)
        # Create test image
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        result = detector.detect(image)
        
        # Result can be None or DetectionResult
        assert result is None or hasattr(result, 'part_type')


class TestOCRExtractor:
    """Tests for OCRExtractor class."""
    
    def test_ocr_initialization(self):
        """Test OCR extractor initialization."""
        from pipeline.ocr_extractor import OCRExtractor
        
        ocr = OCRExtractor(confidence_threshold=0.6)
        
        assert ocr.confidence_threshold == 0.6
    
    def test_get_known_brands(self):
        """Test getting known brands list."""
        from pipeline.ocr_extractor import OCRExtractor
        
        ocr = OCRExtractor()
        brands = ocr.get_known_brands()
        
        assert isinstance(brands, list)
        assert 'bosch' in brands
        assert 'denso' in brands
    
    def test_add_brand(self):
        """Test adding a new brand."""
        from pipeline.ocr_extractor import OCRExtractor
        
        ocr = OCRExtractor()
        initial_count = len(ocr.get_known_brands())
        
        ocr.add_brand('new_brand')
        
        assert 'new_brand' in ocr.get_known_brands()
        assert len(ocr.get_known_brands()) == initial_count + 1
    
    def test_is_brand_name(self):
        """Test brand name detection."""
        from pipeline.ocr_extractor import OCRExtractor
        
        ocr = OCRExtractor()
        
        assert ocr._is_brand_name('BOSCH') is True
        assert ocr._is_brand_name('NGK Spark Plug') is True
        assert ocr._is_brand_name('Unknown Brand') is False


class TestCLIPEmbedding:
    """Tests for CLIPEmbedding class."""
    
    def test_clip_initialization(self):
        """Test CLIP embedder initialization."""
        from pipeline.embedding import CLIPEmbedding, get_clip_model_dimension
        
        embedder = CLIPEmbedding(model_name="openai/clip-vit-base-patch32")
        
        assert embedder.model_name == "openai/clip-vit-base-patch32"
        
        # Test dimension helper
        dim = get_clip_model_dimension("openai/clip-vit-base-patch32")
        assert dim == 512
    
    @pytest.mark.integration
    def test_encode_image(self):
        """Integration test: actual image encoding."""
        from pipeline.embedding import CLIPEmbedding
        
        embedder = CLIPEmbedding(model_name="openai/clip-vit-base-patch32")
        image = Image.new('RGB', (224, 224), color='red')
        
        embedding = embedder.encode_image(image)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (512,)  # CLIP base dimension
        
        # Check normalization
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01  # Should be approximately normalized


class TestFAISSIndex:
    """Tests for FAISSIndex class."""
    
    def test_faiss_initialization(self):
        """Test FAISS index initialization."""
        from search.faiss_index import FAISSIndex
        
        index = FAISSIndex(dimension=512)
        
        assert index.dimension == 512
        assert index.index is None
    
    def test_create_flat_index(self):
        """Test creating a flat index."""
        from search.faiss_index import FAISSIndex
        
        index = FAISSIndex(dimension=512, index_type="flat")
        index.create_index()
        
        assert index.index is not None
    
    def test_add_and_search(self):
        """Test adding embeddings and searching."""
        from search.faiss_index import FAISSIndex
        
        index = FAISSIndex(dimension=512)
        index.create_index()
        
        # Add some embeddings
        embeddings = np.random.randn(5, 512).astype('float32')
        product_ids = [1, 2, 3, 4, 5]
        
        index.add_embeddings(embeddings, product_ids)
        
        # Search with first embedding
        query = embeddings[0]
        results = index.search(query, k=3)
        
        assert len(results) > 0
        assert results[0][0] == 1  # First result should be product_id 1
    
    def test_get_stats(self):
        """Test getting index statistics."""
        from search.faiss_index import FAISSIndex
        
        index = FAISSIndex(dimension=512)
        stats = index.get_stats()
        
        assert "status" in stats
        assert stats["status"] == "empty"
        
        # Add some data
        index.create_index()
        embeddings = np.random.randn(3, 512).astype('float32')
        index.add_embeddings(embeddings, [1, 2, 3])
        
        stats = index.get_stats()
        
        assert stats["status"] == "ready"
        assert stats["total_vectors"] == 3
    
    def test_save_and_load_index(self):
        """Test saving and loading index."""
        from search.faiss_index import FAISSIndex
        
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test_index")
            
            # Create and add embeddings
            index = FAISSIndex(dimension=512, index_path=index_path)
            index.create_index()
            embeddings = np.random.randn(3, 512).astype('float32')
            index.add_embeddings(embeddings, [10, 20, 30])
            index.save_index()
            
            # Load in new instance
            new_index = FAISSIndex(dimension=512, index_path=index_path)
            success = new_index.load_index()
            
            assert success is True
            assert new_index.index is not None
            assert new_index.index.ntotal == 3


class TestResultMerger:
    """Tests for ResultMerger class."""
    
    def test_merge_empty_results(self):
        """Test merging empty results."""
        from search.merger import ResultMerger
        
        merger = ResultMerger()
        results = merger.merge([], [], 0.8)
        
        assert results == []
    
    def test_merge_catalog_only(self):
        """Test merging catalog results only."""
        from search.merger import ResultMerger
        
        merger = ResultMerger()
        catalog_results = [
            {"product_id": 1, "score": 1.0},
            {"product_id": 2, "score": 1.0}
        ]
        
        results = merger.merge(catalog_results, [], 0.8)
        
        assert len(results) == 2
        assert all(r.match_type == 'metadata' for r in results)
    
    def test_merge_image_only(self):
        """Test merging image (vector) results only."""
        from search.merger import ResultMerger
        
        merger = ResultMerger()
        image_results = [(1, 0.9), (2, 0.8)]
        
        results = merger.merge([], image_results, 0.8)
        
        assert len(results) == 2
        assert all(r.match_type == 'image' for r in results)
    
    def test_merge_hybrid(self):
        """Test merging both result types."""
        from search.merger import ResultMerger
        
        merger = ResultMerger()
        catalog_results = [{"product_id": 1, "score": 1.0}]
        image_results = [(1, 0.9), (2, 0.8)]
        
        results = merger.merge(catalog_results, image_results, 0.8, max_results=10)
        
        assert len(results) == 2
        # Product 1 should have hybrid match type
        product_1 = next(r for r in results if r.product_id == 1)
        assert product_1.match_type == 'hybrid'
    
    def test_low_confidence_weighting(self):
        """Test that low confidence changes weights."""
        from search.merger import ResultMerger
        
        merger = ResultMerger(confidence_threshold=0.5)
        
        catalog_results = [{"product_id": 1, "score": 1.0}]
        vector_results = [(2, 0.9)]
        
        # High confidence
        high_conf_results = merger.merge(catalog_results, vector_results, 0.8)
        
        # Low confidence - should weight vector higher
        low_conf_results = merger.merge(catalog_results, vector_results, 0.3)
        
        # Product 2 (image only) should score higher with low confidence
        low_conf_image_score = next(
            r.score for r in low_conf_results if r.product_id == 2
        )
        high_conf_image_score = next(
            r.score for r in high_conf_results if r.product_id == 2
        )
        
        assert low_conf_image_score > high_conf_image_score


class TestFormatResults:
    """Tests for result formatting."""
    
    def test_format_results_basic(self):
        """Test basic result formatting."""
        from search.merger import ResultMerger, format_results_for_response
        
        merger = ResultMerger()
        catalog_results = [{"product_id": 1, "score": 1.0}]
        vector_results = [(2, 0.9)]
        
        results = merger.merge(catalog_results, vector_results, 0.8)
        formatted = format_results_for_response(results)
        
        assert len(formatted) == 2
        assert all('product_id' in r for r in formatted)
        assert all('score' in r for r in formatted)
        assert all('match_type' in r for r in formatted)
    
    def test_format_results_with_products(self):
        """Test result formatting with product data."""
        from search.merger import ResultMerger, format_results_for_response
        
        merger = ResultMerger()
        catalog_results = [{"product_id": 1, "score": 1.0}]
        vector_results = []
        products_data = {1: {"name": "Test Product", "price": 100}}
        
        results = merger.merge(catalog_results, vector_results, 0.8)
        formatted = format_results_for_response(results, products_data)
        
        assert formatted[0]['product']['name'] == "Test Product"


# Integration markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require models)"
    )

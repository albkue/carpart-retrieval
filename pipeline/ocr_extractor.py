"""OCR and Logo Extractor Module.

This module provides OCR-based text extraction for brand/logo recognition.
"""
import os
# Disable OneDNN/MKL-DNN on Windows to avoid PaddleOCR crashes
# Must be set BEFORE paddle/paddleocr is imported anywhere in the process
os.environ["FLAGS_use_mkldnn"] = "0"

from paddleocr import PaddleOCR
from typing import List, Optional
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result of OCR extraction."""
    text: str
    confidence: float
    box: List[List[int]]
    is_brand: bool = False


class OCRExtractor:
    """OCR-based text and brand extractor."""
    
    # Known auto parts brands (expandable)
    KNOWN_BRANDS = [
        # Major auto parts brands
        'bosch', 'denso', 'ngk', 'brembo', 'mobil', 'castrol',
        'bridgestone', 'michelin', 'continental', 'valeo',
        'mann', 'mahle', 'ferodo', 'akebono', 'trw',
        # Additional brands
        'hella', 'magneti marelli', 'delphi', 'acdelco', 'motorcraft',
        'beck arnley', 'gates', 'dayco', 'goodyear', 'pirelli',
        'yokohama', 'dunlop', 'falken', 'toyo', 'hankook',
        'k&n', 'fram', 'purolator', 'wix', 'knorr', 'wabco',
        'sachs', 'zf', 'bilstien', 'monroe', 'kyb',
        'ate', 'textar', 'pagid', 'bendix', 'raybestos',
        'champion', 'autolite', 'e3', 'dens',
        'motul', 'shell', 'total', 'petronas', 'liqui moly',
        # Oil and lubricants
        'valvoline', 'penzoil', 'amsoil', 'royal purple',
    ]
    
    def __init__(
        self,
        confidence_threshold: float = 0.3,
        use_gpu: bool = False,
        lang: str = 'en'
    ):
        """Initialize the OCR extractor.
        
        Args:
            confidence_threshold: Minimum confidence for OCR results
            use_gpu: Whether to use GPU for inference
            lang: Language for OCR (default: 'en' for English)
        """
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu
        self.lang = lang
        self._ocr = None
        
        logger.info(f"Initializing OCR extractor (lang={lang}, gpu={use_gpu})")
    
    @property
    def ocr(self) -> PaddleOCR:
        """Lazy-load the OCR model."""
        if self._ocr is None:
            logger.info("Loading PaddleOCR model...")
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False
            )
        return self._ocr
    
    def extract(self, image: np.ndarray) -> Optional[OCRResult]:
        """Extract text from image.
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            OCRResult with best brand match or highest confidence text
        """
        try:
            result = self.ocr.ocr(image, cls=True)
            
            if not result or not result[0]:
                logger.debug("No text detected in image")
                return None
            
            # Get all detected texts with confidences
            texts = []
            for line in result[0]:
                text = line[1][0]
                confidence = line[1][1]
                box = [[int(p[0]), int(p[1])] for p in line[0]]
                
                if confidence >= self.confidence_threshold:
                    texts.append(OCRResult(
                        text=text.strip(),
                        confidence=confidence,
                        box=box
                    ))
            
            if not texts:
                logger.debug(f"No text above confidence threshold {self.confidence_threshold}")
                return None
            
            # Find brand name from detected texts
            brand_result = self._find_brand(texts)
            if brand_result:
                logger.info(f"Detected brand: {brand_result.text} (confidence: {brand_result.confidence:.2f})")
                return brand_result
            
            # Return highest confidence text if no brand found
            best = max(texts, key=lambda x: x.confidence)
            logger.info(f"Detected text: {best.text} (confidence: {best.confidence:.2f})")
            return best
            
        except Exception as e:
            logger.error(f"Error during OCR extraction: {e}")
            return None
    
    def extract_all(self, image: np.ndarray) -> List[OCRResult]:
        """Extract all text from image.
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            List of OCRResult objects
        """
        try:
            result = self.ocr.ocr(image, cls=True)
            
            if not result or not result[0]:
                return []
            
            texts = []
            for line in result[0]:
                text = line[1][0]
                confidence = line[1][1]
                box = [[int(p[0]), int(p[1])] for p in line[0]]
                
                if confidence >= self.confidence_threshold:
                    is_brand = self._is_brand_name(text)
                    texts.append(OCRResult(
                        text=text.strip(),
                        confidence=confidence,
                        box=box,
                        is_brand=is_brand
                    ))
            
            # Sort by confidence
            texts.sort(key=lambda x: x.confidence, reverse=True)
            return texts
            
        except Exception as e:
            logger.error(f"Error during OCR extraction: {e}")
            return []
    
    def _find_brand(self, results: List[OCRResult]) -> Optional[OCRResult]:
        """Find brand name from OCR results.
        
        Args:
            results: List of OCR results
            
        Returns:
            OCRResult with brand name if found, None otherwise
        """
        for result in results:
            text_lower = result.text.lower()
            
            # Check for known brands
            for brand in self.KNOWN_BRANDS:
                if brand in text_lower:
                    return OCRResult(
                        text=brand.upper(),
                        confidence=result.confidence,
                        box=result.box,
                        is_brand=True
                    )
        
        return None
    
    def _is_brand_name(self, text: str) -> bool:
        """Check if text matches a known brand.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is a known brand
        """
        text_lower = text.lower()
        return any(brand in text_lower for brand in self.KNOWN_BRANDS)
    
    def add_brand(self, brand: str):
        """Add a new brand to the known brands list.
        
        Args:
            brand: Brand name to add
        """
        brand_lower = brand.lower()
        if brand_lower not in self.KNOWN_BRANDS:
            self.KNOWN_BRANDS.append(brand_lower)
            logger.info(f"Added new brand: {brand}")
    
    def get_known_brands(self) -> List[str]:
        """Get list of known brands.
        
        Returns:
            List of brand names
        """
        return self.KNOWN_BRANDS.copy()

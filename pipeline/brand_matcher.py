"""Brand Matcher Module for Auto Parts.

This module provides brand name matching from OCR text using:
- Alias/variation exact matching
- Fuzzy matching (fuzz.WRatio) for noisy OCR like 'B0SCH', 'NGl<'
- Token-level scanning for embedded brands
"""
import json
import os
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import fuzzy matching
try:
    from thefuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz
        FUZZY_AVAILABLE = True
    except ImportError:
        FUZZY_AVAILABLE = False
        logger.warning('thefuzz not installed - fuzzy brand matching disabled. Run: pip install thefuzz')

FUZZY_THRESHOLD = 82  # WRatio score threshold (0-100)


class BrandMatcher:
    """Matcher for auto parts brands from OCR text."""
    
    # Default known brands with variations
    DEFAULT_BRANDS = {
        "Bosch": ["bosch", "bosh", "boch", "bosch auto parts"],
        "Mobil": ["mobil", "mobil1", "mobil 1", "mobil one"],
        "Mann": ["mann", "mann-filter", "mann filter", "mannfilter"],
        "Castrol": ["castrol", "castrol edge", "castrol magnatec"],
        "NGK": ["ngk", "ngk spark plugs"],
        "Denso": ["denso", "denso parts"],
        "ACDelco": ["acdelco", "ac delco", "gm parts"],
        "KYB": ["kyb", "kyb shocks", "kayaba"],
        "Monroe": ["monroe", "monroe shocks"],
        "Delphi": ["delphi", "delphi auto parts"],
        "Valeo": ["valeo", "valeo parts"],
        "Continental": ["continental", "contitech"],
        "Dayco": ["dayco", "dayco belts"],
        "Gates": ["gates", "gates belts"],
        "SKF": ["skf", "skf bearings"],
        "Timken": ["timken", "timken bearings"],
        "Federal Mogul": ["federal mogul", "federal-mogul"],
        "Mahle": ["mahle", "mahle filters"],
        "Mann+Hummel": ["mann+hummel", "mann hummel"],
        "Hengst": ["hengst", "hengst filters"],
        "K&N": ["k&n", "k and n", "kn filters"],
        "FRAM": ["fram", "fram filters"],
        "WIX": ["wix", "wix filters"],
        "Purolator": ["purolator", "purolator filters"],
        "Mopar": ["mopar", "mopar parts"],
        "Motorcraft": ["motorcraft", "ford parts"],
        "Genuine Toyota": ["genuine toyota", "toyota oem"],
        "Genuine Honda": ["genuine honda", "honda oem"],
        "Brembo": ["brembo", "brembo brakes"],
        "ATE": ["ate", "ate brakes"],
        "TRW": ["trw", "trw parts"],
        "Luk": ["luk", "luk clutch"],
        "Sachs": ["sachs", "sachs shocks"],
        "Bilstein": ["bilstein", "bilstein shocks"],
        "Koni": ["koni", "koni shocks"],
    }
    
    def __init__(self, brands_file: Optional[str] = None):
        """Initialize brand matcher.
        
        Args:
            brands_file: Path to JSON file with brand definitions.
                        If None, uses DEFAULT_BRANDS.
        """
        self.brands: Dict[str, List[str]] = {}
        
        if brands_file and os.path.exists(brands_file):
            self._load_from_file(brands_file)
        else:
            self.brands = self.DEFAULT_BRANDS.copy()
            logger.info(f"Loaded {len(self.brands)} default brands")
    
    def _load_from_file(self, filepath: str):
        """Load brands from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.brands = json.load(f)
            logger.info(f"Loaded {len(self.brands)} brands from {filepath}")
        except Exception as e:
            logger.error(f"Error loading brands file: {e}")
            self.brands = self.DEFAULT_BRANDS.copy()
    
    def save_to_file(self, filepath: str):
        """Save brands to JSON file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.brands, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.brands)} brands to {filepath}")
        except Exception as e:
            logger.error(f"Error saving brands file: {e}")
    
    def match(self, text: str) -> Optional[str]:
        """Match OCR text to known brand.
        
        Args:
            text: OCR extracted text
            
        Returns:
            Brand name if matched, None otherwise
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        # Try exact match first
        for brand, variations in self.brands.items():
            for variation in variations:
                if variation in text_lower:
                    logger.debug(f"Matched brand: {brand} from text: {text}")
                    return brand
        
        return None
    
    def match_with_confidence(self, text: str) -> Tuple[Optional[str], float]:
        """Match brand with confidence score using 3-layer approach:
        1. Alias/variation exact match
        2. Token-level scan
        3. Fuzzy match (WRatio) for noisy OCR
        
        Args:
            text: OCR extracted text
            
        Returns:
            Tuple of (brand_name, confidence) or (None, 0.0)
        """
        if not text:
            return None, 0.0
        
        text_lower = text.lower()
        
        # Layer 1: Exact alias match
        for brand, variations in self.brands.items():
            for variation in variations:
                if variation in text_lower:
                    confidence = min(1.0, len(variation) / max(len(text_lower), 1))
                    logger.debug(f'Alias match: {brand} from "{text}"')
                    return brand, confidence
        
        # Layer 2: Token-level scan (handles 'BOSCH 0986424720 GERMANY')
        tokens = text_lower.split()
        for token in tokens:
            for brand, variations in self.brands.items():
                for variation in variations:
                    if token == variation:
                        logger.debug(f'Token match: {brand} from token "{token}"')
                        return brand, 0.85
        
        # Layer 3: Fuzzy match (handles 'B0SCH', 'NGl<', 'DENZ0')
        if FUZZY_AVAILABLE:
            best_brand = None
            best_score = 0
            for brand, variations in self.brands.items():
                # Check brand name itself
                score = fuzz.WRatio(text_lower, brand.lower())
                if score > best_score:
                    best_score = score
                    best_brand = brand
                # Check each variation
                for variation in variations:
                    score = fuzz.WRatio(text_lower, variation)
                    if score > best_score:
                        best_score = score
                        best_brand = brand
            
            if best_score >= FUZZY_THRESHOLD and best_brand:
                # Apply 0.95 penalty for fuzzy match (less trustworthy)
                confidence = (best_score / 100.0) * 0.95
                logger.debug(f'Fuzzy match: {best_brand} score={best_score} from "{text}"')
                return best_brand, confidence
        
        return None, 0.0
    
    def add_brand(self, brand_name: str, variations: List[str]):
        """Add a new brand with variations.
        
        Args:
            brand_name: Official brand name
            variations: List of possible text variations
        """
        self.brands[brand_name] = [v.lower() for v in variations]
        logger.info(f"Added brand: {brand_name}")
    
    def remove_brand(self, brand_name: str) -> bool:
        """Remove a brand.
        
        Args:
            brand_name: Brand name to remove
            
        Returns:
            True if removed, False if not found
        """
        if brand_name in self.brands:
            del self.brands[brand_name]
            logger.info(f"Removed brand: {brand_name}")
            return True
        return False
    
    def get_all_brands(self) -> List[str]:
        """Get list of all brand names."""
        return list(self.brands.keys())
    
    def get_variations(self, brand_name: str) -> List[str]:
        """Get variations for a specific brand."""
        return self.brands.get(brand_name, [])


# Singleton instance for reuse
_brand_matcher: Optional[BrandMatcher] = None


def get_brand_matcher() -> BrandMatcher:
    """Get or create singleton brand matcher instance."""
    global _brand_matcher
    if _brand_matcher is None:
        _brand_matcher = BrandMatcher()
    return _brand_matcher

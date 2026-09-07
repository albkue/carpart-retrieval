"""Adaptive Image Preprocessing Module.

This module provides adaptive image preprocessing that adjusts parameters
based on image characteristics (brightness, contrast, noise level).
"""
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageStat
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class AdaptivePreprocessor:
    """Adaptive image preprocessor that adjusts settings based on image analysis."""
    
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        """Initialize the adaptive preprocessor.
        
        Args:
            target_size: Target size for the processed image (width, height)
        """
        self.target_size = target_size
        self.analysis = {}
    
    def analyze_image(self, image: Image.Image) -> dict:
        """Analyze image characteristics.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Dictionary with image statistics
        """
        # Convert to grayscale for analysis
        gray = image.convert('L')
        
        # Calculate statistics
        stat = ImageStat.Stat(gray)
        
        # Brightness (mean pixel value)
        brightness = stat.mean[0]
        
        # Contrast (standard deviation)
        contrast = stat.stddev[0]
        
        # Estimate noise using Laplacian variance approximation
        # Apply slight blur and compare
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=2))
        blurred_stat = ImageStat.Stat(blurred)
        noise_estimate = abs(stat.stddev[0] - blurred_stat.stddev[0])
        
        analysis = {
            'brightness': brightness,  # 0-255
            'contrast': contrast,      # 0-128 (typical)
            'noise_estimate': noise_estimate,
            'is_dark': brightness < 60,
            'is_bright': brightness > 200,
            'is_low_contrast': contrast < 30,
            'is_high_contrast': contrast > 80,
            'is_noisy': noise_estimate > 10,
        }
        
        logger.debug(f"Image analysis: brightness={brightness:.1f}, "
                    f"contrast={contrast:.1f}, noise={noise_estimate:.1f}")
        
        return analysis
    
    def get_preprocessing_params(self, analysis: dict) -> dict:
        """Determine preprocessing parameters based on image analysis.
        
        Args:
            analysis: Image analysis dictionary from analyze_image()
            
        Returns:
            Dictionary with preprocessing parameters
        """
        params = {
            'denoise_enabled': True,
            'denoise_strength': 'medium',  # light, medium, strong
            'contrast_enhance': 1.2,       # 1.0 = no change
            'brightness_adjust': 0,        # -50 to +50
            'sharpen': False,
        }
        
        # Dark images: increase contrast, don't denoise (loses details)
        if analysis['is_dark']:
            params['contrast_enhance'] = 1.5
            params['brightness_adjust'] = 30
            params['denoise_enabled'] = False
            params['sharpen'] = True
            logger.debug("Dark image detected: high contrast, brighten, no denoise")
        
        # Bright images: reduce brightness, light denoise
        elif analysis['is_bright']:
            params['contrast_enhance'] = 1.0
            params['brightness_adjust'] = -20
            params['denoise_enabled'] = True
            params['denoise_strength'] = 'light'
            logger.debug("Bright image detected: reduce brightness, light denoise")
        
        # Low contrast images: enhance contrast
        elif analysis['is_low_contrast']:
            params['contrast_enhance'] = 1.4
            params['denoise_enabled'] = True
            params['denoise_strength'] = 'medium'
            logger.debug("Low contrast image detected: enhance contrast")
        
        # High contrast images: minimal processing
        elif analysis['is_high_contrast']:
            params['contrast_enhance'] = 1.0
            params['denoise_enabled'] = True
            params['denoise_strength'] = 'light'
            logger.debug("High contrast image detected: minimal processing")
        
        # Noisy images: strong denoise
        if analysis['is_noisy'] and params['denoise_enabled']:
            params['denoise_strength'] = 'strong'
            logger.debug("Noisy image detected: strong denoise")
        
        return params
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Apply adaptive preprocessing to an image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Preprocessed image as numpy array
        """
        # 1. Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 2. Auto-orient based on EXIF data
        image = self._auto_orient(image)
        
        # 3. Analyze image
        self.analysis = self.analyze_image(image)
        
        # 4. Get adaptive parameters
        params = self.get_preprocessing_params(self.analysis)
        
        # 5. Apply denoising
        if params['denoise_enabled']:
            image = self._denoise(image, params['denoise_strength'])
        
        # 6. Adjust brightness
        if params['brightness_adjust'] != 0:
            image = self._adjust_brightness(image, params['brightness_adjust'])
        
        # 7. Enhance contrast
        image = self._enhance_contrast(image, params['contrast_enhance'])
        
        # 8. Sharpen if needed
        if params['sharpen']:
            image = image.filter(ImageFilter.SHARPEN)
        
        # 9. Resize with padding
        image = self._resize_with_padding(image)
        
        logger.info(f"Adaptive preprocessing applied: {params}")
        
        return np.array(image)
    
    def _auto_orient(self, image: Image.Image) -> Image.Image:
        """Fix orientation based on EXIF data."""
        try:
            return ImageOps.exif_transpose(image)
        except Exception as e:
            logger.debug(f"Could not auto-orient image: {e}")
            return image
    
    def _denoise(self, image: Image.Image, strength: str) -> Image.Image:
        """Apply denoising with specified strength."""
        try:
            if strength == 'light':
                # Single median filter
                return image.filter(ImageFilter.MedianFilter(size=3))
            elif strength == 'medium':
                # Gaussian blur + median
                blurred = image.filter(ImageFilter.GaussianBlur(radius=1))
                return blurred.filter(ImageFilter.MedianFilter(size=3))
            elif strength == 'strong':
                # Stronger Gaussian blur
                return image.filter(ImageFilter.GaussianBlur(radius=2))
            else:
                return image
        except Exception as e:
            logger.debug(f"Could not denoise image: {e}")
            return image
    
    def _adjust_brightness(self, image: Image.Image, adjustment: int) -> Image.Image:
        """Adjust image brightness."""
        try:
            enhancer = ImageEnhance.Brightness(image)
            # Convert adjustment (-50 to +50) to factor (0.5 to 1.5)
            factor = 1.0 + (adjustment / 100)
            return enhancer.enhance(factor)
        except Exception as e:
            logger.debug(f"Could not adjust brightness: {e}")
            return image
    
    def _enhance_contrast(self, image: Image.Image, factor: float) -> Image.Image:
        """Enhance image contrast."""
        try:
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)
        except Exception as e:
            logger.debug(f"Could not enhance contrast: {e}")
            return image
    
    def _resize_with_padding(self, image: Image.Image) -> Image.Image:
        """Resize image maintaining aspect ratio with padding."""
        try:
            return ImageOps.fit(image, self.target_size, method=Image.Resampling.LANCZOS)
        except Exception as e:
            logger.debug(f"Could not resize image: {e}")
            return image.resize(self.target_size, Image.Resampling.LANCZOS)
    
    def get_last_analysis(self) -> dict:
        """Get analysis of the last processed image."""
        return self.analysis.copy()


# Backward compatibility - simple preprocessor function
def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """Simple preprocessing function for backward compatibility.
    
    Args:
        image: Input PIL Image
        target_size: Target size for the processed image
        
    Returns:
        Preprocessed image as numpy array
    """
    preprocessor = AdaptivePreprocessor(target_size)
    return preprocessor.preprocess(image)

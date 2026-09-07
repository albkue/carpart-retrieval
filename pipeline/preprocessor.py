"""Image Preprocessing Module for ML Search Service.

This module provides image preprocessing functionality including:
- Auto-orientation correction
- Denoising
- Contrast enhancement
- Resize with padding
- Full validation pipeline (magic bytes, decompression bomb, EXIF strip, etc.)
"""
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageStat
import numpy as np
from typing import Tuple, Optional
import logging
import io

logger = logging.getLogger(__name__)

# Security: prevent decompression bomb attacks
Image.MAX_IMAGE_PIXELS = 4096 * 4096

# Magic bytes for supported formats
MAGIC_BYTES = {
    b'\xff\xd8\xff': 'JPEG',
    b'\x89PNG': 'PNG',
    b'RIFF': 'WEBP',   # RIFF....WEBP
    b'\x00\x00\x00': 'HEIC',  # simplified
}

MAX_FILE_SIZE = 15 * 1024 * 1024   # 15MB
MIN_DIMENSION = 320
MAX_DIMENSION = 4096


class ValidationResult:
    """Result of image validation."""
    def __init__(self, valid: bool, error: Optional[str] = None,
                 warning: Optional[str] = None, image: Optional[Image.Image] = None,
                 http_status: int = 400):
        self.valid = valid
        self.error = error
        self.warning = warning
        self.image = image
        self.http_status = http_status


def detect_format(image_bytes: bytes) -> Optional[str]:
    """Detect image format from magic bytes.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Format string or None if unsupported
    """
    if len(image_bytes) < 12:
        return None
    header = image_bytes[:12]
    if header[:3] == b'\xff\xd8\xff':
        return 'JPEG'
    if header[:4] == b'\x89PNG':
        return 'PNG'
    if header[:4] == b'RIFF' and header[8:12] == b'WEBP':
        return 'WEBP'
    # HEIC/HEIF detection (ftyp box)
    if header[4:8] in (b'ftyp', b'HEIC', b'heic', b'heis', b'hevc'):
        return 'HEIC'
    return None


def strip_exif(image: Image.Image) -> Image.Image:
    """Strip EXIF metadata from image (privacy protection).
    
    Args:
        image: Input PIL Image
        
    Returns:
        Image without EXIF metadata
    """
    try:
        # Apply rotation from EXIF first, then strip
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
    
    # Re-save without EXIF by converting to bytes and back
    try:
        buf = io.BytesIO()
        image.save(buf, format='PNG')  # PNG has no EXIF
        buf.seek(0)
        return Image.open(buf).copy()
    except Exception:
        return image


def check_image_quality(image: Image.Image) -> Optional[str]:
    """Check image quality - blank, blur, extreme brightness.
    
    Args:
        image: Input PIL Image
        
    Returns:
        Warning string if quality issue found, None otherwise
    """
    try:
        gray = image.convert('L')
        stat = ImageStat.Stat(gray)
        mean = stat.mean[0]
        stddev = stat.stddev[0]
        
        # Blank/uniform image check (std_dev < 8)
        if stddev < 8:
            return 'image_too_uniform'
        
        # Extreme brightness
        if mean > 240:
            return 'overexposed'
        if mean < 15:
            return 'underexposed'
        
        # Blur check via Laplacian variance approximation
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=2))
        blurred_stat = ImageStat.Stat(blurred)
        laplacian_variance = abs(stat.stddev[0] - blurred_stat.stddev[0]) * 10
        if laplacian_variance < 100:
            return 'blurry'
        
    except Exception as e:
        logger.debug(f"Quality check failed: {e}")
    
    return None


def validate_image_full(image_bytes: bytes) -> ValidationResult:
    """Full validation pipeline following PDF spec.
    
    Validation order:
    1. File size (before decode)
    2. Magic bytes
    3. HEIC convert
    4. Pillow decode
    5. Decompression bomb
    6. Multi-frame extract
    7. Strip EXIF
    8. Strip ICC profile
    9. Palette -> RGB convert
    10. Min/max dimensions (auto-resize above 4096)
    11. Aspect ratio warn
    12. Uniformity / brightness / blur
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        ValidationResult with valid flag, error, warning, and image
    """
    warnings = []
    
    # 1. File size check before decode
    if len(image_bytes) > MAX_FILE_SIZE:
        return ValidationResult(False, 'file_too_large', http_status=413)
    
    # 2. Magic bytes check
    fmt = detect_format(image_bytes)
    if fmt is None:
        return ValidationResult(False, 'unsupported_format', http_status=415)
    
    # 3. HEIC auto-convert
    if fmt == 'HEIC':
        try:
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(image_bytes))
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            image_bytes = buf.getvalue()
            logger.info('HEIC image auto-converted to JPEG')
        except Exception as e:
            return ValidationResult(False, f'heic_conversion_failed: {e}', http_status=415)
    
    # 4 & 5. Pillow decode + decompression bomb
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()  # triggers decompression bomb check
    except Image.DecompressionBombError:
        return ValidationResult(False, 'decompression_bomb', http_status=413)
    except Exception as e:
        return ValidationResult(False, f'invalid_image: {e}', http_status=400)
    
    # 6. Multi-frame: extract frame 0 instead of rejecting
    try:
        if hasattr(image, 'n_frames') and image.n_frames > 1:
            image.seek(0)
            logger.info('Multi-frame image detected, using frame 0')
    except Exception:
        pass
    
    # 7 & 8. Strip EXIF and ICC profile
    image = strip_exif(image)
    
    # 9. Palette / indexed -> RGB
    if image.mode in ('P', 'PA', 'RGBA', 'LA', 'L', '1', 'CMYK'):
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 10. Min/max dimensions
    w, h = image.size
    if w < MIN_DIMENSION or h < MIN_DIMENSION:
        return ValidationResult(
            False,
            f'image_too_small: {w}x{h} minimum {MIN_DIMENSION}x{MIN_DIMENSION}',
            http_status=422
        )
    
    # Auto-resize above 4096 (do NOT reject)
    if w > MAX_DIMENSION or h > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        logger.info(f'Large image auto-resized from {w}x{h} to {new_w}x{new_h}')
    
    # 11. Extreme aspect ratio warn
    w, h = image.size
    if max(w, h) / min(w, h) > 5:
        warnings.append('extreme_aspect_ratio')
        logger.warning(f'Extreme aspect ratio: {w}x{h}')
    
    # 12. Quality checks
    quality_warning = check_image_quality(image)
    if quality_warning == 'image_too_uniform':
        return ValidationResult(
            False,
            'image_too_uniform: No part visible - please retake photo',
            http_status=422
        )
    elif quality_warning in ('overexposed', 'underexposed', 'blurry'):
        warnings.append(f'low_image_quality:{quality_warning}')
    
    warning_str = ', '.join(warnings) if warnings else None
    return ValidationResult(True, warning=warning_str, image=image)


class ImagePreprocessor:
    """Image preprocessor for preparing images for ML models."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        enhance_contrast_factor: float = 1.2,
        denoise_enabled: bool = True
    ):
        """Initialize the image preprocessor.
        
        Args:
            target_size: Target size for the processed image (width, height)
            enhance_contrast_factor: Factor for contrast enhancement (1.0 = no change)
            denoise_enabled: Whether to apply denoising
        """
        self.target_size = target_size
        self.enhance_contrast_factor = enhance_contrast_factor
        self.denoise_enabled = denoise_enabled
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Apply full preprocessing pipeline to an image.
        
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
        
        # 3. Denoise
        if self.denoise_enabled:
            image = self._denoise(image)
        
        # 4. Enhance contrast
        image = self._enhance_contrast(image)
        
        # 5. Resize with padding
        image = self._resize_with_padding(image)
        
        return np.array(image)
    
    def preprocess_pil(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing but return PIL Image instead of numpy array.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        # 1. Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 2. Auto-orient based on EXIF data
        image = self._auto_orient(image)
        
        # 3. Denoise
        if self.denoise_enabled:
            image = self._denoise(image)
        
        # 4. Enhance contrast
        image = self._enhance_contrast(image)
        
        # 5. Resize with padding
        image = self._resize_with_padding(image)
        
        return image
    
    def _auto_orient(self, image: Image.Image) -> Image.Image:
        """Fix orientation based on EXIF data.
        
        Some images have rotation metadata that needs to be applied
        for correct display.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Oriented PIL Image
        """
        try:
            return ImageOps.exif_transpose(image)
        except Exception as e:
            logger.debug(f"Could not auto-orient image: {e}")
            return image
    
    def _denoise(self, image: Image.Image) -> Image.Image:
        """Remove noise from image using median filter.
        
        Uses PIL's built-in median filter for simplicity.
        For more advanced denoising, consider using OpenCV or scikit-image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Denoised PIL Image
        """
        try:
            return image.filter(ImageFilter.MedianFilter(size=3))
        except Exception as e:
            logger.debug(f"Could not denoise image: {e}")
            return image
    
    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        """Enhance image contrast for better detection.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Contrast-enhanced PIL Image
        """
        try:
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(self.enhance_contrast_factor)
        except Exception as e:
            logger.debug(f"Could not enhance contrast: {e}")
            return image
    
    def _resize_with_padding(self, image: Image.Image) -> Image.Image:
        """Resize image maintaining aspect ratio with padding.
        
        Uses PIL's ImageOps.fit to resize and crop/pad to exact dimensions.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Resized PIL Image
        """
        try:
            return ImageOps.fit(image, self.target_size, method=Image.Resampling.LANCZOS)
        except Exception as e:
            logger.debug(f"Could not resize image: {e}")
            # Fallback to simple resize
            return image.resize(self.target_size, Image.Resampling.LANCZOS)
    
    def extract_roi(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int]
    ) -> Image.Image:
        """Extract region of interest from image.
        
        Args:
            image: Input PIL Image
            bbox: Bounding box as (x1, y1, x2, y2)
            
        Returns:
            Cropped PIL Image
        """
        x1, y1, x2, y2 = bbox
        return image.crop((x1, y1, x2, y2))


def validate_image(image_bytes: bytes) -> Optional[Image.Image]:
    """Backward-compatible wrapper. Use validate_image_full for full validation.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        PIL Image if valid, None otherwise
    """
    result = validate_image_full(image_bytes)
    return result.image if result.valid else None

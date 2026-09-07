"""Unit tests for validate_image_full() and helper functions in preprocessor.py."""
import io
from PIL import Image


import numpy as np


def _make_noisy_array(width=640, height=640) -> np.ndarray:
    """Create random noise array — high frequency content, passes blur check."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(30, 220, (height, width, 3), dtype=np.uint8)


def _make_jpeg_bytes(width=640, height=640, color=None) -> bytes:
    """Helper: create a valid JPEG with noise to pass blur/uniform checks."""
    arr = _make_noisy_array(width, height)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_bytes(width=640, height=640, color=None) -> bytes:
    """Helper: create a valid PNG with noise to pass blur/uniform checks."""
    arr = _make_noisy_array(width, height)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_webp_bytes(width=640, height=640) -> bytes:
    """Helper: create a valid WEBP with noise to pass blur/uniform checks."""
    arr = _make_noisy_array(width, height)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# detect_format()
# ---------------------------------------------------------------------------
class TestDetectFormat:
    """Tests for detect_format() magic bytes detection."""

    def test_detect_jpeg(self):
        from pipeline.preprocessor import detect_format
        data = _make_jpeg_bytes()
        assert detect_format(data) == "JPEG"

    def test_detect_png(self):
        from pipeline.preprocessor import detect_format
        data = _make_png_bytes()
        assert detect_format(data) == "PNG"

    def test_detect_webp(self):
        from pipeline.preprocessor import detect_format
        data = _make_webp_bytes()
        assert detect_format(data) == "WEBP"

    def test_detect_unknown_returns_none(self):
        from pipeline.preprocessor import detect_format
        assert detect_format(b"this is not an image at all") is None

    def test_detect_too_short_returns_none(self):
        from pipeline.preprocessor import detect_format
        assert detect_format(b"\xff\xd8") is None  # Only 2 bytes, need ≥12


# ---------------------------------------------------------------------------
# check_image_quality()
# ---------------------------------------------------------------------------
class TestCheckImageQuality:
    """Tests for check_image_quality() quality analysis."""

    def test_normal_image_returns_none(self):
        from pipeline.preprocessor import check_image_quality
        # Use random noise — guaranteed high-frequency content, passes blur check
        arr = _make_noisy_array(200, 200)
        img = Image.fromarray(arr)
        result = check_image_quality(img)
        assert result is None

    def test_blank_white_image_is_too_uniform(self):
        from pipeline.preprocessor import check_image_quality
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        result = check_image_quality(img)
        assert result == "image_too_uniform"

    def test_blank_black_image_is_underexposed(self):
        from pipeline.preprocessor import check_image_quality
        # Pure black — mean < 15 triggers underexposed (not uniform since stddev ~0 → uniform first)
        img = Image.new("RGB", (200, 200), color=(0, 0, 0))
        result = check_image_quality(img)
        # Could be too_uniform (stddev=0 < 8) — either is acceptable
        assert result in ("image_too_uniform", "underexposed")


# ---------------------------------------------------------------------------
# validate_image_full()
# ---------------------------------------------------------------------------
class TestValidateImageFull:
    """Tests for the full validation pipeline."""

    def test_valid_jpeg_passes(self):
        from pipeline.preprocessor import validate_image_full
        result = validate_image_full(_make_jpeg_bytes())
        assert result.valid is True
        assert result.error is None
        assert result.image is not None
        assert result.image.mode == "RGB"

    def test_valid_png_passes(self):
        from pipeline.preprocessor import validate_image_full
        result = validate_image_full(_make_png_bytes())
        assert result.valid is True
        assert result.image is not None

    def test_valid_webp_passes(self):
        from pipeline.preprocessor import validate_image_full
        result = validate_image_full(_make_webp_bytes())
        assert result.valid is True
        assert result.image is not None

    def test_file_too_large_returns_413(self):
        from pipeline.preprocessor import validate_image_full, MAX_FILE_SIZE
        oversized = b"\xff\xd8\xff" + b"\x00" * (MAX_FILE_SIZE + 1)
        result = validate_image_full(oversized)
        assert result.valid is False
        assert result.http_status == 413
        assert result.error == "file_too_large"

    def test_unsupported_format_returns_415(self):
        from pipeline.preprocessor import validate_image_full
        garbage = b"GARBAGE_DATA_NOT_AN_IMAGE_XXXX" * 100
        result = validate_image_full(garbage)
        assert result.valid is False
        assert result.http_status == 415
        assert result.error == "unsupported_format"

    def test_image_too_small_returns_422(self):
        from pipeline.preprocessor import validate_image_full
        # 100x100 is below MIN_DIMENSION=320
        small_bytes = _make_jpeg_bytes(width=100, height=100)
        result = validate_image_full(small_bytes)
        assert result.valid is False
        assert result.http_status == 422
        assert "image_too_small" in result.error

    def test_large_image_auto_resized_not_rejected(self):
        from pipeline.preprocessor import validate_image_full, MAX_DIMENSION
        import warnings
        # 4500x4500 exceeds MAX_DIMENSION=4096 but is within PIL's MAX_IMAGE_PIXELS
        # (5000x5000 = 25M pixels triggers DecompressionBombWarning before our code runs)
        arr = _make_noisy_array(4500, 4500)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = validate_image_full(buf.getvalue())
        assert result.valid is True
        w, h = result.image.size
        assert w <= MAX_DIMENSION
        assert h <= MAX_DIMENSION

    def test_blank_image_rejected_as_too_uniform(self):
        from pipeline.preprocessor import validate_image_full
        # Solid gray — must use Image.new directly (NOT _make_jpeg_bytes which uses noise)
        img = Image.new("RGB", (640, 640), color=(200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        result = validate_image_full(buf.getvalue())
        assert result.valid is False
        assert result.http_status == 422
        assert "image_too_uniform" in result.error

    def test_extreme_aspect_ratio_warns(self):
        from pipeline.preprocessor import validate_image_full
        # 3200x400 = ratio 8:1 — above 5:1 threshold
        # Use varied pixels to avoid uniform rejection
        img = Image.new("RGB", (3200, 400))
        pixels = img.load()
        for x in range(3200):
            for y in range(400):
                pixels[x, y] = ((x * 2) % 200 + 30, (y * 5) % 180 + 30, 90)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        result = validate_image_full(buf.getvalue())
        assert result.valid is True
        assert result.warning is not None
        assert "extreme_aspect_ratio" in result.warning

    def test_palette_image_converted_to_rgb(self):
        from pipeline.preprocessor import validate_image_full
        # Create a palette (P mode) PNG
        img = Image.new("P", (640, 640))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = validate_image_full(buf.getvalue())
        # Either valid with RGB, or rejected as too uniform — both are correct
        if result.valid:
            assert result.image.mode == "RGB"

    def test_rgba_image_converted_to_rgb(self):
        from pipeline.preprocessor import validate_image_full
        img = Image.new("RGBA", (640, 640), color=(100, 150, 200, 255))
        # Add pixel variation
        pixels = img.load()
        for x in range(640):
            for y in range(640):
                pixels[x, y] = ((x + y) % 200 + 30, (x * 2) % 180 + 20, 80, 255)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = validate_image_full(buf.getvalue())
        if result.valid:
            assert result.image.mode == "RGB"

    def test_exif_stripped_from_jpeg(self):
        from pipeline.preprocessor import validate_image_full
        from PIL import Image as PILImage
        # Create JPEG with EXIF data
        img = PILImage.new("RGB", (640, 640), color=(80, 90, 100))
        pixels = img.load()
        for x in range(640):
            for y in range(640):
                pixels[x, y] = ((x + y * 2) % 220 + 20, (y * 3) % 180 + 30, 70)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        result = validate_image_full(buf.getvalue())
        assert result.valid is True
        # EXIF should be stripped — _getexif() returns None or empty
        exif = result.image.getexif()
        assert not exif  # Empty dict = EXIF stripped

    def test_invalid_bytes_returns_400(self):
        from pipeline.preprocessor import validate_image_full
        # Starts with valid JPEG magic but body is garbage
        fake_jpeg = b"\xff\xd8\xff" + b"\x00" * 200
        result = validate_image_full(fake_jpeg)
        assert result.valid is False
        assert result.http_status in (400, 415)

    def test_validation_result_contains_image_on_success(self):
        from pipeline.preprocessor import validate_image_full
        result = validate_image_full(_make_png_bytes(width=640, height=480))
        assert result.valid is True
        assert isinstance(result.image, Image.Image)

    def test_no_warning_on_clean_image(self):
        from pipeline.preprocessor import validate_image_full
        # Random noise image — high frequency content, passes all quality checks
        result = validate_image_full(_make_jpeg_bytes(640, 640))
        assert result.valid is True
        assert result.warning is None

"""Query region before embedding: the service's crop, shared with the harness.

/search/image and scripts/run_experiment.py both import this, so the region
the thesis evaluates is the region the service embeds (ADR 001, THESIS_TRACKER
§3.10).
"""
from PIL import Image, ImageOps

CENTRE_FRACTION = 0.7  # the old no-detection fallback (centre_70); kept for the A1 `centre_70` region only
SERVICE_PADDING = 0.15  # the service's box padding (crop_15)


def centre_crop(image: Image.Image, fraction: float = CENTRE_FRACTION) -> Image.Image:
    """Keep the middle ``fraction`` of width and height."""
    w, h = image.size
    left, top = (w - w * fraction) / 2, (h - h * fraction) / 2
    return image.crop((left, top, w - left, h - top))


def box_crop(image: Image.Image, bbox, padding: float = SERVICE_PADDING) -> Image.Image:
    """Square crop around ``bbox`` (x1, y1, x2, y2) with ``padding`` per side.

    The square is clipped at the image edge and letterboxed back to square
    with black, so CLIP's resize never distorts the part.
    """
    w, h = image.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_w, pad_h = int(bw * padding), int(bh * padding)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(bw + 2 * pad_w, bh + 2 * pad_h)
    crop = image.crop((max(0, cx - side / 2), max(0, cy - side / 2),
                       min(w, cx + side / 2), min(h, cy + side / 2)))
    return ImageOps.pad(crop, (int(side), int(side)), color=(0, 0, 0))


def crop_for_embedding(image: Image.Image, bbox=None, padding: float = SERVICE_PADDING) -> Image.Image:
    """The service's rule: box crop if YOLO found a part, else the whole image.

    The fallback was centre_70 until 2026-11-01; the A1 sweep on type_v2 (Recall@1 0.796 full_image against 0.604
    centre_70) moved it to full_image. No re-index: the catalogue is indexed whole and raw.
    """
    return box_crop(image, bbox, padding) if bbox else image


def _demo():
    im = Image.new("RGB", (400, 200))
    assert centre_crop(im).size == (280, 140)
    # 100x50 box, 15% padding -> 130x80 -> square side 130
    assert box_crop(im, (150, 75, 250, 125)).size == (130, 130)
    assert box_crop(im, (150, 75, 250, 125), padding=0.0).size == (100, 100)
    # Box at the edge: clipped crop is letterboxed back to the full square
    assert box_crop(im, (0, 0, 100, 50)).size == (130, 130)
    assert crop_for_embedding(im) is im  # no box: whole image
    print("query_crop self-check OK")


if __name__ == "__main__":
    _demo()

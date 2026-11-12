import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))
from build_clip_dataset import (
    clip_preprocess,
    magic_format,
    stratified_split,
    to_rgb,
)


def test_magic_bytes_ignore_extension():
    assert magic_format(b"\xff\xd8\xff\xe0" + b"0" * 12) == "jpeg"
    assert magic_format(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"
    assert magic_format(b"GIF89a" + b"0" * 10) == "gif"
    assert magic_format(b"<html>not an image</html>") == "unknown"


def test_rgb_alpha_and_clip_crop():
    im = Image.new("RGBA", (400, 300), (255, 0, 0, 0))  # fully transparent red -> white
    assert to_rgb(im).getpixel((5, 5)) == (255, 255, 255)
    assert to_rgb(Image.new("L", (10, 10))).mode == "RGB"
    assert clip_preprocess(Image.new("RGB", (640, 320))).size == (224, 224)


def test_split_is_disjoint_stratified_and_flags_small_classes():
    ids = {"big": [f"b{i}" for i in range(50)], "tiny": ["t1", "t2", "t3"]}
    split, skipped = stratified_split(ids, min_per_split=2)
    assert skipped == ["tiny"] and set(split) == set(ids["big"])
    counts = {s: sum(v == s for v in split.values()) for s in ("train", "validation", "test")}
    assert counts == {"train": 40, "validation": 5, "test": 5}

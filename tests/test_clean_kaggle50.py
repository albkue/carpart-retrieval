import sys
from pathlib import Path

import torch  # noqa: F401  (Windows: torch's DLLs must load before pandas, which clean_kaggle50 imports)
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))
from clean_kaggle50 import norm_label, uniform, white_bg


def test_labels_align_with_commons_and_fix_typo():
    assert norm_label("BRAKE ROTOR") == "brake disc"
    assert norm_label("CARBERATOR") == "carburetor"
    assert norm_label(" OIL FILTER ") == "oil filter"


def test_uniform_and_white_background():
    blank = Image.new("RGB", (224, 224), "white")
    assert uniform(blank) and white_bg(blank) == 1.0
    part = blank.copy()
    part.paste((40, 40, 40), (20, 20, 204, 204))
    for x in range(20, 204, 8):
        part.paste((200, 30, 30), (x, 20, x + 4, 204))
    assert not uniform(part) and white_bg(part) == 1.0
    assert white_bg(Image.new("RGB", (224, 224), (90, 90, 90))) == 0.0

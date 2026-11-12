import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))
from catalog_pilot_mann import image_view, parse_page

PAGE = """Search
![Product Image W71294_MANN-FILTER](https://s7g10.scene7.com/is/image/mannhummel/W_712.94-1?qlt=82&ts=1)
![Product Image W71294_MANN-FILTER](https://s7g10.scene7.com/is/image/mannhummel/W_712.94-filter-with-box?qlt=82&ts=2)
![Product Image W71294_MANN-FILTER](https://s7g10.scene7.com/is/image/mannhummel/W_712.94-1?qlt=82&ts=1)
![CUK 26 023](https://s7g10.scene7.com/is/image/mannhummel/CUK_26_023-dim-1?qlt=82)
![workshop photo](https://s7g10.scene7.com/is/image/mannhummel/workshop-installing-oil-filter-in-car?qlt=82)
# Cabin Air Filter \\| Activated carbon    CUK 26 023
GTIN Code: 4011558041403
- Main application: VW (VOLKSWAGEN) Golf V
### OE Numbers
"""


def test_parse_page_keeps_raw_number_and_product_images_only():
    page = parse_page(PAGE)
    assert page["part_number_raw"] == "CUK 26 023"
    assert page["name"] == "Cabin Air Filter | Activated carbon"
    assert page["gtin"] == "4011558041403"
    assert page["lists_oe_numbers"]
    assert len(page["images"]) == 3  # duplicate and generic workshop photo dropped
    assert [image_view(u) for u in page["images"]] == ["part", "part_with_box", "dimension_drawing"]

"""Single source of truth for part-number normalisation.

Imported by both the service (image-search) and this research repo so
catalogue matching never has two implementations that can drift apart.
See docs/DATASET_SPEC.md §4.1.
"""
import re

_SEPARATOR_RE = re.compile(r"[\s\-/]+")

# OCR confusion classes. Applied only when matching, never when storing --
# a stored part_number must stay the value that was actually printed.
_CONFUSION_TABLE = str.maketrans({
    "O": "0",
    "I": "1",
    "L": "1",
    "S": "5",
    "B": "8",
})


def normalize_part_number(raw: str) -> str:
    """Canonical, storable form: uppercased, separators removed.

    '13000-K62-B00', '13000 K62 B00' and '13000K62B00' all normalise to
    '13000K62B00'.
    """
    if not raw:
        return ""
    return _SEPARATOR_RE.sub("", raw.strip().upper())


def part_number_match_key(raw: str) -> str:
    """Matching-only key: normalised form with OCR confusion classes collapsed.

    Never store this value -- it is deliberately lossy (e.g. '8' and 'B'
    become indistinguishable) and only safe for equality checks between
    two already-normalised numbers.
    """
    return normalize_part_number(raw).translate(_CONFUSION_TABLE)


def part_numbers_equivalent(a: str, b: str) -> bool:
    """True if two raw part numbers are the same part under OCR confusion."""
    return part_number_match_key(a) == part_number_match_key(b)


def _demo():
    assert normalize_part_number("13000-K62-B00") == "13000K62B00"
    assert normalize_part_number("13000 K62 B00") == "13000K62B00"
    assert normalize_part_number("13000K62B00") == "13000K62B00"
    assert normalize_part_number("") == ""
    assert normalize_part_number(None) == ""
    assert part_numbers_equivalent("W712/80", "w712-80")
    assert part_numbers_equivalent("BOSCH-8", "BOSCH-B")  # 8/B confusion
    assert not part_numbers_equivalent("13000K62B00", "13000K62B01")
    print("part_number self-check OK")


if __name__ == "__main__":
    _demo()

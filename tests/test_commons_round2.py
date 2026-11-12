import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))
from commons_round2 import near_dup, variants


def test_variants_are_twelve_and_skip_repeated_word():
    assert len(variants("water pump")) == 12 and "water pump close-up" in variants("water pump")
    assert "car battery car" not in variants("car battery") and len(variants("car battery")) == 11


def test_near_dup_uses_hamming_six():
    assert near_dup(0b1111, [(0b0000, "a")]) == ("a", 4)
    assert near_dup(0xFF, [(0x00, "a")]) is None

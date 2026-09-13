from pipeline.part_number import (
    normalize_part_number,
    part_number_match_key,
    part_numbers_equivalent,
)


def test_separator_normalisation():
    assert normalize_part_number("13000-K62-B00") == "13000K62B00"
    assert normalize_part_number("13000 K62 B00") == "13000K62B00"
    assert normalize_part_number("13000K62B00") == "13000K62B00"


def test_case_folding():
    assert normalize_part_number("w712/80") == "W71280"


def test_empty_input():
    assert normalize_part_number("") == ""
    assert normalize_part_number(None) == ""


def test_raw_preserved_by_caller():
    # normalize_part_number never mutates in place; raw stays whatever the caller kept.
    raw = "13000-K62-B00"
    normalize_part_number(raw)
    assert raw == "13000-K62-B00"


def test_ocr_confusion_matching_only():
    assert part_numbers_equivalent("BOSCH-8", "BOSCH-B")
    assert part_number_match_key("BOSCH-8") == part_number_match_key("BOSCH-B")
    # storage form must NOT collapse the confusion classes
    assert normalize_part_number("BOSCH-8") != normalize_part_number("BOSCH-B")


def test_not_equivalent():
    assert not part_numbers_equivalent("13000K62B00", "13000K62B01")

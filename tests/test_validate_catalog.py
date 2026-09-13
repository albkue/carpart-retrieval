import csv
import json

from scripts.validate_catalog import validate_catalog


def _write_batch(root, rows, name="batch_001.csv"):
    fields = [
        "part_number_raw", "brand", "category", "name", "oem_number",
        "source_url", "source_site", "collected_by", "is_distractor",
        "verified", "notes",
    ]
    with open(root / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_taxonomy(path):
    path.write_text(json.dumps([{"category_id": "braking", "label": "braking"}]))


def test_valid_row_passes(tmp_path):
    root = tmp_path / "catalog_images"
    root.mkdir()
    (root / "13000-K62-B00").mkdir()
    _write_batch(root, [{
        "part_number_raw": "13000-K62-B00", "brand": "bosch", "category": "braking",
        "name": "Brake pad set", "oem_number": "", "source_url": "http://example.com",
        "source_site": "example", "collected_by": "sothy", "is_distractor": "false",
        "verified": "true", "notes": "",
    }])
    taxonomy = tmp_path / "taxonomy.json"
    _write_taxonomy(taxonomy)

    clean_rows, error_rows = validate_catalog(root, taxonomy)

    assert error_rows == []
    assert len(clean_rows) == 1
    row = clean_rows[0]
    assert row["part_number"] == "13000K62B00"
    assert row["brand"] == "Bosch"  # resolved to canonical form
    assert row["verified"] == "true"


def test_missing_required_field_rejected(tmp_path):
    root = tmp_path / "catalog_images"
    root.mkdir()
    _write_batch(root, [{
        "part_number_raw": "13000-K62-B00", "brand": "Bosch", "category": "braking",
        "name": "", "oem_number": "", "source_url": "http://example.com",
        "source_site": "example", "collected_by": "sothy", "is_distractor": "false",
        "verified": "true", "notes": "",
    }])
    taxonomy = tmp_path / "taxonomy.json"
    _write_taxonomy(taxonomy)

    clean_rows, error_rows = validate_catalog(root, taxonomy)

    assert clean_rows == []
    assert len(error_rows) == 1
    assert "name" in error_rows[0]["errors"]


def test_unknown_category_rejected(tmp_path):
    root = tmp_path / "catalog_images"
    root.mkdir()
    _write_batch(root, [{
        "part_number_raw": "13000-K62-B00", "brand": "Bosch", "category": "not-a-real-category",
        "name": "Brake pad set", "oem_number": "", "source_url": "http://example.com",
        "source_site": "example", "collected_by": "sothy", "is_distractor": "false",
        "verified": "true", "notes": "",
    }])
    taxonomy = tmp_path / "taxonomy.json"
    _write_taxonomy(taxonomy)

    clean_rows, error_rows = validate_catalog(root, taxonomy)

    assert clean_rows == []
    assert "taxonomy" in error_rows[0]["errors"]


def test_unknown_brand_rejected(tmp_path):
    root = tmp_path / "catalog_images"
    root.mkdir()
    _write_batch(root, [{
        "part_number_raw": "13000-K62-B00", "brand": "TotallyMadeUpBrand", "category": "braking",
        "name": "Brake pad set", "oem_number": "", "source_url": "http://example.com",
        "source_site": "example", "collected_by": "sothy", "is_distractor": "false",
        "verified": "true", "notes": "",
    }])
    taxonomy = tmp_path / "taxonomy.json"
    _write_taxonomy(taxonomy)

    clean_rows, error_rows = validate_catalog(root, taxonomy)

    assert clean_rows == []
    assert "brand" in error_rows[0]["errors"]


def test_distractor_row_does_not_require_brand(tmp_path):
    root = tmp_path / "catalog_images"
    root.mkdir()
    _write_batch(root, [{
        "part_number_raw": "UNKNOWN-1", "brand": "", "category": "braking",
        "name": "Distractor brake pad", "oem_number": "", "source_url": "http://example.com",
        "source_site": "example", "collected_by": "auto-import", "is_distractor": "true",
        "verified": "false", "notes": "",
    }])
    taxonomy = tmp_path / "taxonomy.json"
    _write_taxonomy(taxonomy)

    clean_rows, error_rows = validate_catalog(root, taxonomy)

    assert error_rows == []
    assert len(clean_rows) == 1
    assert clean_rows[0]["is_distractor"] == "true"


def test_missing_image_dir_is_a_warning_not_a_rejection(tmp_path):
    root = tmp_path / "catalog_images"
    root.mkdir()
    # note: no folder created for this part number
    _write_batch(root, [{
        "part_number_raw": "13000-K62-B00", "brand": "Bosch", "category": "braking",
        "name": "Brake pad set", "oem_number": "", "source_url": "http://example.com",
        "source_site": "example", "collected_by": "sothy", "is_distractor": "false",
        "verified": "true", "notes": "",
    }])
    taxonomy = tmp_path / "taxonomy.json"
    _write_taxonomy(taxonomy)

    clean_rows, error_rows = validate_catalog(root, taxonomy)

    assert error_rows == []
    assert len(clean_rows) == 1
    assert "image dir" in clean_rows[0]["notes"]

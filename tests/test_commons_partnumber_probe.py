import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))
from commons_partnumber_probe import code_in


def test_code_in_drops_brand_and_ignores_separators():
    assert code_in("NGK BKR6E", "File:Spark plug BKR6-E.jpg")
    assert code_in("Mann W 712", "Oil filter MANN W712/75")
    assert code_in("06A 115 561 B", "VW filter 06A115561B")
    assert code_in("Optima RedTop", "An Optima Red Top battery")
    assert not code_in("NGK BKR6E", "NGK spark plug BKR5E")
    assert not code_in("Bosch FR7DC", "Bosch spark plug")

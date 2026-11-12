"""The licence filter is the only thing standing between the collector and unusable images."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "commons_collect", Path(__file__).resolve().parent.parent / "scripts" / "data" / "commons_collect.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)


def test_open_licences_are_kept():
    for lic in ["CC0", "Public domain", "CC BY 4.0", "CC BY 2.0", "CC BY 3.0", "CC-BY-4.0", "PD"]:
        assert cc.licence_ok(lic), lic


def test_share_alike_needs_the_flag():
    assert not cc.licence_ok("CC BY-SA 4.0")
    assert cc.licence_ok("CC BY-SA 4.0", allow_sa=True)


def test_non_commercial_no_derivatives_and_unknown_are_never_kept():
    for lic in ["CC BY-NC 4.0", "CC BY-NC-SA 3.0", "CC BY-ND 2.0", "CC BY-NC-ND 4.0", "", "Fair use", "GFDL", "Copyrighted free use"]:
        assert not cc.licence_ok(lic, allow_sa=True), lic

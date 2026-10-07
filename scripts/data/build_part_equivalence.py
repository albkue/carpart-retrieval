#!/usr/bin/env python3
"""Propose equivalence groups from catalogue folder names (DATASET_SPEC 4.3).

Evidence used: codes named in the same folder name. Nothing else (no oem_number
yet, no box/invoice, no external cross-references). Unsure = not grouped.

A folder name holds several codes when it joins them with '-', e.g.
`90306-KGH-901-S1200790-0334158`. Dashes also live inside one code (Honda
`90306-KGH-901`), so known single-code shapes are consumed first; whatever is
left over splits on '-'. Every group lands with `reviewed: false`: the hand check
(spec 4.3 item 3) flips it, and `needs_check` marks the parses most likely wrong.

Usage:
    python scripts/data/build_part_equivalence.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.part_number import normalize_part_number

CATALOG = ROOT / "data/raw/catalog_images"
OUT = ROOT / "data/processed/part_equivalence.json"

# One code with dashes inside; tried in this order at each position.
# ponytail: shapes seen in the Supply Spare Parts sets; a new family needs a row here.
SHAPES = [
    re.compile(r"\d{5}-[A-Z0-9]{3}-[A-Z0-9]{2,3}"),  # Honda 90306-KGH-901
    re.compile(r"\d{5}-\d{5}(?:-\d{2})?(?=-|$)"),    # 94201-30280, 95701-06016-00
    re.compile(r"IP-\d{2}-\d{4}[A-Z]*"),             # IP-01-0012
]


def split_codes(name: str) -> list[str]:
    """Longest-shape-first scan; leftovers split on '-'."""
    codes, i = [], 0
    while i < len(name):
        for shape in SHAPES:
            m = shape.match(name, i)
            if m:
                codes.append(m.group())
                i = m.end()
                break
        else:
            j = name.find("-", i)
            j = len(name) if j == -1 else j
            codes.append(name[i:j])
            i = j
        i += 1  # skip the joining dash
    return [c for c in codes if c]


def main():
    groups = []
    for d in sorted(p.name for p in CATALOG.iterdir() if p.is_dir()):
        codes = split_codes(d)
        if len(codes) < 2:
            continue
        members = [normalize_part_number(c) for c in codes]
        # Likely mis-parses: a code of <=6 chars after normalising (a stray
        # fragment like "00" or "0334548"-style tails) or a duplicate member.
        needs_check = len(set(members)) < len(members) or any(len(m) < 6 for m in members)
        groups.append({
            "folder": d,
            "members": members,
            "evidence": "folder_name",
            "needs_check": needs_check,
            "reviewed": False,
        })
    OUT.write_text(json.dumps({"rule": "DATASET_SPEC 4.3", "groups": groups}, indent=2) + "\n")
    print(f"{len(groups)} multi-code groups ({sum(g['needs_check'] for g in groups)} flagged) -> {OUT}")
    for g in groups:
        print(("? " if g["needs_check"] else "  ") + g["folder"], "=>", g["members"])


if __name__ == "__main__":
    main()

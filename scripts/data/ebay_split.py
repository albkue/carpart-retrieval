#!/usr/bin/env python3
"""Provisional part-level split of the cleaned eBay set (PROVISIONAL, not splits_v1.json).

Follows RESEARCH_PROTOCOL 3.1 and 3.2:
  * the unit is the physical part, never the image;
  * parts that may be one product travel together: a long number (7+ characters) shared
    between two parts' titles, a code shared between the sellers' own part-number lists,
    or a near-duplicate image (pHash distance <= --dup-dist)
    shared between two parts. Over-grouping is the safe direction for leakage; it is a
    SPLIT constraint only, not a scoring equivalence (DATASET_SPEC 4.3);
  * lighting and suspension and steering are the held-out categories: they get the
    partition "heldout" and never "train".

Everything else is split 60/20/20 by group inside each category (seeded).
Writes data/interim/ebay_split_provisional.json.

The eBay API licence (section 9.10) may bar training on these images: the file carries
`train_use_cleared: false` until that is settled.

Usage:
    python scripts/data/ebay_split.py
"""
import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HELD_OUT = {"lighting", "suspension and steering"}
NUMBER = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{5,}[A-Za-z0-9]")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def numbers(text: str) -> set[str]:
    return {norm(t) for t in NUMBER.findall(text) if re.search(r"\d", t) and len(norm(t)) >= 7}


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/interim/ebay_clean_v2"))
    ap.add_argument("--out", type=Path, default=Path("data/interim/ebay_split_provisional.json"))
    ap.add_argument("--items", type=Path, default=Path("data/raw/ebay/raw/items.jsonl"))
    ap.add_argument("--dup-dist", type=int, default=6)
    ap.add_argument("--generic", type=int, default=4, help="skip images matching more than this many other parts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", default="0.6,0.2,0.2", help="train,val,test")
    a = ap.parse_args()
    ratios = [float(x) for x in a.ratios.split(",")]

    catalog = read(a.src / "catalog_clean.csv")
    gallery = read(a.src / "gallery_clean.csv")
    queries = read(a.src / "queries_clean.csv")
    cat_of = {c["part_id"]: c["category"] for c in catalog}

    parent = {p: p for p in cat_of}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    # 1. long numbers shared in titles (a part's own number counts as one of its numbers)
    owners: dict[str, set[str]] = defaultdict(set)
    titles = defaultdict(list)
    for q in queries:
        titles[q["part_id"]].append(q["title"])
    for c in catalog:
        nums = numbers(c["title"]) | {norm(c["mpn"])}
        for t in titles[c["part_id"]]:
            nums |= numbers(t)
        for n in nums:
            if len(n) >= 7:
                owners[n].add(c["part_id"])
    by_number = 0
    for ps in owners.values():
        if 2 <= len(ps) <= 6:  # a number on dozens of parts is a generic word, not a product
            ps = sorted(ps)
            for p in ps[1:]:
                if find(p) != find(ps[0]):
                    by_number += 1
                union(p, ps[0])

    # 1b. every code in the seller's own part-number list (OEM, supplier, cross-reference)
    codes: dict[str, set[str]] = defaultdict(set)
    mpn_to_part = {norm(c["mpn"]): c["part_id"] for c in catalog}
    if a.items.exists():
        for line in a.items.open(encoding="utf-8"):
            it = json.loads(line)
            pid = mpn_to_part.get(norm(it.get("mpn") or ""))
            if pid:
                for code in re.split(r"[,;/]", it.get("mpn_raw") or ""):
                    if len(norm(code)) >= 6 and re.search(r"\d", code):
                        codes[norm(code)].add(pid)
    by_code = 0
    for ps in codes.values():
        if 2 <= len(ps) <= 6:
            ps = sorted(ps)
            for p in ps[1:]:
                if find(p) != find(ps[0]):
                    by_code += 1
                union(p, ps[0])

    # 2. near-duplicate images across different parts. An image that near-matches photos
    # of more than --generic parts is a banner or logo, not evidence of one product.
    imgs = [(r["part_id"], int(r["phash"], 16)) for r in gallery + queries]
    near: dict[int, set[str]] = defaultdict(set)
    pairs = []
    for i in range(len(imgs)):
        for j in range(i + 1, len(imgs)):
            if imgs[i][0] != imgs[j][0] and (imgs[i][1] ^ imgs[j][1]).bit_count() <= a.dup_dist:
                pairs.append((i, j))
                near[i].add(imgs[j][0])
                near[j].add(imgs[i][0])
    by_image = skipped_generic = 0
    for i, j in pairs:
        if len(near[i]) > a.generic or len(near[j]) > a.generic:
            skipped_generic += 1
            continue
        if find(imgs[i][0]) != find(imgs[j][0]):
            by_image += 1
        union(imgs[i][0], imgs[j][0])

    groups: dict[str, list[str]] = defaultdict(list)
    for p in cat_of:
        groups[find(p)].append(p)

    rng = random.Random(a.seed)
    assign: dict[str, dict] = {}
    names = ["train", "val", "test"]
    by_cat: dict[str, list[list[str]]] = defaultdict(list)
    for members in groups.values():
        cats = Counter(cat_of[p] for p in members)
        if cats.keys() & HELD_OUT:
            key = next(c for c in sorted(cats) if c in HELD_OUT)  # a held-out member holds the whole group out
        else:
            key = cats.most_common(1)[0][0]
        by_cat[key].append(sorted(members))
    for label, gs in sorted(by_cat.items()):
        gs.sort(key=lambda g: g[0])
        rng.shuffle(gs)
        total = sum(len(g) for g in gs)
        done, k = 0, 0
        for g in gs:
            if label in HELD_OUT:
                part = "heldout"
            else:
                while k < 2 and done >= sum(ratios[: k + 1]) * total:
                    k += 1
                part = names[k]
            for p in g:
                assign[p] = {"partition": part, "category": cat_of[p], "group": g[0]}
            done += len(g)

    # a group spanning two categories would split one product across category rules
    mixed = [g for g in groups.values() if len({cat_of[p] for p in g}) > 1]

    stats = Counter()
    for p, v in assign.items():
        stats[(v["category"], v["partition"])] += 1
    report = {
        "provisional": True,
        "train_use_cleared": False,
        "licence_note": "eBay API licence 9.10 may bar training on eBay images; do not train until cleared",
        "seed": a.seed,
        "ratios": dict(zip(names, ratios)),
        "parts": len(assign),
        "groups": len(groups),
        "parts_in_multi_part_groups": sum(len(g) for g in groups.values() if len(g) > 1),
        "unions_by_title_number": by_number,
        "unions_by_seller_code_list": by_code,
        "unions_by_image": by_image,
        "image_pairs_skipped_as_generic": skipped_generic,
        "largest_groups": sorted((len(g) for g in groups.values()), reverse=True)[:5],
        "groups_spanning_categories": len(mixed),
        "counts": {f"{c} | {p}": n for (c, p), n in sorted(stats.items())},
    }
    a.out.write_text(json.dumps({"report": report, "assignments": assign}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

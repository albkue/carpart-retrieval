#!/usr/bin/env python3
"""Provisional train / validation split of the CC BY spare-part photos in data/raw/catalog_images.

Steps: skip Kaggle50 folders (one image each, type labels only); drop near-duplicate images
(perceptual hash, Hamming <= 6) inside each part; join parts that share a near-duplicate image
across folders into one split group, so no look-alike straddles train and validation; keep parts
with 2+ unique images; split by group. Writes two manifests:

  <out>_train.json  catalog = training images, heldout_part_ids = everything not trained on
  <out>_eval.json   catalog = validation gallery, queries.valid = one image per validation part,
                    far_unknown = parts kept out of both, for the abstention test

Provisional: labels come from folder names and are unverified. Usage:
    python scripts/data/make_ccby_split.py
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def thumb(im):
    """32x32 gray crop of the non-white content. pHash alone gives the same hash to different small objects on a
    white background (a bolt, a washer and a spring did), so a pHash match is only a duplicate if this agrees too."""
    import numpy as np
    from PIL import Image
    g = np.asarray(im.convert("L").resize((128, 128)), dtype=np.uint8)
    ys, xs = np.nonzero(g < 235)
    if len(ys) >= 20:
        g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return np.asarray(Image.fromarray(g).resize((32, 32)), dtype=np.int16)


def same_content(a, b, tol=10) -> bool:
    import numpy as np
    return float(np.abs(a - b).mean()) <= tol


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def split_groups(group_ids: list[str], valid_frac: float, unknown: int, seed: int):
    """Shuffle group ids, take `unknown` for far unknowns, `valid_frac` for validation, rest train."""
    ids = sorted(group_ids)
    random.Random(seed).shuffle(ids)
    far = set(ids[:unknown])
    n_valid = max(1, round((len(ids) - unknown) * valid_frac))
    valid = set(ids[unknown:unknown + n_valid])
    train = set(ids[unknown + n_valid:])
    return train, valid, far


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=REPO / "data/raw/catalog_images")
    ap.add_argument("--out", type=Path, default=REPO / "data/processed/splits_ccby_v0")
    ap.add_argument("--valid-frac", type=float, default=0.15)
    ap.add_argument("--far-unknown", type=int, default=15)
    ap.add_argument("--hamming", type=int, default=6)
    ap.add_argument("--content-tol", type=float, default=10,
                    help="a pHash match must also have a mean abs difference <= this on the content thumbnail (-1 = pHash only, the v0 behaviour)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude", nargs="*", default=["IMAGES"],
                    help="folders that are not parts (IMAGES is a 280-photo dump, not one part)")
    a = ap.parse_args()

    import imagehash
    from PIL import Image

    records = []  # (part, path, hash)
    for d in sorted(p for p in a.root.iterdir() if p.is_dir() and not p.name.startswith("KAGGLE50") and p.name not in a.exclude):
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in SUFFIXES:
                try:
                    with Image.open(f) as im:
                        rgb = im.convert("RGB")
                        records.append((d.name, f, imagehash.phash(rgb), thumb(rgb)))
                except Exception:  # noqa: S110  an unreadable image is skipped, the count line below shows how many were read
                    pass
    print(f"{len(records)} images in {len({r[0] for r in records})} part folders")

    # near-duplicate clusters over all images
    uf = UnionFind()
    for i in range(len(records)):
        uf.find(i)
    import numpy as np
    hashes = np.array([int(str(r[2]), 16) for r in records], dtype=np.uint64)
    for i in range(len(hashes) - 1):  # exact pairwise Hamming distance, vectorised per row
        xor = hashes[i] ^ hashes[i + 1:]
        dist = np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        for j in np.nonzero(dist <= a.hamming)[0]:
            if a.content_tol < 0 or same_content(records[i][3], records[i + 1 + int(j)][3], a.content_tol):
                uf.union(i, i + 1 + int(j))

    # one image per duplicate cluster inside a part
    kept, seen = defaultdict(list), set()
    for i, (part, path, *_) in enumerate(records):
        key = (part, uf.find(i))
        if key not in seen:
            seen.add(key)
            kept[part].append(path)
    parts = {p: imgs for p, imgs in kept.items() if len(imgs) >= 2}
    print(f"{len(parts)} parts with 2+ unique images, {sum(map(len, parts.values()))} unique images")

    # split groups: parts joined by a shared near-duplicate cluster
    gu = UnionFind()
    owner = {}
    for i, (part, *_) in enumerate(records):
        gu.find(part)
        root = uf.find(i)
        if root in owner:
            gu.union(owner[root], part)
        else:
            owner[root] = part
    groups = defaultdict(list)
    for p in parts:
        groups[gu.find(p)].append(p)
    train_g, valid_g, far_g = split_groups(list(groups), a.valid_frac, a.far_unknown, a.seed)
    members = lambda gs: sorted(p for g in gs for p in groups[g])
    train_p, valid_p, far_p = members(train_g), members(valid_g), members(far_g)

    rel = lambda f: f.relative_to(REPO).as_posix()
    rng = random.Random(a.seed)
    train_catalog = [{"image": rel(f), "part_id": p} for p in train_p for f in parts[p]]
    gallery, queries = [], []
    for p in valid_p:
        imgs = list(parts[p])
        q = rng.choice(imgs)
        queries.append({"image": rel(q), "part_id": p})
        gallery += [{"image": rel(f), "part_id": p} for f in imgs if f != q]
    far_unknown = [{"image": rel(f), "part_id": p} for p in far_p for f in parts[p][:1]]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    base = {"provisional": True, "seed": a.seed, "note": "labels from folder names, unverified; CC BY 4.0 sources"}
    Path(f"{a.out}_train.json").write_text(json.dumps({
        **base, "catalog": train_catalog, "heldout_part_ids": sorted(valid_p + far_p)}, indent=1) + "\n", encoding="utf-8")
    Path(f"{a.out}_eval.json").write_text(json.dumps({
        **base, "catalog": gallery, "queries": {"valid": queries}, "far_unknown": far_unknown}, indent=1) + "\n", encoding="utf-8")
    print(f"train: {len(train_p)} parts, {len(train_catalog)} images | valid: {len(valid_p)} parts, "
          f"{len(queries)} queries, {len(gallery)} gallery images | far unknown: {len(far_p)} parts")


if __name__ == "__main__":
    main()

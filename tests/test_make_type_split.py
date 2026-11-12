import random
import sys
from pathlib import Path

import numpy as np
import torch  # noqa: F401  (Windows: torch's DLLs must load before pandas, which make_type_split imports)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))
from make_type_split import near_groups, split_type


def test_split_type_is_disjoint_and_capped():
    imgs = [f"img{i}.jpg" for i in range(200)]
    tr, ga, qu = split_type(imgs, random.Random(0), max_queries=20)
    assert len(qu) == 20 and len(tr) == 160 and len(ga) == 20
    assert not (set(tr) & set(ga)) and not (set(tr) & set(qu)) and not (set(ga) & set(qu))
    tr, ga, qu = split_type([f"s{i}" for i in range(6)], random.Random(0))
    assert len(tr) == 2 and len(ga) + len(qu) == 4 and len(qu) == 2


def test_near_groups_needs_hash_and_content():
    th_a, th_b = np.zeros((32, 32), np.int16), np.full((32, 32), 200, np.int16)
    ph = np.array([0, 1, 0, (1 << 40) - 1], dtype=np.uint64)  # 0~1 close, 2 same hash as 0 but different content
    uf = near_groups(ph, [th_a, th_a, th_b, th_a])
    assert uf.find(0) == uf.find(1)
    assert uf.find(0) != uf.find(2) and uf.find(0) != uf.find(3)

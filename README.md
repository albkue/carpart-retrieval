# Car-Part Retrieval — thesis repo

Hybrid Multimodal Car-Part Retrieval under Real-World Capture Conditions.
Thesis documents — contribution, milestones, gap audit, research protocol and
dataset spec — live in `docs/` and are deliberately kept off this remote.

## Layout

    configs/                   one YAML per experiment. Seeds live here, not in code.
    scripts/                   thesis entrypoints — what the Makefile targets call
    scripts/setup/             env bootstrap: setup_env.ps1, check_env.py
    scripts/data/              acquisition, audit and raw -> interim processing
    tests/                     assertions the protocol depends on (e.g. L2 normalisation)
    sql/                       catalogue schema
    docs/                      thesis documents. Local only, see the note above.
    literature/                paper PDFs. Local only — cite them, don't redistribute.

Generated or versioned outside git:

    data/raw/roboflow/         the 30K detection set, untouched
    data/raw/catalog_images/   catalog CSVs from juniors + downloaded product images
    data/raw/query_photos/     YOUR shop photos: <date>_<shop>_<name>/item_XXX/{1_box,2_part,3_hard}.jpg
    data/interim/              dedup output, taxonomy mapping, work in progress
    data/processed/            what experiments actually read. DVC-tracked.
    indexes/                   Qdrant snapshots, named with the embedding-model tag
    results/<config-hash>/     one directory per run. Never overwritten.
    batches/                   labelling batches for juniors (batches/_keys is PRIVATE)

The production service is a **separate repository**
([services-management/image-search](https://github.com/services-management/image-search)),
cloned as a sibling of this one. Two repos, two roles: research here, service there.

## Rules that protect the thesis

1. `data/processed/splits_v1.json` is SEALED on 15 Dec. Test is read once, in April.
2. Part numbers are normalised in exactly one function. Never by hand, never in a spreadsheet.
3. Nothing enters the index with `verified = false`.
4. `batches/_keys/` never leaves this machine.
5. Every number in the thesis comes from a committed config: `make repro EXP=<name>`.

## Collection order (this matters)

Photograph FIRST, then build catalog entries for what you photographed - that
guarantees every query is answerable. Then add scraped distractors in the same
categories (aim 1 core : 5-10 distractors) so retrieval is not artificially easy.

## Environment setup

You're on a Windows box with GPU, using conda env `ml` (Anaconda at `C:\anaconda3\envs\ml`).

```
conda activate ml
.\scripts\setup\setup_env.ps1        # installs torch + paddlepaddle-gpu (CUDA-matched) first, then requirements.txt
python scripts\setup\check_env.py    # confirms CUDA actually landed, not silently CPU
```

**Order matters.** `torch` and `paddlepaddle-gpu` need CUDA-specific package indexes.
If you `pip install -r requirements.txt` first, pip satisfies the torch dependency of
`ultralytics` / `open-clip-torch` / `FlagEmbedding` with the plain CPU wheel and gives
no error — you only find out when training is 10x slower than it should be. Always run
`setup_env.ps1`, not a bare `pip install -r requirements.txt`.

Check your CUDA version with `nvidia-smi` (top-right of the output) before running the
script, and pass it if it's not 12.6: `.\scripts\setup\setup_env.ps1 -CudaTag cu128`. Note
PyTorch and PaddlePaddle don't support identical CUDA tag sets (torch: cu118/cu126/cu128,
paddle: cu118/cu126/cu129) — `cu126` is the one both support, use it unless your driver
requires newer.

## Vector store

Qdrant, run as a container (`docker compose up qdrant`) — not an in-process library.
The reasons are metadata filtering, real deletes, and persistence: category and brand
filters run as pre-filters on the vector search itself rather than as a separate call
to the backend API, a discontinued part can actually be removed, and the index
survives a restart without a rebuild endpoint.

**Research runs use exact search, the service uses HNSW.** Pass `exact: true` on
queries from `run_experiment.py` so retrieval metrics measure the embedding and not
the approximation; the service takes the HNSW path for latency. The gap between the
two is itself measured — see `docs/RESEARCH_PROTOCOL.md`, experiment A8.

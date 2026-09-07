#!/usr/bin/env bash
# Seed D2 (retrieval catalog) with real image<->part-number pairs.
# These are DISTRACTORS (is_distractor = TRUE) until you manually verify a
# part number is genuine -- they exist to make retrieval non-trivial and to
# give you an MVB catalog before your own capture exists. They do NOT
# replace photographing your own query items first (see README "Collection
# order"). Run from the repo root.
set -euo pipefail
mkdir -p data/raw/catalog_images/tier1b

# --- Tier 1b: Supply Spare Parts (Honda/moto), true part-number labels ------
# ~1,400 images / ~660 part identities. The only public source where the
# class label IS the OEM part number (e.g. 14670-KWB-600). CC BY 4.0 --
# credit "Supply Spare Parts" (Roboflow) in Ch. 3 and in SOURCES.md.
#   pip install roboflow
#   export ROBOFLOW_API_KEY=...   (free, from app.roboflow.com -> Settings -> API)
# python - <<'PY'
# import os
# from roboflow import Roboflow
# rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
# projects = [
#     ("supply-spare-parts", "complete-spare-parts"),
#     ("supply-spare-parts", "supply-spare-parts"),
#     ("supply-spare-parts", "supply-spare-parts-detection"),
# ]
# for ws, proj in projects:
#     (rf.workspace(ws).project(proj).version(1)
#        .download("folder", location=f"data/raw/catalog_images/tier1b/{proj}"))
# PY

# --- Optional lead: enumerate more part-number-labelled workspaces ---------
# Roboflow is searchable by literal part number. Worth ~1h to find siblings
# of the supply-spare-parts family:
#   https://universe.roboflow.com/search?q=class:<part_number>
# Log anything you find in data/raw/roboflow/SOURCES.md before downloading.

echo "Uncomment the download you want, add your API key, and re-run."
echo "Then: python scripts/catalog_from_tier1b.py data/raw/catalog_images/tier1b"

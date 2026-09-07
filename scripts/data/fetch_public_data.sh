#!/usr/bin/env bash
# Public sources for D1 (detection set). Run from the repo root.
# NONE of these supply part numbers - D2 (catalog) and D3 (query photos)
# must be built by hand. See docs/DATASET_SPEC.md.
set -euo pipefail
mkdir -p data/raw/roboflow

# --- Roboflow Universe -------------------------------------------------------
# Needs your own free API key: https://app.roboflow.com  -> Settings -> API
# Record licence + download date for EVERY dataset in data/raw/roboflow/SOURCES.md
#   pip install roboflow
#   export ROBOFLOW_API_KEY=...
# python - <<'PY'
# import os
# from roboflow import Roboflow
# rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
# (rf.workspace("project-p5nyc").project("car-parts-o7dlr")
#    .version(1).download("yolov8", location="data/raw/roboflow/car-parts-o7dlr"))
# PY

# --- Kaggle ------------------------------------------------------------------
#   pip install kaggle   # then put kaggle.json in ~/.kaggle/
# kaggle datasets download -d gpiosenka/car-parts-40-classes -p data/raw/roboflow --unzip
# kaggle datasets download -d qubdidata/auto-parts-dataset    -p data/raw/roboflow --unzip

# --- Hugging Face ------------------------------------------------------------
#   pip install huggingface_hub
# huggingface-cli download DrBimmer/car-parts-and-damage-dataset \
#     --repo-type dataset --local-dir data/raw/roboflow/drbimmer

echo "Uncomment the source you want, add your key, and re-run."
echo "Then: python scripts/data/audit_dataset.py data/raw/roboflow"

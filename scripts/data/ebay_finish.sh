#!/usr/bin/env bash
# details -> download -> build, one process, resumable. Needs EBAY_* env vars.
set -u
PY=".venv/Scripts/python"
$PY scripts/data/ebay_collect.py details --limit "${1:-3700}"
$PY scripts/data/ebay_collect.py download --max-images-per-item 3
$PY scripts/data/ebay_collect.py build --min-sellers "${2:-2}"
echo BATCH_DONE

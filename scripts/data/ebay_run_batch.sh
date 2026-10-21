#!/usr/bin/env bash
# Batch driver for ebay_collect.py: brand + part-type queries so several sellers
# list the SAME part number (needed for the gallery/query split).
# Needs EBAY_CLIENT_ID / EBAY_CLIENT_SECRET in the environment. Resumable.
# Usage: bash scripts/data/ebay_run_batch.sh [details_limit] [min_sellers]
set -u
PY=".venv/Scripts/python"
LIMIT="${1:-4000}"
MINS="${2:-2}"

QUERIES=(
  "Bosch oil filter" "WIX oil filter" "Fram oil filter" "Mann oil filter" "Toyota oil filter"
  "Bosch air filter" "WIX air filter" "Fram cabin air filter" "K&N air filter"
  "Akebono brake pad" "Bosch brake pad" "Wagner brake pad" "ACDelco brake pad" "Brembo brake pad" "Power Stop brake pad"
  "brake rotor disc" "brake caliper"
  "NGK spark plug" "Denso spark plug" "Bosch spark plug" "Champion spark plug" "ignition coil"
  "Bosch wiper blade" "headlight bulb H4" "Philips headlight bulb"
  "water pump car" "radiator cooling" "thermostat car" "radiator cap"
  "control arm" "ball joint" "tie rod end" "shock absorber" "wheel bearing hub"
  "alternator" "starter motor" "car battery"
  "fuel pump" "timing belt" "serpentine belt" "oxygen sensor" "mass air flow sensor"
  "side mirror" "wheel center cap"
)

for q in "${QUERIES[@]}"; do
  $PY scripts/data/ebay_collect.py search --query "$q" --max-items 300 || break
done
$PY scripts/data/ebay_collect.py details --limit "$LIMIT"
$PY scripts/data/ebay_collect.py download --max-images-per-item 3
$PY scripts/data/ebay_collect.py build --min-sellers "$MINS"
echo BATCH_DONE

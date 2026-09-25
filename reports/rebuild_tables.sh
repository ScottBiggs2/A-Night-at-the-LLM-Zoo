#!/usr/bin/env bash
# Rebuild every LaTeX table fragment from the scored JSON, in ONE invocation.
#
#   bash reports/rebuild_tables.sh                 # -> outputs/tables/*.tex
#   OUT_DIR=/some/dir bash reports/rebuild_tables.sh
#
# One invocation matters: fragments regenerated at different times can disagree in
# the last digit (a value of 6.007 printing 6.00 in one table and 6.01 in another).
#
# Pure stdlib, no GPU, no torch -- runs on a laptop in under a second.
set -euo pipefail
cd "$(dirname "$0")/.."
D=reports/data
OUT="${OUT_DIR:-outputs/tables}"
mkdir -p "$OUT"

python3 reports/build_appendix_tables.py \
  --tier     "Mini (51M)=$D/fn_lomo_mini.json" \
  --tier     "Small (124M)=$D/fn_lomo_small.json" \
  --tier     "Medium (355M)=$D/fn_lomo_medium.json" \
  --baseline "Mini (51M)=$D/fn_phase3c_mini.json" \
  --baseline "Small (124M)=$D/fn_phase3c_small.json" \
  --baseline "Medium (355M)=$D/fn_phase3c_medium.json" \
  --density  "Mini (51M)=$D/density_mini_k50.json" \
  --density  "Small (124M)=$D/density_small_k50.json" \
  --density  "Medium (355M)=$D/density_medium_k50.json" \
  --spectrum "Mini (51M)=$D/spectrum_mini_k99.json" \
  --spectrum "Small (124M)=$D/spectrum_small_k99.json" \
  --spectrum "Medium (355M)=$D/spectrum_medium_k99.json" \
  --flowmeta "Mini (51M)=$D/flowmeta_mini.json" \
  --flowmeta "Small (124M)=$D/flowmeta_small.json" \
  --flowmeta "Medium (355M)=$D/flowmeta_medium.json" \
  --mcd      "Mini=$D/mcd_mini_043_r0p4.json" \
  --mcd      "Mini=$D/mcd_mini_048_r0p4.json" \
  --mcd      "Mini=$D/mcd_mini_050_r0p4.json" \
  --mcd      "Mini=$D/mcd_mini_075_r0p4.json" \
  --mcd      "Medium=$D/mcd_medium_043_r0p4.json" \
  --mcd      "Medium=$D/mcd_medium_048_r0p4.json" \
  --mcd      "Medium=$D/mcd_medium_050_r0p4.json" \
  --mcd      "Medium=$D/mcd_medium_075_r0p4.json" \
  --path     "$D/path_mult_k50.json" \
  --path_domain multilingual \
  --out_dir  "$OUT"

#!/usr/bin/env bash
# Rebuild every figure (PDF + PNG) -- generator side and population side.
#
#   bash reports/rebuild_figures.sh                # -> outputs/figs/
#   OUT_DIR=/some/dir bash reports/rebuild_figures.sh
#
# All three tiers, always: the generalisation and ladder figures compare scales.
#
# Pure matplotlib over JSON. No torch, no GPU -- runs on a laptop.
set -euo pipefail
cd "$(dirname "$0")/.."
D=reports/data
OUT="${OUT_DIR:-outputs/figs}"
mkdir -p "$OUT"

python3 reports/plot_conditioning.py \
  --tier       "Mini (51M)=$D/fn_lomo_mini.json" \
  --tier       "Small (124M)=$D/fn_lomo_small.json" \
  --tier       "Medium (355M)=$D/fn_lomo_medium.json" \
  --functional "$D/fn_lomo_mini.json" \
  --gen        "flow=$D/path_mult_k50.json" \
  --density    "$D/density_mini_k50.json" \
  --density    "$D/density_small_k50.json" \
  --density    "$D/density_medium_k50.json" \
  --mcd        "$D/mcd_mini_043_r0p4.json" \
  --mcd        "$D/mcd_mini_048_r0p4.json" \
  --mcd        "$D/mcd_mini_050_r0p4.json" \
  --mcd        "$D/mcd_mini_075_r0p4.json" \
  --arms       "$D/arms/mini_sweep.json" \
  --out_dir    "$OUT"

# The population-side figures, from the same payload the tables read.
python3 reports/plot_population.py \
  --payload reports/population_payload.json \
  --gram    "Mini (51M)=reports/data/population/mini_b015/weight_embeddings" \
  --out_dir "$OUT"

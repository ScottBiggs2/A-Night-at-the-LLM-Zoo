#!/bin/bash
# Function-space half of the population report: do the members DISAGREE?
#
#   ZOO=zoo_b015_mini_n100/gpt2_zoo_mini ARCH=gpt2_zoo_mini TAG=mini_b015 \
#       sbatch slurm/population_fn.sbatch
#   ZOO=zoo_b015_small_n100/gpt2_zoo_small ARCH=gpt2_zoo_small TAG=small_b015 \
#       TIME=02:00:00 sbatch slurm/population_fn.sbatch
#
# `slurm/population.sbatch` answers the weight-space half on the `cpu` partition.
# This one needs forward passes, so it needs a GPU -- one is enough.
#
# TWO MEASUREMENTS, AND THEY ARE NOT INTERCHANGEABLE
#
#   diag_predictions.py          pairwise next-token disagreement. This is the
#                                diversity evidence (Fort, Hu & Lakshminarayanan
#                                2019): a function-space distance matrix to put
#                                beside the weight-space branch-delta cosines.
#
#   diag_interpolation_barrier.py  linear-interpolation barrier (Frankle et al.
#                                2020). A LOW barrier licenses skipping weight
#                                canonicalization, and a
#                                sceptic reads the same number as "so they are
#                                all the same model". Report it BESIDE the
#                                cosines, never instead of them.
#
# NEITHER SCRIPT TOUCHES eval_domains.py. That file is the domain-separation gate
# and may be chained afterok behind branch arrays; a queued job runs whatever is
# deployed at exec time. Both scripts import the gate's own `load_evalsets`, so
# they score byte-identical tokens with no risk to a queued chain. `load_evalsets`
# REFUSES to build a slice, so the gate must have run on this zoo first -- which
# it has, or there would be no domain_separation.json to report against.

#SBATCH --job-name=population_fn
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
CODE_DIR="${CODE_DIR:-$SLURM_SUBMIT_DIR}"
source "$CODE_DIR/slurm/env.sh"

ZOO="${ZOO:-zoo_b015_mini_n100/gpt2_zoo_mini}"
ARCH="${ARCH:-gpt2_zoo_mini}"
TAG="${TAG:-$(basename "$(dirname "$ZOO")")}"
OUT="${OUT:-$ARTIFACT_DIR/population/$TAG}"
N_PAIRS="${N_PAIRS:-20}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"

ZOO_DIR="$ARTIFACT_DIR/$ZOO"
if [ ! -f "$ZOO_DIR/zoo_meta.json" ]; then
    echo "No zoo_meta.json at $ZOO_DIR." >&2
    exit 1
fi

echo "=============================================="
echo "Zoo    : $ZOO_DIR"
echo "Arch   : $ARCH"
echo "Out    : $OUT"
echo "Pairs  : $N_PAIRS per pool"
echo "=============================================="
echo "members on disk: $(find "$ZOO_DIR" -maxdepth 1 -name 'w_*.npy' | wc -l)"

echo; echo "######## next-token disagreement ########"
python -u scripts/diag_predictions.py \
    --arch "$ARCH" --zoo_dir "$ZOO_DIR" --out "$OUT" \
    ${SAVE_PREDICTIONS:+--save_predictions}

echo; echo "######## linear-interpolation barrier ########"
python -u scripts/diag_interpolation_barrier.py \
    --arch "$ARCH" --zoo_dir "$ZOO_DIR" --out "$OUT" \
    --n_pairs "$N_PAIRS"

echo
echo "Done. Copy down with:"
echo "  rsync -av --exclude 'top1_predictions.npy' <cluster>:$OUT/ reports/data/population/$TAG/"

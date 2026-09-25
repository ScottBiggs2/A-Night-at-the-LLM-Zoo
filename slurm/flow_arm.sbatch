#!/bin/bash
# Train ONE arm of a pi-flow sweep -- one array task per line of a manifest.
#
#   RUN=zoo_b015_mini_k99 K=50 sbatch slurm/flow_arm.sbatch                    # baseline
#   RUN=zoo_b015_mini_k99 K=50 MANIFEST=reports/data/manifests/sweep_b015.tsv \
#       sbatch --array=0-5 slurm/flow_arm.sbatch
#   RUN=zoo_b015_mini_k99 K=50 MANIFEST=reports/data/manifests/lomo_b015.tsv \
#       sbatch --array=0-84%16 slurm/flow_arm.sbatch
#
# The manifest comes from scripts/make_flow_manifest.py and has one arm per line:
#     tag <TAB> nn_l1 <TAB> n_excluded_mixtures <TAB> n_train_rows <TAB> extra_args
# `tag` becomes train_flow.py --tag, so each arm writes flow_k<K>_codes_pi_<tag>
# and they cannot overwrite one another.
#
# THE BASIS IS THE WHOLE ZOO, ON PURPOSE. A train-only (`tr92`) basis made every held-out number
# uninterpretable: a member absent from the basis keeps only 40-64% of its squared
# displacement when projected into it, which inflates its distance from everything
# and biases the data-copying test TOWARD declaring copying. That is a property of
# dual-Gram PCA, not of the flow, so it is documented as a limitation of the
# representation rather than fought. Holding out a MIXTURE from the flow's training
# rows -- which is what every manifest line does -- tests the generator without
# touching the coordinate system.
#
# `cpu` partition: 51k parameters over <=100 rows is a CPU job, and it should not
# compete with GPU training for a slot.

#SBATCH --job-name=flow_arm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
CODE_DIR="${CODE_DIR:-$SLURM_SUBMIT_DIR}"
source "$CODE_DIR/slurm/env.sh"

RUN="${RUN:-zoo_b015_mini_k99}"
K="${K:-50}"
# 20000, not the committed 500. Round-trip skill runs 0.031 at 500 epochs and 0.785
# at 20000, monotonically; 85 rows at batch 64 is ~1000 optimiser steps at the
# default, i.e. the committed default trains for about a thousandth of what this
# needs. The conditional does over-tighten by 20000 (a_pi 0.46 at sealed mixtures),
# which is what the --path_noise arms of the sweep exist to fix.
EPOCHS="${EPOCHS:-20000}"
HIDDEN="${HIDDEN:-64 128 64}"
MANIFEST="${MANIFEST:-}"
FORCE="${FORCE:-0}"
# 'pi' for a conditional flow; 'family' for the MI ablation, which on a
# one-architecture zoo is an nn.Embedding with a single row and therefore genuinely
# unconditional. Set here rather than smuggled in through a manifest's extra_args:
# argparse would take the last --cond_mode and the winner would depend on argument
# order, which is exactly the kind of thing that silently measures the wrong arm.
COND_MODE="${COND_MODE:-pi}"

FORCE_FLAG=""
if [ "$FORCE" = "1" ]; then FORCE_FLAG="--force"; fi

TAG_ARGS=()
EXTRA_ARGS=()
if [ -n "$MANIFEST" ]; then
    IDX="${SLURM_ARRAY_TASK_ID:?MANIFEST set but not submitted with --array}"
    [ -f "$MANIFEST" ] || { echo "no manifest at $MANIFEST" >&2; exit 1; }
    # Skip the header, then take line IDX+1 of what remains.
    LINE=$(grep -v '^#' "$MANIFEST" | sed -n "$((IDX + 1))p")
    [ -n "$LINE" ] || { echo "manifest has no arm at index $IDX" >&2; exit 1; }
    TAG=$(printf '%s' "$LINE" | cut -f1)
    NN=$(printf '%s' "$LINE" | cut -f2)
    NROWS=$(printf '%s' "$LINE" | cut -f4)
    EXTRA=$(printf '%s' "$LINE" | cut -f5)
    TAG_ARGS=(--tag "$TAG")
    # Word-splitting of EXTRA is the point: it is a pre-validated argv fragment.
    # shellcheck disable=SC2206
    EXTRA_ARGS=($EXTRA)
    echo "arm $IDX: tag=$TAG  nearest-surviving-mixture-L1=$NN  train_rows=$NROWS"
fi

echo "=============================================="
echo "Run     : $RUN   k=$K   epochs=$EPOCHS   hidden=($HIDDEN)"
echo "Basis   : WHOLE ZOO (no tr92); exclusion is on training rows only"
echo "=============================================="

# Expand optional arrays as ${A[@]+"${A[@]}"}: this script runs `set -u`, and
# expanding an EMPTY array under `set -u` is fatal on bash < 4.4.
# shellcheck disable=SC2086
python -u scripts/train_flow.py \
    --artifact_dir "$ARTIFACT_DIR" --run_name "$RUN" \
    --k "$K" --space codes --cond_mode "$COND_MODE" \
    --hidden_dims $HIDDEN --cond_dim 64 --time_embed_dim 64 \
    --cond_dropout 0.15 \
    --epochs "$EPOCHS" --patience "$EPOCHS" --holdout_frac 0.15 \
    --lr 3e-4 --batch_size 64 --weight_decay 1e-4 --n_steps 50 \
    --dispersion_n 256 $FORCE_FLAG \
    ${TAG_ARGS[@]+"${TAG_ARGS[@]}"} \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "=== done ==="

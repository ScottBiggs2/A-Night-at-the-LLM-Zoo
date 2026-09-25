#!/bin/bash
# Dump the per-component conditioning payload, then score it.
#
#   sbatch slurm/conditioning.sbatch                         # data only, all ranks
#   FLOW_DIRS="$(ls -d diag_cond/*)" sbatch slurm/conditioning.sbatch
#   RUN=zoo_b030_mini_k99 K="99 50" sbatch slurm/conditioning.sbatch
#
# The `cpu` partition, deliberately. Sampling a 51k-parameter flow 256 times is
# microseconds and the payload is a 100x99 matrix, so there is nothing here a GPU
# would help with, and a cpu job does not compete with GPU training for a slot.
#
# Stage B is pure stdlib and could equally run on a laptop against the copied-down
# payload; it runs here only so the log carries the numbers next to the dump that
# produced them. `report_conditioning.py --data_dir <dir>` reproduces it offline
# forever, which is the same contract report_retrieval.py has.

#SBATCH --job-name=conditioning
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
# sbatch does not set $0 to the script path, so point at the header explicitly.
CODE_DIR="${CODE_DIR:-$SLURM_SUBMIT_DIR}"
source "$CODE_DIR/slurm/env.sh"

RUN="${RUN:-zoo_b015_mini_k99}"
K="${K:-99 50 25 10}"
N_SAMPLES="${N_SAMPLES:-256}"
N_DIRICHLET="${N_DIRICHLET:-64}"
# Unset by default: the data path needs no flow at all, and the zoo's own conditional
# structure is the half that has to be right before any generator number means
# anything. Pass FLOW_DIRS to add sealed cells.
FLOW_DIRS="${FLOW_DIRS:-}"
OUT="${OUT:-$ARTIFACT_DIR/runs/$RUN/conditioning}"

RUN_ROOT="$ARTIFACT_DIR/runs/$RUN"
if [ ! -d "$RUN_ROOT" ]; then
    echo "No run at $RUN_ROOT." >&2
    exit 1
fi

echo "=============================================="
echo "Run       : $RUN"
echo "Ranks     : $K"
echo "Samples   : $N_SAMPLES per target, $N_DIRICHLET probe mixtures"
echo "Flow dirs : ${FLOW_DIRS:-<none — data path only>}"
echo "Out       : $OUT"
echo "=============================================="

echo; echo "######## Stage A: dump ########"
# An empty FLOW_DIRS must expand to NOTHING, not to an empty string argument --
# argparse would take "" as a directory name and fail on the missing flow_meta.json.
# Word-splitting is intended here, so the variable is deliberately unquoted.
# shellcheck disable=SC2086
python -u scripts/diag_conditioning.py \
    --artifact_dir "$ARTIFACT_DIR" --run_name "$RUN" \
    --k $K --n_samples "$N_SAMPLES" --n_dirichlet "$N_DIRICHLET" \
    --out "$OUT" \
    ${FLOW_DIRS:+--flow_dirs $FLOW_DIRS}

echo; echo "######## Stage B: score ########"
python -u scripts/report_conditioning.py --data_dir "$OUT" \
    --json_out "$OUT/conditioning_report.json"

echo; echo "Done. Copy down with:"
echo "  rsync -av <cluster>:$OUT/ reports/data/cond/"

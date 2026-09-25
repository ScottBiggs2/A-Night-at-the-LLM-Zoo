#!/bin/bash
# Score ONE trained flow arm at the mixture it was held out from.
#
#   RUN=zoo_b015_mini_k99 K=50 ZOO=zoo_b015_mini_n100 ARCH=gpt2_zoo_mini \
#   GLOB='flow_k50_codes_pi_*lomo_*' OUT=gen/select \
#       sbatch --array=0-35%8 slurm/eval_arm.sbatch
#
# THE TARGET MIXTURE IS READ FROM THE FLOW'S OWN flow_meta.json, not from the
# manifest that trained it. The manifest is a file on disk that can be regenerated,
# reordered or overwritten while an array is still queued -- and `train.center_pi`
# inside the checkpoint is what that flow was ACTUALLY blinded to. Reading the
# checkpoint makes the evaluation impossible to mis-join, and it means this script
# has no dependency on the design that produced the arms.
#
# One GPU: the flow sample is free, the cohort decode is one streaming pass over the
# ensemble, and the five perplexities are the only real work (~0.55 s per model at
# Mini, measured).
#
# Measured resource shape at Mini, 32 samples/arm: ~4 min and 11-25 GB on one GPU.
# Slurm must find the whole request free on one node, so an inflated reservation
# waits longer. Raise --mem for a bigger arch rather than padding the default:
# Small and Medium decode a larger cohort (memory is cohort x D x 4 bytes).

#SBATCH --job-name=eval_arm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
CODE_DIR="${CODE_DIR:-$SLURM_SUBMIT_DIR}"
source "$CODE_DIR/slurm/env.sh"

RUN="${RUN:-zoo_b015_mini_k99}"
K="${K:-50}"
ZOO="${ZOO:-zoo_b015_mini_n100}"
ARCH="${ARCH:-gpt2_zoo_mini}"
GLOB="${GLOB:-flow_k${K}_codes_pi_*lomo_*}"
N_SAMPLES="${N_SAMPLES:-32}"
GUIDANCE="${GUIDANCE:-}"
# Sampling-time only: n_steps changes the ODE integration, never the
# checkpoint, so a step sweep re-scores sealed flows and retrains nothing.
# Empty means "use the flow's own sealed default" (50), so every existing
# invocation of this script is byte-identical.
N_STEPS="${N_STEPS:-}"
SEED="${SEED:-}"
OUT="${OUT:-gen/select}"

RUN_ROOT="$ARTIFACT_DIR/runs/$RUN"
# --zoo_dir takes the ARCH directory, not its parent: zoo_meta.json and the w_*.npy
# live one level down. Checked here because the alternative is discovering it after
# a GPU has been allocated and the torch import has been paid for.
ZOO_DIR="$ARTIFACT_DIR/$ZOO/$ARCH"
if [ ! -f "$ZOO_DIR/zoo_meta.json" ]; then
    echo "no zoo_meta.json under $ZOO_DIR -- ZOO should be the directory that" >&2
    echo "CONTAINS $ARCH, e.g. ZOO=zoo_b015_mini_n100 ARCH=gpt2_zoo_mini." >&2
    exit 1
fi
IDX="${SLURM_ARRAY_TASK_ID:?submit with --array}"

# Sorted, so the index->arm map is stable across re-runs of the same array.
# A directory is only an ARM once flow_meta.json exists: train_flow.py creates the
# directory before it starts and seals the meta at the end, so a failed or still
# running arm would otherwise take an index and then die on a missing file --
# shifting every later index and silently scoring the wrong mixture.
mapfile -t DIRS < <(cd "$RUN_ROOT" &&     for d in $GLOB; do [ -f "$d/flow_meta.json" ] && echo "$d"; done | sort)
if [ "${#DIRS[@]}" -eq 0 ]; then
    echo "no flow dirs under $RUN_ROOT matching '$GLOB'" >&2; exit 1
fi
if [ "$IDX" -ge "${#DIRS[@]}" ]; then
    echo "array index $IDX >= ${#DIRS[@]} arms; nothing to do" >&2; exit 1
fi
FDIR="${DIRS[$IDX]}"

PI=$(python - "$RUN_ROOT/$FDIR/flow_meta.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
c = (m.get("train") or {}).get("center_pi")
if not c:
    sys.exit("flow_meta.json has no train.center_pi -- this arm was not trained "
             "with an exclusion ball, so there is no held-out mixture to score.")
print(" ".join(repr(float(x)) for x in c))
PY
)

# ---- optional multiple-choice benchmarks ------------------------------------
#
# BENCH="hellaswag mmlu" adds a downstream suite beside the five-domain perplexity,
# scored on the same in-memory model, so one decode pass yields both. Off by
# default: the decode is the expensive part of every arm that already ran and
# nothing existing should change shape. BENCH_N_SHOTS=0 is the measured choice for
# these models -- at n_questions=2000, 18.8% of 5-shot MMLU prompts exceed GPT-2's
# 1024 positions and get left-truncated, while 0-shot exceeds it for none.
BENCH="${BENCH:-}"
BENCH_NQ="${BENCH_NQ:-200}"
BENCH_SEED="${BENCH_SEED:-0}"
BENCH_N_SHOTS="${BENCH_N_SHOTS:-}"
BENCH_ARGS=()
if [ -n "$BENCH" ]; then
    # shellcheck disable=SC2206
    BENCH_ARGS=(--bench $BENCH --bench_n_questions "$BENCH_NQ"
                --bench_seed "$BENCH_SEED")
    if [ -n "$BENCH_N_SHOTS" ]; then
        BENCH_ARGS+=(--bench_n_shots "$BENCH_N_SHOTS")
    fi
fi

OUT_DIR="$ARTIFACT_DIR/$OUT"
mkdir -p "$OUT_DIR"
OUT_JSON="$OUT_DIR/${FDIR}.json"

echo "=============================================="
echo "arm $IDX/${#DIRS[@]} : $FDIR"
echo "target pi          : $PI"
echo "samples            : $N_SAMPLES"
echo "n_steps            : ${N_STEPS:-<flow default>}   seed=${SEED:-<default 0>}"
echo "=============================================="

GUID_ARGS=()
if [ -n "$GUIDANCE" ]; then GUID_ARGS=(--guidance_scale "$GUIDANCE"); fi
STEP_ARGS=()
if [ -n "$N_STEPS" ]; then STEP_ARGS=(--n_steps "$N_STEPS"); fi
SEED_ARGS=()
if [ -n "$SEED" ]; then SEED_ARGS=(--seed "$SEED"); fi

# shellcheck disable=SC2086
python -u scripts/eval_generated.py \
    --artifact_dir "$ARTIFACT_DIR" --run_name "$RUN" --k "$K" \
    --source flow --flow_dir "$FDIR" \
    --zoo_dir "$ARTIFACT_DIR/$ZOO/$ARCH" --arch "$ARCH" \
    --targets none --pi $PI \
    --n_samples "$N_SAMPLES" --arm "$FDIR" \
    --out "$OUT_JSON" \
    ${GUID_ARGS[@]+"${GUID_ARGS[@]}"} \
    ${STEP_ARGS[@]+"${STEP_ARGS[@]}"} \
    ${SEED_ARGS[@]+"${SEED_ARGS[@]}"} \
    ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}

echo "=== done -> $OUT_JSON ==="

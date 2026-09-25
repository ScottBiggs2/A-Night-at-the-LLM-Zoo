#!/bin/bash
# Tiny-mode end-to-end smoke test of the FULL stack: ensembles -> per-family PCA ->
# codes artifact -> codes-space flows -> all four eval arms -> report.
#
#   sbatch slurm/flow_smoke.sbatch
#
# Separate from slurm/stack_smoke.sbatch on purpose: that one is the known-green
# ~1-minute baseline for the pipeline without flows, and keeping it fast and green is
# worth more than one fewer file. It is also the bisect point if this script starts
# failing.

#SBATCH --job-name=flow_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
# sbatch does not set $0 to the script path, so point at the header explicitly.
CODE_DIR="${CODE_DIR:-$SLURM_SUBMIT_DIR}"
source "$CODE_DIR/slurm/env.sh"

RUN=flow_smoke
RUN_ROOT="$ARTIFACT_DIR/runs/$RUN"
rm -rf "$RUN_ROOT"

COMMON="--mode tiny --run_name $RUN --artifact_dir $ARTIFACT_DIR --n_samples 12 \
        --noise_scale 1e-2 --chunk_budget_mb 8 --no_wandb"

FCOMMON="--run_name $RUN --artifact_dir $ARTIFACT_DIR --no_wandb \
         --epochs 300 --patience 300 --batch_size 8 \
         --hidden_dims 64 128 64 --cond_dim 16 --time_embed_dim 16 --n_steps 20"

echo "############ 1. train_stack k=11 (N-1) ############"
python -u scripts/train_stack.py $COMMON --k 11

echo; echo "############ 2. train_stack k=6 (N/2) — must reuse Stage 2 ############"
python -u scripts/train_stack.py $COMMON --k 6

echo; echo "############ 3. codes-space flows, both ranks ############"
for K in 11 6; do
  echo "--- k=$K space=codes ---"
  python -u scripts/train_flow.py $FCOMMON --k "$K" --space codes
done

echo; echo "############ 4. eval, ALL FOUR arms, both ranks ############"
python -u scripts/eval_stack.py --artifact_dir "$ARTIFACT_DIR" --run_name "$RUN" \
    --k 11 6 \
    --arms pca_only gauss_codes flow_codes flow_rt_codes \
    --flow_rt_steps 5 20 \
    --bench synthetic --bench_n_questions 8 \
    --eval_seq_len 64 --eval_n_sequences 4 --no_wandb

echo; echo "############ 5. report ############"
python -u scripts/report_stack.py --run_dir "$RUN_ROOT" \
    --output "$RUN_ROOT/results/report_stack.md"
head -60 "$RUN_ROOT/results/report_stack.md"

echo; echo "############ 6. refusals must fire ############"
# Capture first, then grep: under pipefail, `python ... | grep -q` takes python's
# (intended) non-zero exit and the check fails even when the refusal fired.
out=$(python -u scripts/eval_stack.py --artifact_dir "$ARTIFACT_DIR" --run_name "$RUN" \
       --mode tiny --k 11 --arms pca_only --bench mmlu --no_wandb 2>&1 || true)
if grep -q "Refusing --bench" <<<"$out"; then
    echo "  OK: tiny mode refused a real benchmark"
else
    echo "  FAIL: tiny mode accepted a real benchmark"; exit 1
fi

out=$(python -u scripts/train_flow.py $FCOMMON --k 11 --space codes 2>&1 || true)
if grep -q "already holds a sealed flow" <<<"$out"; then
    echo "  OK: refused to overwrite a sealed flow without --force"
else
    echo "  FAIL: overwrote a sealed flow"; exit 1
fi

# The gate reads ev0/MEDIAN, not ev0/ev[k-1]. On this pure-noise ensemble the
# former is ~1.0 (correctly flat) while the latter is ~12 at k=N-1, because ev[k-1]
# there is the smallest direction surviving the rank floor. Gating on the tail ratio
# would let a flat spectrum straight through, so this assertion pins the choice.
out=$(python -u scripts/train_flow.py $FCOMMON --k 11 --space codes --force \
       --require_spectrum_ratio 10.0 2>&1 || true)
if grep -q "Refusing to train" <<<"$out"; then
    echo "  OK: refused a flat spectrum under --require_spectrum_ratio"
else
    echo "  FAIL: trained on a flat spectrum despite the gate"; exit 1
fi

echo; echo "FLOW SMOKE OK"

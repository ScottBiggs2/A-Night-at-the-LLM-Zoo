#!/bin/bash
# Score ONE arm across EVERY mixture in the plan, in one job.
#
#   SOURCE=soup  RUN=zoo_b015_mini_k99 K=50 ZOO=zoo_b015_mini_n100 \
#       ARCH=gpt2_zoo_mini OUT=gen/soup_mini.json sbatch slurm/eval_baseline.sbatch
#   SOURCE=trunk TRUNK_FLAT=$ARTIFACT_DIR/population/mini_b015/trunk_flat.npy \
#       ... OUT=gen/trunk_mini.json sbatch slurm/eval_baseline.sbatch
#   SOURCE=flow  FLOW_DIR=flow_k50_codes_pi_anchoronly N_SAMPLES=16 \
#       ... OUT=gen/anchoronly_mini.json sbatch slurm/eval_baseline.sbatch
#
# WHY THIS IS NOT eval_arm.sbatch
# eval_arm.sbatch scores one LOMO flow at the ONE mixture it was blinded to, read
# out of its own flow_meta.json. That is right for an 85-cell array and wrong for
# everything here: these arms answer the whole plan at once, so they take
# --targets all and there is no per-arm target to look up. One job, 85 cells.
#
# --targets all resolves to exactly the 85 distinct mixtures: holdout interior (4)
# + holdout vertex (1) + the 80 distinct pi among the 92 train members. Confirm the
# count in the log rather than trusting this comment.
#
# DETERMINISTIC ARMS. soup, quadratic and trunk produce one code per mixture, so
# N_SAMPLES=1 is not a shortcut, it is the whole arm; eval_generated.py forces it
# for trunk. Their CRPS degenerates to mean absolute error, which is correct --
# a deterministic forecast's CRPS IS its MAE -- but it earns none of the diversity
# credit the second CRPS term gives a real distribution, so do not read a
# deterministic arm's CRPS skill as if it were comparable to a flow's.
#
# The trunk is pi-independent: eval_generated.py scores it ONCE and fans the
# profile over every target, because 85 identical decodes would stream 1.75 TB at
# Mini to compute one number 85 times. It emits TWO arms, trunk and trunk_proj.

#SBATCH --job-name=eval_baseline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
CODE_DIR="${CODE_DIR:-$SLURM_SUBMIT_DIR}"
source "$CODE_DIR/slurm/env.sh"

SOURCE="${SOURCE:?set SOURCE=soup|trunk|flow|quadratic|gauss}"
RUN="${RUN:-zoo_b015_mini_k99}"
K="${K:-50}"
ZOO="${ZOO:-zoo_b015_mini_n100}"
ARCH="${ARCH:-gpt2_zoo_mini}"
TARGETS="${TARGETS:-all}"
N_SAMPLES="${N_SAMPLES:-1}"
COHORT="${COHORT:-16}"
ARM="${ARM:-$SOURCE}"
OUT="${OUT:?set OUT to a path relative to ARTIFACT_DIR}"
FLOW_DIR="${FLOW_DIR:-}"
TRUNK_FLAT="${TRUNK_FLAT:-}"

# --zoo_dir takes the ARCH directory, not its parent. Checked
# before a GPU is allocated and the torch import is paid for.
ZOO_DIR="$ARTIFACT_DIR/$ZOO/$ARCH"
if [ ! -f "$ZOO_DIR/zoo_meta.json" ]; then
    echo "no zoo_meta.json under $ZOO_DIR -- ZOO should be the directory that" >&2
    echo "CONTAINS $ARCH, e.g. ZOO=zoo_b015_mini_n100 ARCH=gpt2_zoo_mini." >&2
    exit 1
fi

# Expand optional arrays as ${A[@]+"${A[@]}"}: this runs `set -u`, and expanding
# an EMPTY array under `set -u` is fatal on bash < 4.4.
EXTRA=()
if [ "$SOURCE" = "flow" ]; then
    [ -n "$FLOW_DIR" ] || { echo "SOURCE=flow needs FLOW_DIR" >&2; exit 1; }
    [ -f "$ARTIFACT_DIR/runs/$RUN/$FLOW_DIR/flow_meta.json" ] || {
        echo "no flow_meta.json under runs/$RUN/$FLOW_DIR -- not a sealed arm" >&2
        exit 1; }
    EXTRA=(--flow_dir "$FLOW_DIR")
fi
if [ "$SOURCE" = "trunk" ]; then
    [ -n "$TRUNK_FLAT" ] || { echo "SOURCE=trunk needs TRUNK_FLAT" >&2; exit 1; }
    [ -f "$TRUNK_FLAT" ] || { echo "no trunk at $TRUNK_FLAT -- generate it with" \
        "scripts/diag_zoo_geometry.py --dump_trunk_flat" >&2; exit 1; }
    EXTRA=(--trunk_flat "$TRUNK_FLAT")
fi

OUT_JSON="$ARTIFACT_DIR/$OUT"

# ---- optional sharding over the 85 mixtures --------------------------------
#
# SHARDS=9 with --array=0-8 splits the target list so each task is a SHORT job.
# On a shared cluster a long reservation for a 1360-model sweep is hard to place,
# while nine ~20-minute tasks backfill into gaps and run in parallel.
#
# Sharding passes explicit --pi, so eval_generated labels rows pi_000.. PER SHARD
# and labels collide across shards. Harmless: every downstream join is on the
# requested pi (L1 < 1e-6), never on the label -- score_generated keys its cells
# by rounded pi. merge_gen.py then carries the shared exclusion ball through,
# because all shards come from one checkpoint.
#
# PI_MANIFEST is the OTHER way to choose targets: a manifest's target_pi column,
# one --pi per line. SHARDS slices the whole plan for throughput; PI_MANIFEST picks
# a named subset, which is what the 10-cell benchmark pilot needs so that member /
# pca_only / trunk are scored at exactly the cells the flow arm is scored at. The
# two are mutually exclusive -- sharding a hand-picked list of ten is pointless and
# would only make the shard->cell map another thing to get wrong.
SHARDS="${SHARDS:-0}"
PI_MANIFEST="${PI_MANIFEST:-}"
PI_ARGS=()
if [ -n "$PI_MANIFEST" ] && [ "$SHARDS" != "0" ]; then
    echo "PI_MANIFEST and SHARDS are mutually exclusive" >&2; exit 1
fi
if [ -n "$PI_MANIFEST" ]; then
    [ -f "$PI_MANIFEST" ] || { echo "no manifest at $PI_MANIFEST" >&2; exit 1; }
    # Column 6 is target_pi, written by scripts/make_flow_manifest.py. Word-splitting
    # each line is the point: it is P space-separated weights.
    mapfile -t PI_LINES < <(grep -v '^#' "$PI_MANIFEST" | cut -f6 | grep -v '^$')
    if [ "${#PI_LINES[@]}" -eq 0 ]; then
        echo "$PI_MANIFEST has no target_pi column" >&2; exit 1
    fi
    for line in "${PI_LINES[@]}"; do
        # shellcheck disable=SC2206
        PI_ARGS+=(--pi $line)
    done
    TARGETS=none
    echo "PI_MANIFEST $PI_MANIFEST -> ${#PI_LINES[@]} target mixtures"
fi
if [ "$SHARDS" != "0" ]; then
    IDX="${SLURM_ARRAY_TASK_ID:?SHARDS set but not submitted with --array}"
    OUT_JSON="${OUT_JSON%.json}_sh${IDX}.json"
    mapfile -t PI_LINES < <(python - "$ZOO_DIR/zoo_meta.json" "$SHARDS" "$IDX" <<'PY'
import json, sys
meta, shards, idx = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
z = json.load(open(meta))
members = {int(m["idx"]): m for m in z["members"]}
ho = z.get("holdout") or {}
# Exactly resolve_targets(--targets all): holdout interior, holdout vertex, then
# the distinct pi among the train members. Order is deterministic (dict insertion),
# so the shard map is stable across re-runs.
seen, out = set(), []
for group in ("interior", "vertex", "train"):
    for i in ho.get(group, []):
        m = members.get(i)
        if m is None:
            continue
        key = tuple(round(float(x), 9) for x in m["pi"])
        if key not in seen:
            seen.add(key)
            out.append(m["pi"])
# Contiguous slices, remainder spread over the first few shards.
n = len(out)
lo = (n * idx) // shards
hi = (n * (idx + 1)) // shards
for pi in out[lo:hi]:
    print(" ".join(repr(float(x)) for x in pi))
print(f"# shard {idx}/{shards}: targets {lo}..{hi - 1} of {n}", file=sys.stderr)
PY
)
    if [ "${#PI_LINES[@]}" -eq 0 ]; then
        echo "shard $IDX of $SHARDS is empty; nothing to do" >&2; exit 0
    fi
    for line in "${PI_LINES[@]}"; do
        # Word-splitting is the point: each line is one mixture's P weights.
        # shellcheck disable=SC2206
        PI_ARGS+=(--pi $line)
    done
    TARGETS=none
    echo "shard $IDX/$SHARDS -> ${#PI_LINES[@]} target mixtures"
fi

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

mkdir -p "$(dirname "$OUT_JSON")"

echo "=============================================="
echo "source   : $SOURCE   arm=$ARM"
echo "run      : $RUN  k=$K  arch=$ARCH"
echo "targets  : $TARGETS   samples=$N_SAMPLES   cohort=$COHORT"
echo "out      : $OUT_JSON"
echo "=============================================="

python -u scripts/eval_generated.py \
    --artifact_dir "$ARTIFACT_DIR" --run_name "$RUN" --k "$K" \
    --source "$SOURCE" \
    --zoo_dir "$ZOO_DIR" --arch "$ARCH" \
    --targets "$TARGETS" --n_samples "$N_SAMPLES" --cohort "$COHORT" \
    --arm "$ARM" --out "$OUT_JSON" \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"} \
    ${PI_ARGS[@]+"${PI_ARGS[@]}"}

echo "=== done -> $OUT_JSON ==="

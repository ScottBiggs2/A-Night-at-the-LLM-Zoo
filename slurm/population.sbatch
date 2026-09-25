#!/bin/bash
# Population statistics for one zoo: the imported src/llmzoo/analysis/ suite.
#
#   ZOO=zoo_b015_mini_n100/gpt2_zoo_mini  sbatch slurm/population.sbatch
#   ZOO=zoo_b015_small_n100/gpt2_zoo_small STAGE=all TIME=08:00:00 \
#       sbatch slurm/population.sbatch
#
# The `cpu` partition, deliberately -- same reasoning as conditioning.sbatch.
# Every analysis here is a streamed Gram over mmapped float32 .npy members: one
# BLAS call per chunk, no model is ever built, and nothing a GPU would help with.
#
# STAGE=trunk is the one exception and it is still not a GPU job: it flattens
# trunk.pt through flatten_model (a torch CPU import, seconds) so that
# mixture_composition gets a reference without torch-loading anything itself.
# It also seals the N x N displacement Gram, which is the artifact every
# branch-delta cosine is computed from offline afterwards.
#
# SIZING. One streaming pass reads the whole zoo: 100 x 51.5M x 4 B = 20.6 GB at
# Mini, 49.8 GB at Small, off $ARTIFACT_DIR. Five analyses each make their own pass, so
# budget wall time by read bandwidth, not by arithmetic. Peak RSS is one
# (N, chunk) float64 block -- 100 x 1M x 8 B = 0.8 GB -- plus mmap page cache,
# so 32G is honest and 64G is the ceiling that matters.

#SBATCH --job-name=population
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
CODE_DIR="${CODE_DIR:-$SLURM_SUBMIT_DIR}"
source "$CODE_DIR/slurm/env.sh"

ZOO="${ZOO:-zoo_b015_mini_n100/gpt2_zoo_mini}"
STAGE="${STAGE:-all}"
CHUNK="${CHUNK:-1048576}"
SKIP="${SKIP:-}"
TAG="${TAG:-$(basename "$(dirname "$ZOO")")}"
OUT="${OUT:-$ARTIFACT_DIR/population/$TAG}"

ZOO_DIR="$ARTIFACT_DIR/$ZOO"
if [ ! -f "$ZOO_DIR/zoo_meta.json" ]; then
    echo "No zoo_meta.json at $ZOO_DIR." >&2
    exit 1
fi

echo "=============================================="
echo "Zoo    : $ZOO_DIR"
echo "Stage  : $STAGE"
echo "Chunk  : $CHUNK elements"
echo "Skip   : ${SKIP:-<none>}"
echo "Out    : $OUT"
echo "=============================================="
# Members are the unit of work and a partial zoo is the failure mode that wastes
# a whole pass, so count them before spending four hours of read bandwidth.
echo "members on disk: $(find "$ZOO_DIR" -maxdepth 1 -name 'w_*.npy' | wc -l)"

# shellcheck disable=SC2086
python -u scripts/run_population_analysis.py \
    --zoo_dir "$ZOO_DIR" --out_dir "$OUT" \
    --stage "$STAGE" --chunk_size "$CHUNK" \
    ${SKIP:+--skip $SKIP}

echo
echo "Done. Copy down with:"
echo "  rsync -av --exclude '*.npy' <cluster>:$OUT/ reports/data/population/$TAG/"
echo "  rsync -av <cluster>:$OUT/displacement_gram.npy reports/data/population/$TAG/"

#!/bin/bash
# Branch the zoo off a trained trunk -- one array task per member.
#
#   ZOO_ROOT=$ARTIFACT_DIR/zoo_b030 ARCH=gpt2_zoo_mini BETA=0.30 \
#       sbatch --array=0-99 slurm/zoo_branch.sbatch
#   ZOO_ROOT=$ARTIFACT_DIR/zoo_b030 ARCH=gpt2_zoo_mini BETA=0.30 N_MEMBERS=12 \
#       sbatch --array=0-11 slurm/zoo_branch.sbatch          # beta calibration
#
# GIVE EVERY (arch, beta) ITS OWN ZOO_ROOT, and note WHY this is not cosmetic.
# The resume check below is `w_<i>.npy exists -> exit 0`, and it runs BEFORE the
# trunk's beta is ever read. So two betas sharing a directory used to hand back
# the FIRST beta's weights under the second beta's label -- silently, exit 0, on
# every member, invalidating a whole calibration with no error anywhere.
# train_zoo.py now also refuses a directory whose zoo_meta.json disagrees on
# beta/arch/n_members, so both layers have to be defeated to repeat that.
#
# ONE GPU PER TASK, and that is the whole point. Branches are independent, so
# Slurm backfills them across partially-free nodes and they start immediately.
# Throttle concurrency with --array=0-99%N to match your allocation. Submitting this as a single --gres=gpu:8 job would be slower AND queue longer.
#
# Re-running is safe and cheap: a member whose w_<i>.npy already exists exits
# immediately, so a partially-failed array is resubmitted verbatim.

# RESOURCE SHAPE, measured on 12 Mini probe branches:
#
#   MaxRSS   16.1-17.8 GB
#   AveCPU   ~0.87 cores    -> the data path tokenizes one document at a time,
#                              synchronously, in the training thread, so extra
#                              CPUs are idle
#
# Slurm must find the whole request free on one node, so an inflated --mem waits
# longer for no benefit.
#
# 96G is 5.4x Mini's observed peak and still covers Medium, whose extra cost is
# bounded: the model (1.42 GB), Adam's two moments, and the flattened w (1.42 GB)
# -- the 5 x 10k-document shuffle buffers dominate and do not scale with the
# model. Raise it if a Medium branch ever OOMs; do not raise it pre-emptively.
#SBATCH --job-name=zoo_branch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
CODE_DIR="${CODE_DIR:-$SLURM_SUBMIT_DIR}"
source "$CODE_DIR/slurm/env.sh"

ARCH="${ARCH:-gpt2_zoo_mini}"
BETA="${BETA:-0.30}"
N_MEMBERS="${N_MEMBERS:-100}"
EXTRA="${EXTRA:-}"

# Mid-run checkpointing. OFF unless CKPT_EVERY is set, and with it off the python
# command line below is byte-for-byte what it was before this existed.
#
# Expand as ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}, NOT "${CKPT_ARGS[@]}". This script
# runs `set -euo pipefail`, and on bash < 4.4 expanding an EMPTY array under `set
# -u` is a fatal "unbound variable". Test with `bash -euo pipefail -c`, never
# `srun bash -c`: that shell has no `set -u` and will not reproduce it.
CKPT_EVERY="${CKPT_EVERY:-0}"
CKPT_DIR="${CKPT_DIR:-$SCRATCH_DIR/zoo_ckpt}"
CKPT_ARGS=()
if [ "$CKPT_EVERY" -gt 0 ]; then
    CKPT_ARGS=(--ckpt_every "$CKPT_EVERY" --ckpt_dir "$CKPT_DIR")
    echo "checkpointing: every $CKPT_EVERY steps -> $CKPT_DIR"
fi
# Parent of the arch dir -- the same value `train_stack.py --zoo_dir` takes.
#
# RESOLVE AND VALIDATE IT HERE, never at submit time. sbatch exports the SUBMITTING
# shell's environment, and an interactive login shell has not sourced slurm/env.sh --
# so `ZOO_ROOT="$ARTIFACT_DIR/zoo_b015_medium_n100" sbatch ...` typed on the login
# node expands, right there, to the root-relative "/zoo_b015_medium_n100". ZOO_ROOT
# then arrives SET and NON-EMPTY, the :- default below never fires, and train_zoo.py
# dies ~6 s in on
#     PermissionError: [Errno 13] Permission denied: '/zoo_b015_medium_n100'
# and resubmitting the array verbatim resubmits the bug along with it.
#
# ZOO_NAME is the interface that cannot go wrong: it is joined to ARTIFACT_DIR
# INSIDE the job, after slurm/env.sh has run. Prefer it.
if [ -n "${ZOO_NAME:-}" ]; then
    ZOO_ROOT="$ARTIFACT_DIR/$ZOO_NAME"
fi
ZOO_ROOT="${ZOO_ROOT:-$ARTIFACT_DIR/zoo}"
case "$ZOO_ROOT" in
    "$ARTIFACT_DIR"/*) ;;
    *)
        echo "FATAL: ZOO_ROOT='$ZOO_ROOT' is not under ARTIFACT_DIR='$ARTIFACT_DIR'." >&2
        echo "" >&2
        echo "This almost always means \$ARTIFACT_DIR was EMPTY in the shell that ran" >&2
        echo "sbatch (a login shell does not source slurm/env.sh). Resubmit as" >&2
        echo "" >&2
        echo "    ZOO_NAME=zoo_b015_medium_n100 ARCH=... BETA=... \\" >&2
        echo "        sbatch --array=... slurm/zoo_branch.sbatch" >&2
        echo "" >&2
        echo "Refusing rather than creating a directory outside the artifact tree." >&2
        exit 1
        ;;
esac
ZOO_DIR="$ZOO_ROOT/$ARCH"
IDX="${SLURM_ARRAY_TASK_ID:?this script must be submitted with --array}"

echo "=== zoo branch: arch=$ARCH beta=$BETA member=$IDX -> $ZOO_DIR ==="
python -u scripts/train_zoo.py --mode branch --arch "$ARCH" --beta "$BETA" \
       --n_members "$N_MEMBERS" --member_idx "$IDX" --zoo_dir "$ZOO_DIR" \
       ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"} $EXTRA
echo "=== done ==="

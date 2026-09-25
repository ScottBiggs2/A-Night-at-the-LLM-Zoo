#!/bin/bash
# Launch ONE beta arm of the zoo as a dependency chain:
#
#     trunk  ->  branch array  ->  { domain-separation gate , k=N-1 spectrum }
#
# Run from the repository root on a login node (not inside a job). It reads
# ARTIFACT_DIR / SCRATCH_DIR from slurm/site.sh.
#
#   bash slurm/launch_beta_arm.sh 0.30
#   bash slurm/launch_beta_arm.sh 0.15 12 gpt2_zoo_mini
#   TRUNK_TIME=04:00:00 BRANCH_TIME=02:00:00 bash slurm/launch_beta_arm.sh 0.60
#
# WHY THIS IS A SCRIPT AND NOT THREE TYPED COMMANDS
#
# Every beta needs its own ZOO_ROOT, and getting that wrong is silent rather than
# loud. The branch resume check is `w_<i>.npy exists -> exit 0` and it runs before
# the trunk's beta is ever read, so two betas sharing a directory hand back the
# first beta's weights under the second beta's label, on every member, with
# exit 0. Deriving ZOO_ROOT from BETA in one place removes the chance to mistype
# it in three.
#
# afterok, not afterany: a failed trunk must not spawn 12 doomed branches, and the
# gate must not score a half-built zoo. A gate that runs and exits 1 is a RESULT
# (raise beta); a gate that never runs is a build failure. Keep them distinguishable.
#
# Re-running is safe and cheap. A member whose w_<i>.npy exists exits immediately,
# so resubmit this verbatim after a partial array.

set -euo pipefail

BETA="${1:?usage: launch_beta_arm.sh <beta> [n_members] [arch]}"
N_MEMBERS="${2:-12}"
ARCH="${3:-gpt2_zoo_mini}"

CODE_DIR="${CODE_DIR:-$(pwd)}"
SITE_FILE="${SITE_FILE:-$CODE_DIR/slurm/site.sh}"
if [ ! -f "$SITE_FILE" ]; then
    echo "ERROR: no $SITE_FILE. Run this from the repository root, after copying" >&2
    echo "slurm/site.example.sh to slurm/site.sh and editing it." >&2
    exit 1
fi
# shellcheck disable=SC1090
. "$SITE_FILE"
ART="${ARTIFACT_DIR:?set ARTIFACT_DIR in slurm/site.sh}"
: "${SCRATCH_DIR:?set SCRATCH_DIR in slurm/site.sh}"

# 0.30 -> b030, so the directory name is unambiguous in a listing and sorts.
TAG="b$(printf '%03d' "$(python3 -c "print(round(float('$BETA')*100))")")"

# ZOO_ROOT is keyed on (beta, ARCH, N), not on beta alone, and every part of that
# is load-bearing:
#
#   * ARCH, because the three scales are separate zoos that happen to share a
#     beta. They land in different <root>/<arch> subdirs either way, so this is
#     belt-and-braces -- but the RUN NAME below is not, see there.
#   * N_MEMBERS, because train_zoo.py refuses a directory whose zoo_meta.json
#     disagrees on n_members. An N=12 calibration and an N=100 zoo at the same
#     beta sharing a directory would have EVERY branch exit 2 while the trunk
#     step reported "nothing to do".
#
# Overridable: pass ZOO_ROOT explicitly to reuse an existing directory.
ZOO_ROOT="${ZOO_ROOT:-$ART/zoo_${TAG}_${ARCH#gpt2_zoo_}_n${N_MEMBERS}}"

# An EXPLICIT ZOO_ROOT is the one that has actually gone wrong, so validate it.
# `ZOO_ROOT="$ARTIFACT_DIR/zoo_b015_medium_n100" bash slurm/launch_beta_arm.sh ...`
# in a shell where ARTIFACT_DIR is not yet set expands to the root-relative
# "/zoo_b015_medium_n100" AT TYPE TIME -- set and non-empty, so the :- default
# above never fires. Every branch would then die ~6 s in on
#     PermissionError: [Errno 13] Permission denied: '/zoo_b015_medium_n100'
# $ART is checked above and cannot be empty here; ZOO_ROOT can.
# slurm/zoo_branch.sbatch carries the same guard, because it is also submitted
# directly. Refuse rather than mkdir outside the artifact tree.
case "$ZOO_ROOT" in
    "$ART"/*) ;;
    *)
        echo "FATAL: ZOO_ROOT='$ZOO_ROOT' is not under ARTIFACT_DIR='$ART'." >&2
        echo "" >&2
        echo "If \$ARTIFACT_DIR was empty in your shell, \"\$ARTIFACT_DIR/zoo_...\"" >&2
        echo "expanded to \"/zoo_...\" before this script saw it." >&2
        echo "Drop the ZOO_ROOT override and let it be derived from (beta, arch, N)," >&2
        echo "or pass an absolute path under $ART." >&2
        exit 1
        ;;
esac

# --time is sized for Mini. Small needs ~4x and Medium ~50x; override with the
# env vars rather than editing the sbatch, so the committed default stays honest
# about which scale it was measured for.
TRUNK_TIME="${TRUNK_TIME:-02:00:00}"
BRANCH_TIME="${BRANCH_TIME:-01:00:00}"
# The spectrum fit streams every member TWICE (mean pass, then Gram pass) off
# shared storage: 12 x 206 MB = 2.5 GB at Mini N=12, but 100 x 1.42 GB = 142 GB
# per pass at Medium N=100.
SPEC_TIME="${SPEC_TIME:-02:00:00}"
# The gate rebuilds each member in place and runs 5 domains x 64 blocks of forwards, so it grows with
# both D and N.
GATE_TIME="${GATE_TIME:-01:30:00}"

# Mid-run checkpointing for the trunk and the branches. 0 = OFF, which is the
# default and is what Mini and Small run. Turn it on for spans long enough that
# losing one hurts, or that exceed your site's maximum walltime: Medium's trunk is
# ~16.5 h on one GPU.
CKPT_EVERY="${CKPT_EVERY:-0}"

# W&B project, one per workstream: `llmzoo-genverify` is the generative-verification
# project and this is its zoo-training sibling.
WANDB_PROJECT="${WANDB_PROJECT:-llmzoo-zoo}"
export WANDB_PROJECT

# Skip the trunk job when the trunk has already been staged into ZOO_ROOT (which
# is valid: total_steps/trunk_steps depend on tokens_per_param, batch,
# grad_accum, n_ctx and beta -- NOT on n_members, so a 12-member trunk serves a
# 100-member plan). The staging step must then have written zoo_plan.json too,
# because that is what zoo_trunk.sbatch would otherwise have done.
SKIP_TRUNK="${SKIP_TRUNK:-0}"

# An ALREADY-SUBMITTED trunk job to chain onto, instead of submitting one here.
#
# Distinct from SKIP_TRUNK, and the difference is the point: SKIP_TRUNK=1 asserts the
# trunk is already ON DISK and refuses if it is not, whereas TRUNK_JOB says a trunk is
# still RUNNING under that job id and the array should wait for it with afterok. That
# matters because the honest BRANCH_TIME is only knowable once the trunk has reported
# a few hundred steps of ktok/s -- so the useful sequence is: launch the trunk alone,
# read its throughput, then queue the array behind it with a measured walltime instead
# of sitting idle for hours or guessing.
TRUNK_JOB="${TRUNK_JOB:-}"

# Concurrent array tasks. Set it to the number of GPUs you can hold at once; the
# wave count (echoed below) is what sets wall clock. At N=100: 32 -> 4 waves,
# 10 -> 10 waves.
THROTTLE="${THROTTLE:-32}"

# Extra flags forwarded to train_zoo.py (trunk AND branch) and, separately,
# to eval_domains.py (e.g. EXTRA="--alpha 8.0").
EXTRA="${EXTRA:-}"
GATE_EXTRA="${GATE_EXTRA:-}"

# Where the trunk and branches run. Note --peak_tflops (train_zoo.py) sets the
# denominator of the sealed MFU percentage; set it for your GPU. ktok/s stands
# either way.
PARTITION="${PARTITION:-gpu}"

# Where the GATE and the SPECTRUM run. Both want one GPU for a few minutes, so a
# separate, less-contended partition helps if your site has one.
EVAL_PARTITION="${EVAL_PARTITION:-gpu}"

# Nodes to keep off, e.g. a node with a known hardware fault (comma-separated).
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
EXC=()
if [ -n "$EXCLUDE_NODES" ]; then EXC=(--exclude="$EXCLUDE_NODES"); fi

LAST=$((N_MEMBERS - 1))
K=$((N_MEMBERS - 1))
# Which array indices to submit. The default submits every member, which is the
# documented and tested path: a member whose w_<i>.npy exists exits immediately,
# so resubmitting verbatim after a partial array is safe and costs one no-op per
# finished member. Override to skip a staged prefix (ARRAY=12-99) or to retry a
# handful (ARRAY=17,43,81).
ARRAY="${ARRAY:-0-$LAST}"
SLUG="${ARCH#gpt2_zoo_}"
RUN_NAME="zoo_${TAG}_${SLUG}_k$K"
# The W&B group for this whole zoo. train_zoo.py and eval_domains.py each build this
# same string internally, but train_stack.py has NO fallback -- it passes
# `group=args.wandb_group` straight through -- so it is sent explicitly. Must match
# train_zoo.py's group exactly, including N.
WB_GROUP="zoo_${ARCH}_${TAG}_n${N_MEMBERS}"
WAVES=$(( (N_MEMBERS + THROTTLE - 1) / THROTTLE ))

echo "=================================================================="
echo " beta arm      : BETA=$BETA  ARCH=$ARCH  N=$N_MEMBERS"
echo " zoo root      : $ZOO_ROOT"
echo " spectrum run  : runs/$RUN_NAME   (k=$K)"
echo " partition     : $PARTITION   (gate+spec: $EVAL_PARTITION)"
echo " walltimes     : trunk=$TRUNK_TIME  branch=$BRANCH_TIME  spec=$SPEC_TIME"
echo " array         : $ARRAY   ($THROTTLE at a time -> ~$WAVES wave(s))"
echo " skip trunk    : $SKIP_TRUNK"
echo " exclude nodes : ${EXCLUDE_NODES:-<none>}"
echo " checkpointing : CKPT_EVERY=$CKPT_EVERY$([ "$CKPT_EVERY" = "0" ] && echo '  (OFF)')"
echo " wandb         : project=$WANDB_PROJECT"
echo "                 group=$WB_GROUP"
echo "=================================================================="

# A long span with no checkpointing is the expensive mistake this warns about.
# Keyed on the actual walltimes rather than on the arch string, so it fires for any
# configuration that has grown long enough to matter.
if [ "$CKPT_EVERY" = "0" ] \
   && { [ "${TRUNK_TIME%%:*}" -ge 8 ] || [ "${BRANCH_TIME%%:*}" -ge 8 ]; }; then
  echo "WARNING: CKPT_EVERY=0 with TRUNK_TIME=$TRUNK_TIME BRANCH_TIME=$BRANCH_TIME." >&2
  echo "         A walltime kill or a bad node loses the WHOLE span -- there is no" >&2
  echo "         resume below whole-member granularity. Suggest CKPT_EVERY=1000." >&2
fi

cd "$CODE_DIR"
# sbatch --output=logs/... is relative to the submit directory and Slurm does not
# create it.
mkdir -p logs

# Refuse a mismatched directory HERE rather than 100 times inside the array.
# train_zoo.py's own guard is correct but per-member, so a mistyped root
# means N failed submissions and N log files instead of one message.
META="$ZOO_ROOT/$ARCH/zoo_meta.json"
if [ -f "$META" ]; then
  python3 - "$META" "$BETA" "$ARCH" "$N_MEMBERS" <<'PYCHECK'
import json, sys
meta, beta, arch, n = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
z = json.load(open(meta))
bad = []
if z.get("beta") is not None and abs(float(z["beta"]) - beta) > 1e-9:
    bad.append(f"beta={z['beta']!r} (asked {beta!r})")
if z.get("arch") not in (None, arch):
    bad.append(f"arch={z['arch']!r} (asked {arch!r})")
if z.get("n_members") not in (None, n):
    bad.append(f"n_members={z['n_members']!r} (asked {n!r})")
if bad:
    print(f"REFUSING: {meta} already describes a zoo with " + ", ".join(bad))
    print("Give this configuration its own ZOO_ROOT. Mixing them would relabel")
    print("already-written members, and the w_<i>.npy resume check returns 0")
    print("before the trunk's beta is ever read.")
    sys.exit(1)
print(f"  existing zoo_meta.json agrees (beta={z.get('beta')} "
      f"arch={z.get('arch')} n_members={z.get('n_members')}); resuming into it")
PYCHECK
fi

if [ -n "$TRUNK_JOB" ]; then
  if [ "$SKIP_TRUNK" = "1" ]; then
    echo "Set TRUNK_JOB or SKIP_TRUNK, not both: one waits for a trunk, the other" >&2
    echo "asserts one already exists." >&2
    exit 1
  fi
  t="$TRUNK_JOB"
  echo "trunk      : $t  (ALREADY SUBMITTED; the array chains onto it with afterok)"
elif [ "$SKIP_TRUNK" = "1" ]; then
  if [ ! -f "$ZOO_ROOT/$ARCH/trunk.pt" ]; then
    echo "SKIP_TRUNK=1 but no trunk at $ZOO_ROOT/$ARCH/trunk.pt" >&2
    exit 1
  fi
  if [ ! -f "$ZOO_ROOT/$ARCH/zoo_plan.json" ]; then
    echo "SKIP_TRUNK=1 but no zoo_plan.json at $ZOO_ROOT/$ARCH/ -- the staging" >&2
    echo "step must write the plan, since zoo_trunk.sbatch is what normally does." >&2
    exit 1
  fi
  t=""
  echo "trunk      : SKIPPED (staged trunk.pt + zoo_plan.json already present)"
else
  t=$(ZOO_ROOT="$ZOO_ROOT" ARCH="$ARCH" BETA="$BETA" N_MEMBERS="$N_MEMBERS" \
      EXTRA="$EXTRA" CKPT_EVERY="$CKPT_EVERY" \
      sbatch --parsable --partition="$PARTITION" --time="$TRUNK_TIME" \
             --job-name="trunk_$TAG" ${EXC[@]+"${EXC[@]}"} \
             slurm/zoo_trunk.sbatch)
  echo "trunk      : $t"
fi

# One GPU per array task, always. Branches are independent, and on a shared
# cluster a 1-GPU job starts far sooner than a whole-node request. %N throttles
# concurrency to what your allocation allows.
#
# Optional arrays are expanded as ${A[@]+"${A[@]}"}: under `set -u`, expanding an
# EMPTY array is fatal on bash < 4.4.
DEP=()
if [ -n "$t" ]; then DEP=(--dependency="afterok:$t"); fi
b=$(ZOO_ROOT="$ZOO_ROOT" ARCH="$ARCH" BETA="$BETA" N_MEMBERS="$N_MEMBERS" \
    EXTRA="$EXTRA" CKPT_EVERY="$CKPT_EVERY" \
    sbatch --parsable --partition="$PARTITION" --time="$BRANCH_TIME" \
           --job-name="branch_$TAG" \
           --array="$ARRAY%$THROTTLE" ${DEP[@]+"${DEP[@]}"} ${EXC[@]+"${EXC[@]}"} \
           slurm/zoo_branch.sbatch)
echo "branches   : $b  (array $ARRAY%$THROTTLE, ~$WAVES wave(s))"

# The domain-separation gate.
g=$(ZOO_ROOT="$ZOO_ROOT" ARCH="$ARCH" EXTRA="$GATE_EXTRA" \
    sbatch --parsable --job-name="gate_$TAG" --dependency="afterok:$b" \
           --time="$GATE_TIME" --partition="$EVAL_PARTITION" ${EXC[@]+"${EXC[@]}"} \
           slurm/eval_domains.sbatch)
echo "gate       : $g  ($EVAL_PARTITION, exit 1 = gate fired)"

# The k=N-1 spectrum. Four flags and every one is load-bearing:
#   --source zoo / --zoo_dir   point at the zoo at all (parent of the arch dir)
#   --noise_scale 0.0          EnsembleDataset refuses a zoo with augmentation noise
#   --no_exclude_1d            train_zoo writes exclude_1d=False; the default
#                              here is True and the cache gate refuses a mismatch
# The run name carries the ARCH: every N=100 scale has k=99, so without it Mini,
# Small and Medium would all write runs/zoo_<tag>_k99 and the later fits would
# silently overwrite the earlier ones.
s=$(sbatch --parsable --job-name="spec_$TAG" --dependency="afterok:$b" \
           ${EXC[@]+"${EXC[@]}"} \
           --partition="$EVAL_PARTITION" \
           --nodes=1 --gres=gpu:1 \
           --cpus-per-task=8 --mem=200G --time="$SPEC_TIME" \
           --output="logs/spec_$TAG-%j.out" \
           --error="logs/spec_$TAG-%j.err" \
           --wrap "set -euo pipefail
                   export CODE_DIR='$CODE_DIR'
                   source '$CODE_DIR/slurm/env.sh'
                   python -u scripts/train_stack.py \
                     --arch_list $ARCH --mode full \
                     --run_name $RUN_NAME --artifact_dir \$ARTIFACT_DIR \
                     --wandb_group $WB_GROUP --wandb_project $WANDB_PROJECT \
                     --source zoo --zoo_dir '$ZOO_ROOT' \
                     --n_samples $N_MEMBERS --k $K \
                     --noise_scale 0.0 --no_exclude_1d")
echo "spectrum   : $s  ($EVAL_PARTITION, k=$K)"

echo
echo "watch:  squeue -u \$USER"
echo "gate  : $ZOO_ROOT/$ARCH/domain_separation.json"
echo "spec  : \$ARTIFACT_DIR/runs/$RUN_NAME/pipeline_summary_k$K.json"
echo "        -> per_arch.$ARCH.spectrum_ev0_over_median"
echo "        -> per_arch.$ARCH.spectrum_effective_rank_ratio"
echo "        NOT spectrum_ev0_over_evlast (reads 12.09 on pure-noise data"
echo "        whose true ratio is 1.01)"

# Sourced by every job script:  source "$CODE_DIR/slurm/env.sh"
#
# Jobs are submitted from the repository root, so CODE_DIR defaults to
# $SLURM_SUBMIT_DIR. Site-specific paths live in slurm/site.sh (see
# slurm/site.example.sh).

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SITE_FILE="${SITE_FILE:-$CODE_DIR/slurm/site.sh}"
if [ -f "$SITE_FILE" ]; then
  . "$SITE_FILE"
else
  echo "ERROR: no $SITE_FILE -- copy slurm/site.example.sh to slurm/site.sh and edit it." >&2
  exit 1
fi
: "${ARTIFACT_DIR:?set ARTIFACT_DIR in slurm/site.sh}"
S="${SCRATCH_DIR:?set SCRATCH_DIR in slurm/site.sh}"

if declare -F activate_env >/dev/null; then activate_env; fi
cd "$CODE_DIR" || { echo "ERROR: CODE_DIR not found: $CODE_DIR" >&2; exit 1; }
mkdir -p "$S/logs" "$S/hf_cache/datasets" "$S/triton_cache" "$S/wandb" "$ARTIFACT_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Secrets are read from a file outside the repository. Only their lengths are
# printed, never their values.
if [ -n "${LLMZOO_ENV_FILE:-}" ] && [ -f "$LLMZOO_ENV_FILE" ]; then
  . "$LLMZOO_ENV_FILE"
fi
_wandb_len="${WANDB_API_KEY:-}"
_hf_len="${HF_TOKEN:-}"
echo "secrets: WANDB_API_KEY len=${#_wandb_len}  HF_TOKEN len=${#_hf_len}"
unset _wandb_len _hf_len

# HF_HOME redirects the token lookup away from ~/.cache/huggingface, so the token
# must come from HF_TOKEN (above) rather than a prior `huggingface-cli login`.
export HF_HOME="$S/hf_cache"
export HF_DATASETS_CACHE="$S/hf_cache/datasets"
export TRITON_CACHE_DIR="$S/triton_cache"
export ARTIFACT_DIR

if grep -qs "api.wandb.ai" "$HOME/.netrc" || [ -n "${WANDB_API_KEY:-}" ]; then
  export WANDB_MODE="${WANDB_MODE:-online}"
else
  export WANDB_MODE="${WANDB_MODE:-offline}"   # then: wandb sync "$WANDB_DIR/<run>"
fi
export WANDB_DIR="${WANDB_DIR:-$S/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-llmzoo}"

python -c "import llmzoo" 2>/dev/null || {
  echo "ERROR: llmzoo is not importable. Run: pip install -e $CODE_DIR" >&2
  exit 1
}

echo "--- env ---"
echo "host=$(hostname)  job=${SLURM_JOB_ID:-local}  part=${SLURM_JOB_PARTITION:-local}"
echo "cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}M  gpus=${SLURM_GPUS_ON_NODE:-0}"
echo "code=$CODE_DIR  artifacts=$ARTIFACT_DIR  scratch=$S"
echo "wandb=$WANDB_MODE  project=$WANDB_PROJECT"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'gpus',torch.cuda.device_count(),[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])" 2>&1
echo "-----------"

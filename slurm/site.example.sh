# Site configuration, sourced by slurm/env.sh in every job.
# Copy to slurm/site.sh (gitignored) and edit.

# Persistent storage for zoos, PCA runs, flows and evaluations.
# Medium alone is ~150 GB of member weights.
export ARTIFACT_DIR="/path/to/persistent/llmzoo_artifacts"

# Fast scratch for the HF dataset cache, W&B files, logs and mid-run checkpoints.
export SCRATCH_DIR="/path/to/scratch/$USER"

# How to activate the Python environment that has torch + `pip install -e .`
# e.g.  source /opt/miniforge3/etc/profile.d/conda.sh && conda activate llmzoo
#       source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
activate_env () {
    source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
}

# Optional: secrets file with `export HF_TOKEN=...`, `export WANDB_API_KEY=...`,
# `export WANDB_ENTITY=...`. Keep it outside the repository.
# export LLMZOO_ENV_FILE="$HOME/.config/llmzoo/env"

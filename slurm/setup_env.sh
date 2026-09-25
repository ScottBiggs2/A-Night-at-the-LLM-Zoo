#!/bin/bash
# One-time environment setup: a conda env with a CUDA torch, then this package.
#
#   ENV_PATH=/path/to/envs/llmzoo bash slurm/setup_env.sh
#
# torch comes FIRST and from the CUDA index; `pip install -e .` never declares it,
# so it cannot pull a default-index wheel over the CUDA build.
set -euo pipefail

ENV_PATH="${ENV_PATH:?set ENV_PATH to where the conda env should live}"
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

if [ ! -d "$ENV_PATH" ]; then
    conda create -y -p "$ENV_PATH" python=3.11
fi
eval "$(conda shell.bash hook)"
conda activate "$ENV_PATH"

pip install --index-url "$TORCH_INDEX" "torch==$TORCH_VERSION"
pip install -e "$CODE_DIR[plots]"

python - <<'PY'
import torch, transformers, datasets, numpy, llmzoo
print("torch       ", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("datasets    ", datasets.__version__)
print("numpy       ", numpy.__version__)
assert transformers.__version__.startswith("4."), "pyproject pins transformers<5"
assert datasets.__version__.startswith("3."), "pyproject pins datasets<4"
print("llmzoo      ", llmzoo.__file__)
PY

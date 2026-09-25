# Figure 2: geometry of domain effects

```bash
python geometry.py --checkpoint-path /path/to/zoo/gpt2_zoo_mini --output-dir results
python geometry.py --config /path/to/tiny_gpt2_config.json --output-dir random_results
```

Average the repeated runs for each domain, center the D domain means, and form
the P-by-D domain-effect matrix A conceptually. Accumulate the small Gram matrix
`A.T @ A` in float64 chunks, without materializing full model-sized bases.
Its eigenvalues are the squared singular values of A.

`geometry.csv/json` report the D-1 possible eigenvalues, their explained-variance
fractions, k90, and the participation-ratio effective rank:

```text
p_j = eigenvalue_j / sum(eigenvalues)
effective_rank = 1 / sum(p_j²)
```

Between/within sums are also reported to identify layers with weak domain signal.
Finite-run noise remains in the estimated means. At most D-1 directions exist
by construction: with five domains, finding rank <= 4 is not itself a discovery.
No stability intervals or noise-corrected spectra are claimed. Zero variation
produces an undefined effective rank, not a claim of rank one.

Layer and whole-model spectra are obtained by summing tensor Gram matrices before
eigendecomposition. They are not averages of tensor spectra. This analyzes
flattened domain effects, not UniSub input/output matrix-mode subspaces.
`figure.svg` plots effective rank across transformer layers.

Tests: `python test_geometry.py` (NumPy only).

## Model and input options

With no input path, the script runs an explicitly labeled random-initialization
control: independent GPT-2 initializations, five artificial domain groups and four
models per group. These models have not seen any data and do not represent
shared-trunk training. Default architecture is `gpt2_zoo_mini`.

- `--checkpoint-path DIR` (aliases `--dataset-path`, `--zoo-dir`): a saved zoo
  architecture directory containing `zoo_meta.json` and `w_<index>.npy`.
- `--arch gpt2_zoo_mini|gpt2_zoo_small|gpt2_zoo_medium`: random-control architecture;
  trained inputs use the architecture recorded in metadata.
- `--config FILE`: optional Hugging Face GPT2Config JSON for initialization or
  validating a trained input's schema.
- `--seed 1234`: random-control seed. Models use `seed + 1 + member_idx`.
- `--chunk-size 65536`: maximum coordinates processed together.
- `--work-dir DIR`: temporary random-weight storage parent; Mini needs several GB.
- `--output-dir DIR`: CSV, JSON and `figure.svg` destination.

Raw text datasets are unnecessary. Domain groups and mixture coefficients come
from the saved metadata. All required anchors must exist; missing files never
silently become random models. Keep each architecture/beta/trunk separate.

NumPy suffices for saved `.npy` analysis. Initialization or `.pt` restoration needs
PyTorch and `transformers>=4.57,<5`. No `llmzoo`, downloads, training, or plotting
packages are needed. Run PyTorch operations on cluster compute as required by
this repository.

Each folder is independent, including its local `zoo.py`. From inside the folder:

```python
from zoo import load_model

random_model = load_model("gpt2_zoo_mini")
member = load_model(checkpoint_path="/path/to/zoo/w_0.npy")
trunk = load_model(checkpoint_path="/path/to/zoo/trunk.pt")
```

Initialization matches zoo pretraining: GPT-2 vocabulary 50257, context 1024,
zero dropout, GELU, tied embeddings and HF residual-projection initialization.
Use matching library versions for reproducibility. The `.npy` loader requires
adjacent metadata and restores extra parameters followed by transformer blocks.
Omitted parameters remain initialized when reconstructing a partial checkpoint;
analysis includes only saved parameters. Only load trusted `.pt` files (pickle).

Outputs retain individual tensors, layer aggregates, and the whole saved model.
Tied embeddings are counted once. GPT-2 Q/K/V stays in its saved fused
`attn.c_attn` tensor. Layer plots use zero-based indices. Undefined fractions are
JSON null / blank CSV cells. Output files in the specified directory are replaced.

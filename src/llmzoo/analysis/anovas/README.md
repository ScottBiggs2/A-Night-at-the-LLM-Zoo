# Figure 1: within-domain and between-domain ANOVA

```bash
python anova.py --checkpoint-path /path/to/zoo/gpt2_zoo_mini --output-dir results
python anova.py --config /path/to/tiny_gpt2_config.json --output-dir random_results
```

For D domains with S repeated models, compute domain means m_d and grand mean m:

```text
SS_between = S * sum_d ||m_d - m||²
SS_within  = sum_d,s ||w_d,s - m_d||²
SS_total   = sum_d,s ||w_d,s - m||²
```

The exact identity `SS_total = SS_between + SS_within` is checked. The script
selects repeated one-hot anchors and excludes singleton mixtures. It requires a
balanced design, with at least two runs per domain. Subtracting a common trunk
cancels from these centered quantities; no trunk is needed.

`anova.csv/json` report raw sums and observed between/within fractions. Fractions
are descriptive, not causal attribution. Layer/whole-model aggregates sum squared
variation before dividing, rather than averaging tensor fractions.

A fixed-domain noise correction is also included:

```text
MS_within = SS_within / (D * (S - 1))
V_domain  = (SS_between - (D - 1) * MS_within) / (D * S)
R_domain  = V_domain / (V_domain + MS_within)
```

It assumes independent repetitions and common within-domain covariance. Negative
corrected estimates are retained. No p-values or uncertainty intervals are claimed.
`figure.svg` plots observed between/within fractions across transformer layers.

Tests: `python test_anova.py` (NumPy only).

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

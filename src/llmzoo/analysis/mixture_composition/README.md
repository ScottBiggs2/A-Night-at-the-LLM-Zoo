# Figure 3: mixture composition from domain experts

```bash
python composition.py --checkpoint-path /path/to/zoo/gpt2_zoo_mini \
  --members 20 21 22 --output-dir results
python composition.py --config /path/to/tiny_gpt2_config.json \
  --random-mixtures 4 --output-dir random_results
```

Average each domain's repeated anchors. For each selected singleton mixture pi,
predict `w_pred = sum_d pi_d * mean_d` and measure `r = w_mix - w_pred`.
The common trunk cancels from this prediction, since the weights sum to one.
No expert-merged checkpoint is trained or saved.

Project r onto the span of **centered** domain means:

- `parallel_squared`: mismatch within existing domain directions (coefficient error).
- `perpendicular_squared`: mismatch outside that estimated affine domain space.
- `residual_squared`: their sum, the total prediction error.

All reliably resolved domain directions are retained. `--rank-rtol 1e-8` controls
the relative Gram eigenvalue cutoff; the retained projection rank is recorded.
Layer/whole-model projections use aggregated Gram/cross-product statistics and
therefore project in that full region, not separately in each tensor.

`residual_to_noise` is the square root of residual energy divided by
`MS_within * (1 + sum(pi²)/S)`. This reference includes uncertainty in the expert
means, but assumes independent repetitions and the same covariance for anchors
and mixtures. It is not a significance test. A large parallel error alone does
not imply new directions; perpendicular errors also include finite-sample noise.

Additional options:

- `--members 20 21 ...`: selected singleton IDs; default all saved singletons.
- `--reference-checkpoint FILE`: compatible flat `.npy` or trusted zoo `trunk.pt`.
  The local `trunk.pt` is detected automatically. Enables
  `relative_residual = ||r|| / ||w_mix - w_reference||`. Without a reference,
  this metric is null; all other metrics remain available. An explicitly supplied
  reference must belong to the intended experiment. In random mode the reference
  is a separate untrained initialization, not a shared trained trunk.
- `--token-weights FILE`: JSON mapping member IDs to measured token proportions,
  e.g. `{"20": [0.4, 0.3, 0.1, 0.1, 0.1]}` in metadata domain order. Defaults to
  saved document-sampling proportions, which need not equal token exposure.
- `--random-mixtures 4`: number of independently initialized mixture controls;
  artificial coefficients are seeded Dirichlet(1) draws.

`composition.csv/json` retain each member's per-tensor/layer/model results.
`figure.svg` plots the mean residual/noise ratio over selected members per layer.
The JSON records the coefficient source and reference. No functional model
performance or statistical significance is inferred from weight distances.

Tests: `python test_composition.py` (NumPy only).

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

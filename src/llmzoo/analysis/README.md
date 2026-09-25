# Three standalone weight-space analyses

| Folder | Question | Entry point |
|---|---|---|
| `anovas` | How much variation is between domains versus within a domain? | `anova.py` |
| `domain_geometry` | How concentrated is the spectrum of centered domain means? | `geometry.py` |
| `mixture_composition` | Do domain experts predict mixture models, and where does prediction fail? | `composition.py` |

Each folder can be copied and run independently. Its local `zoo.py` contains the
same loader and input utilities; no imports from `llmzoo` or sibling folders.

## Inputs shared by all three

With **no input path**, each entry point initializes independent random GPT-2
models using the same configuration and Hugging Face initialization procedure as
the zoo. Defaults: Mini (8 layers, width 512), five artificial domain labels,
four models per label. Composition adds four artificial mixtures. These are
**untrained null controls**, not shared-trunk pretrained branches or evidence of
dataset effects. Random models use seeds `seed + 1 + member_idx`; a separate
reference initialization uses `seed` (default 1234).

Models are created sequentially on CPU and flattened to temporary files. Default
Mini uses roughly 4.3 GB of temporary weights for ANOVA/geometry and 5.1 GB for
composition, plus model and chunk memory. `--work-dir` chooses the temporary-file
parent; files are removed after the run. Use a small GPT2Config JSON with
`--config` for inexpensive initialization checks.

```bash
python analysis/anovas/anova.py --output-dir results/random_anova
python analysis/domain_geometry/geometry.py --output-dir results/random_geometry
python analysis/mixture_composition/composition.py --output-dir results/random_composition
```

To analyze trained models, pass a **single architecture's zoo directory**:

```bash
python analysis/anovas/anova.py \
  --checkpoint-path /path/to/zoo/gpt2_zoo_mini --output-dir results/mini_anova
python analysis/domain_geometry/geometry.py \
  --checkpoint-path /path/to/zoo/gpt2_zoo_mini --output-dir results/mini_geometry
python analysis/mixture_composition/composition.py \
  --checkpoint-path /path/to/zoo/gpt2_zoo_mini --members 20 21 22 \
  --output-dir results/mini_composition
```

`--dataset-path` and `--zoo-dir` are aliases for `--checkpoint-path`: they refer to
the saved **weight dataset**, with `zoo_meta.json` and `w_<index>.npy`, not raw text.
These analyses need no training-text dataset or tokenizer. Keep architectures,
betas, and shared trunks separate. Missing checkpoints never fall back to random.

Trained weight analysis reads the exact saved tensor schema with NumPy memory
maps; it does not construct models unless a config override needs validation or
a `.pt` reference needs conversion. The separate `load_model` function in each
folder's `zoo.py` supports random construction, saved member `.npy` reconstruction,
and trusted zoo `trunk.pt` restoration. See each folder's README.

## Outputs and requirements

Every entry point writes CSV, JSON, and `figure.svg`. Output metadata explicitly
identifies random versus pretrained inputs. Tables include tensors, layers, and
the whole saved model; plots show transformer layers (zero-based). Composition
plots the mean residual/noise ratio across selected mixtures; individual results
remain in the tables. Existing files in the selected output directory are replaced.

NumPy is sufficient for saved `.npy` analysis. Random model initialization,
checkpoint reconstruction, and `.pt` loading additionally require PyTorch and
`transformers>=4.57,<5`. Use versions matching pretraining for reproducibility.
No downloads, training, or extra plotting dependencies are needed. Follow the
repository's policy of running PyTorch model operations on cluster compute jobs.

Each folder has NumPy-only numerical tests runnable as `python test_*.py` with
the actual test filename (no pytest dependency). These do not construct models.

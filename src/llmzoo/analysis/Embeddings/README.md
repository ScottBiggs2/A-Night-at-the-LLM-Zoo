# Data-space and weight-space embeddings

Two entry points, with one local helper. No imports from the training repository
or other analysis folders. These scripts do not train models.

- `data_embeddings.py`: load a given causal LM with `AutoModelForCausalLM.from_pretrained`,
  sample documents for each domain/run, pool hidden states and save reusable
  features. Plot documents, run centroids or domain centroids with PCA/t-SNE.
- `weight_embeddings.py`: load actual zoo checkpoint vectors, select the full
  saved model or specified transformer layers/tensors, then compute PCA/t-SNE.
  **Each point is one checkpoint**, not a layer.

## Pair domain/seed runs with their corresponding checkpoints

Use the same zoo path, domain selection and member IDs for both scripts. For zoo
anchor member i, the base data-selection seed is `10000 + i`, and its weights are
read from `w_i.npy`. The domain and run index come from `zoo_meta.json`. No weights
are averaged across seeds before the checkpoint projection.

For example, compare all four Web runs and all four Code runs:

```bash
ZOO=/path/to/zoo/gpt2_zoo_mini
ENCODER=/path/to/huggingface-format-causal-lm

python analysis/Embeddings/data_embeddings.py \
  --model-path "$ENCODER" \
  --zoo-path "$ZOO" --domains web code --members 0 1 2 3 4 5 6 7 \
  --samples-per-dataset 100 --point-level run \
  --hidden-layer -1 --pooling mean \
  --method both --scope both --output-dir results/data_web_code

python analysis/Embeddings/weight_embeddings.py \
  --checkpoint-path "$ZOO" --domains web code --members 0 1 2 3 4 5 6 7 \
  --method both --scope both --output-dir results/weights_web_code
```

The data plot has eight points, each averaging 100 document embeddings. The
weight plot has eight points, one for the **corresponding** pretrained checkpoint.
`member_idx`, `domain`, `run_id` and `sampling_seed` are saved in both label files.
Match rows by `member_idx`, not an assumed row order. For Web only, select
`--domains web --members 0 1 2 3` in both commands.

All documents use **one fixed pretrained encoder**, including across sampling
seeds. This keeps representation coordinates comparable. Using a different
encoder checkpoint for each run would conflate dataset changes with encoder
changes and is not what the data script does.

These are fresh analysis samples using the corresponding **base** seed. They do
not replay the exact documents, packed tokens, training steps or resumed stream
segments that originally trained each checkpoint. Runs can share documents.

Mixtures use the same pairing. Select identical mixture member IDs in both commands:

```bash
python analysis/Embeddings/data_embeddings.py \
  --model-path "$ENCODER" --zoo-path "$ZOO" \
  --include-mixtures --members 20 21 22 23 \
  --samples-per-dataset 200 --point-level run \
  --method both --output-dir results/data_mixtures

python analysis/Embeddings/weight_embeddings.py \
  --checkpoint-path "$ZOO" --include-mixtures --members 20 21 22 23 \
  --method both --output-dir results/weights_mixtures
```

For each mixture member, 200 documents **in total** are drawn using that member's
saved five-domain document probabilities and base sampling seed. The data centroid
is paired with that exact trained mixture checkpoint. Individual document labels
record both the mixture ID (`domain`) and the originating dataset (`source_domain`).
`encoding.json` records requested probabilities, actual sampled document counts,
and token counts after truncation. Small samples need not contain every domain.
The same member-selection function is used by both scripts.

You can include anchors and mixtures in the same comparison by listing both kinds
of member IDs with `--include-mixtures`. The current zoo has only one run per
singleton mixture, so it cannot support a within-mixture seed comparison; the
four-run within-domain comparison is available for anchors.

## Data-space options and files

`--model-path` accepts a Hugging Face model ID or a local `save_pretrained`
directory. A raw zoo `.npy` or `trunk.pt` is **not** a Hugging Face model directory.
The model must expose the usual `base_model` hidden-state interface. The pretrained
backbone is used to avoid allocating vocabulary logits. No custom remote code is
enabled. `--tokenizer-path` optionally supplies a separate tokenizer;
`--revision` pins the model/tokenizer revision.

`--datasets-config` accepts a JSON object mapping domain names to dataset sources.
The default `datasets.json` copies the five dataset definitions in the zoo code,
including French under `multilingual`. Dataset availability was not checked.
Sources can be Hugging Face datasets, local files, or saved datasets:

```json
{
  "web": {
    "path": "HuggingFaceFW/fineweb-edu",
    "name": "sample-10BT",
    "split": "train",
    "text_column": "text"
  },
  "my_json": {"file": "data/documents.jsonl", "text_column": "text"},
  "my_text": {"file": "data/documents.txt", "text_column": "text"},
  "my_saved_dataset": {"disk_path": "data/saved_dataset", "split": "train", "text_column": "text"}
}
```

Local file paths are relative to the config file. Supported file formats are
JSON/JSONL, CSV, text and Parquet. Each line of a plain text file is one document.
Use zoo domain names for automatic matching; custom dataset names have no
checkpoint correspondence unless the zoo metadata uses those names too.

- `--samples-per-dataset 100`: **100 documents per domain/mixture per run**.
- Without `--zoo-path`, `--sampling-seeds 0 1 2 3` creates independent sampling
  runs and leaves `member_idx` null. These are not automatically paired to models.
  Mixture sampling requires `--zoo-path --include-mixtures` to obtain the actual plan.
- `--hidden-layer -1`: final hidden-state output. Hugging Face index 0 is the
  embedding output, index 1 is after block 0, etc. This differs from the weight
  script's zero-based transformer block indices.
- `--pooling mean`: attention-mask-weighted mean over non-padding tokens;
  `last` uses the last non-padding token. Tokenizer-added special tokens are included.
- `--max-length 512`: truncate each document to this token length.
- `--point-level document|run|domain`: points represent individual documents,
  a centroid per run/checkpoint, or a centroid per domain.
- `--split-mode train|holdout|all`: default excludes the zoo's SHA1-based 1/64
  document holdout. Selection uses a seeded shuffle buffer, not uniform sampling
  from the entire streaming corpus. It deduplicates documents within a run and
  fails if the requested count cannot be found within `--max-documents` scanned rows.
- `--device auto` selects CUDA when available, otherwise CPU. `--dtype` defaults
  to float32; float16 and bfloat16 can be selected for compatible hardware.
- `--encode-only` saves features and sample labels without computing projections.
- `--save-text` includes sampled text in sample labels; default stores hashes only.

Data files:

- `features.npy`: float32 matrix, one document per row.
- `sample_labels.json`: document rows, hashes, token counts and domain/run/model IDs.
- `encoding.json`: encoder, sampling and pooling configuration.
- `projection_features.npy`: document features or aggregated run/domain centroids
  actually used for the projection.
- `point_selection.json`: aggregation level and correspondence information.

Replot saved document features without loading a model or dataset:

```bash
python analysis/Embeddings/data_embeddings.py \
  --features-path results/data_web_code/features.npy \
  --point-level document --method pca --output-dir results/data_documents
```

The default labels for `--features-path` are the adjacent `sample_labels.json`.
Pass `--labels-path` for another compatible label file.

## Weight-space options and files

Input is one zoo architecture directory containing layout-version-2 metadata and
flat float32 `w_i.npy` files. The loader respects saved extra-parameter and block
schemas. Tied parameters are represented once. All checkpoints in a comparison
must share the architecture, parameter layout, trunk and training configuration.

Full saved model is the default. For selected layers:

```bash
python analysis/Embeddings/weight_embeddings.py \
  --checkpoint-path "$ZOO" --layers 0 3 7 --per-layer \
  --method both --output-dir results/weights_selected_layers
```

The main result concatenates those layers' coordinates for each checkpoint.
`--per-layer` also produces separate projections in `layer_0`, `layer_3`, etc.
`--include-extra` adds saved embeddings/final normalization to a selected-layer
analysis; full-model analysis already includes every saved tensor. Optional
`--tensor-pattern 'transformer.h.*.attn.*'` filters full parameter names.

All single-domain anchors are selected by default. `--include-mixtures` allows
singleton mixtures, each labeled by its mixture ID; these have no four-run
within-mixture baseline. Missing checkpoints and unmatched selections are errors.
`weights.json` records selected members, tensor names and parameter counts.

Weights are centered coordinate-wise and processed in chunks. Only an N-by-N
Gram matrix is retained, not an N-by-full-model dense matrix. Subtracting a common
trunk gives the same centered PCA and pairwise distances. The script does not
download, initialize or substitute weights for a missing checkpoint.

## Common projection outputs and interpretation

Both scripts write `labels.json`, PCA scores/eigenvalues, centered Gram matrices,
2-D coordinates as NPY/CSV, and PNG plots. The PCA eigenvalues are squared singular
values of the centered feature matrix (divide by N-1 for covariance eigenvalues).

- `--scope both` creates cross-domain and within-domain views. Within-domain
  panels reuse the **same globally fitted coordinates**, with shared axis limits.
- `--method pca|tsne|both` chooses plots. PCA scores are always saved.
- t-SNE uses up to 50 PCA dimensions by default; `--tsne-pca-dim 0` keeps all
  numerically resolved directions. Retained variance and actual perplexity are
  recorded in `projection.json`. Perplexity is reduced for small ensembles.
- No coordinate standardization or row normalization is applied. Zero-variation
  inputs get zero PCA coordinates; t-SNE is skipped. Rank-one PCA is padded with
  a zero second coordinate.

`domains_geometry.npz` and `domain_runs_geometry.npz` contain labeled centroid
distance matrices and centered cosine similarities **before 2-D projection**.
Undefined cosines for zero-norm centroids are NaN. These matrices let you compare
whether corresponding domain/run relationships are similar in data and weight
space. Compare matching labels and normalize distances when comparing different
units; sample counts and pooling must be kept consistent.

Separate data-space and weight-space projections do not share an axis system:
their apparent plot orientations, rotations and signs cannot be compared directly.
t-SNE is a neighborhood visualization, not evidence of global angles or distances.
Likewise, separately fitted layer plots have independent axes. Use the labeled
original-space distance/cosine matrices for geometric comparison.

The Gram approach takes O(N²) memory; use run centroids or fewer documents for
large samples. A model plot can be small even when checkpoints themselves are huge.

Dependencies: NumPy, scikit-learn >=1.2, Matplotlib; data encoding additionally
requires PyTorch, transformers >=4.57,<5, and datasets >=3.6,<4. No scripts were
executed or models/datasets downloaded during implementation.

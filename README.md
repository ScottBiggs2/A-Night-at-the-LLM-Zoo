# llmzoo — a data-mixture zoo of GPT-2s, and a conditional flow over their weights

Code for *A Night at the LLM Zoo: Data Conditioned Flow Matching for Generating
Pretrained LLM Weights* (anonymous submission).

The pipeline has three parts:

1. **Zoo.** Train a population of GPT-2 models at three scales (Mini 51M, Small 124M,
   Medium 355M), with N = 100 members per scale. All members share a pretraining
   *trunk*. They then *branch* for the last β of the step budget, each on its own
   data mixture π ∈ Δ⁴ over five domains (web, code, math, books, French).
2. **Population statistics.** Analyse each zoo in weight space and function space:
   spectrum, cosine geometry, ANOVA/η², leave-one-mixture-out regression, loss
   barriers and prediction disagreement.
3. **Generation.** Fit a dual (Gram-matrix) PCA over the complete flattened weight
   vectors of each zoo. Train a π-conditioned rectified flow over the PCA codes and
   decode samples back into working models. Score them with held-out-mixture
   perplexity, CRPS, density, and downstream benchmarks, against retrieval, soup,
   trunk and PCA-round-trip baselines.

> **The zoo checkpoints are not included in this repository** (about 100 × 1.4 GB at
> Medium). Every stage from §2 onward needs a trained zoo, and training one needs a
> GPU cluster (see the cost table). The paper's tables and figures can still be
> re-rendered on a laptop from the sealed summaries in `reports/data/` (§8).

---

## 1. Layout

```
src/llmzoo/     library: artifacts/ (sealed bundles), data/ (mixtures, ensemble),
                eval/ (perplexity, domains, benchmarks), gen/ (flow, density),
                models/ (zoo archs), pca/ (streaming Gram PCA), analysis/ (population stats)
scripts/        command-line entry points:  python scripts/<name>.py --help
slurm/          Slurm job scripts (one GPU per job; arrays for members and flow arms)
reports/        table/figure builders and the sealed JSON they read (reports/data/)
tests/          plain scripts, not pytest:  python tests/<name>.py
```

`import llmzoo` pulls in neither numpy nor torch. The report builders and
`tests/test_report.py` use only the standard library (plus matplotlib for figures).

## 2. Installation

Python ≥ 3.11. Install torch **first**, from the index that matches your CUDA. torch is
deliberately not a declared dependency, so `pip install -e .` cannot replace a CUDA
build with a CPU wheel.

```bash
python -m venv .venv && source .venv/bin/activate      # or conda
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1
pip install -e ".[plots]"
```

`slurm/setup_env.sh` does the same inside a conda environment and runs a quick import check.
Versions used for the paper: torch 2.9.1 (cu128), transformers 4.57, datasets 3.6.

**Data.** Five public, ungated Hugging Face datasets are streamed. They are never
downloaded in full. The ids are in `src/llmzoo/data/mixtures.py`:

| domain | dataset |
|---|---|
| web | `HuggingFaceFW/fineweb-edu` (`sample-10BT`) |
| code | `codeparrot/codeparrot-clean` |
| math | `open-web-math/open-web-math` |
| books | `manu/project_gutenberg` |
| multilingual | `HuggingFaceFW/fineweb-2` (`fra_Latn`) |

A 100-wide job array opens many streams at once from one IP. Set `HF_TOKEN` to avoid
anonymous rate limits; the open calls retry with backoff. W&B logging is optional: set
`WANDB_API_KEY` and `WANDB_ENTITY`, or leave them unset and runs log offline.

## 3. Cluster configuration

Every job sources `slurm/env.sh`, which reads a site file you create once:

```bash
cp slurm/site.example.sh slurm/site.sh     # gitignored; edit it
export SBATCH_ACCOUNT=<your-account>       # Slurm reads this natively
mkdir -p logs                              # job logs go to ./logs/
```

Job headers name two generic partitions, `gpu` and `cpu`. Map them to your site's
names by passing `sbatch -p <name>`, or edit the `#SBATCH --partition` lines once.
`launch_beta_arm.sh` takes `PARTITION=` and `EVAL_PARTITION=` instead. Don't export
`SBATCH_PARTITION` globally: it would also redirect the CPU jobs.

`site.sh` sets `ARTIFACT_DIR`, the persistent storage for zoos, PCA runs and flows
(hundreds of GB at Medium). It also sets `SCRATCH_DIR` (HF cache, W&B, checkpoints)
and how to activate the Python environment. **Submit every job from the repository
root.** Jobs use `$SLURM_SUBMIT_DIR` as the code directory.

Requests assume one GPU per job. The paper used a single B200 (Mini/Small/Medium
fit in memory). Flow training runs on CPU in under two minutes per arm.

## 4. Zoo training

```bash
# 0. free checks, seconds on a CPU node: every dataset id resolves, and the
#    five-way interleave works (singleton mixtures open five streams)
python scripts/train_zoo.py --verify_domains
python scripts/train_zoo.py --verify_mixture

# 1. inspect the plan: 20 anchors (4 replicates at each simplex vertex)
#    + 80 Dirichlet singletons = 100 members, 85 distinct mixtures
python scripts/train_zoo.py --mode plan --arch gpt2_zoo_mini --n_members 100 --beta 0.15

# 2. trunk -> 100-branch array -> {domain-separation gate, k=99 PCA spectrum}
bash slurm/launch_beta_arm.sh 0.15 100 gpt2_zoo_mini
TRUNK_TIME=06:00:00 BRANCH_TIME=01:30:00 bash slurm/launch_beta_arm.sh 0.15 100 gpt2_zoo_small
CKPT_EVERY=1000 TRUNK_TIME=24:00:00 BRANCH_TIME=05:00:00 SPEC_TIME=06:00:00 \
    bash slurm/launch_beta_arm.sh 0.15 100 gpt2_zoo_medium
```

The launcher derives `ZOO_ROOT=$ARTIFACT_DIR/zoo_b015_<scale>_n100` and the PCA run
name `zoo_b015_<scale>_k99`, and chains the jobs with `afterok`. A member whose
`w_<i>.npy` exists exits immediately, so a partially failed array can be resubmitted
unchanged. `CKPT_EVERY` enables mid-span checkpointing, which Medium's ~19 h trunk
needs. Checkpoints go to `$SCRATCH_DIR` and are deleted on completion.

**The gate.** `scripts/eval_domains.py` (run by the launcher) exits 1 if mixture
identity is not measurable in the trained models. If it fails, the answer is a larger
β, not a downstream fix. Its output is `$ZOO_ROOT/<arch>/domain_separation.json`.

Settings used in the paper (identical at all three scales): β = 0.15, token budget
20·D, AdamW (0.9, 0.95), wd 0.1, peak LR 6e-4 with 1% warmup and cosine decay to 10%,
batch 16 × grad-accum 8 × 1024 tokens, bf16, trunk seed 1234, branch seed 10000 + i.
β = 0.30 exists only as the Mini calibration point.

| scale | params | steps (trunk / branch) | trunk | per branch | ≈ total GPU-h |
|---|---|---|---|---|---|
| Mini | 51M | 6,676 / 1,178 | ~1.2 h | ~15 min | ~25 |
| Small | 124M | 16,140 / 2,848 | ~3.8 h | ~45 min | ~80 |
| Medium | 355M | 46,021 / 8,121 | ~19 h | ~3.5 h | ~370 |

Times were measured on one B200. Throughput is partly data-bound (tokenisation runs
in-thread), so it does not scale linearly with parameter count.

## 5. Population statistics (paper §3, Appendix "Population")

```bash
# weight space: spectrum, geometry, ANOVA, containment, composition, embeddings,
# leave-one-mixture-out regression (cpu partition; ~4 min / 30 min / 85 min)
ZOO=zoo_b015_mini_n100/gpt2_zoo_mini     TAG=mini_b015   sbatch slurm/population.sbatch
ZOO=zoo_b015_small_n100/gpt2_zoo_small   TAG=small_b015  sbatch slurm/population.sbatch
ZOO=zoo_b015_medium_n100/gpt2_zoo_medium TAG=medium_b015 \
    sbatch --mem=96G --time=04:00:00 slurm/population.sbatch

# block-only vs whole-stack spectrum (no GPU, no torch)
python scripts/diag_block_spectrum.py --zoo_dir $ARTIFACT_DIR/zoo_b015_mini_n100/gpt2_zoo_mini \
    --json_out $ARTIFACT_DIR/ladder/blockspec_mini_b015.json

# function space: prediction disagreement + pairwise interpolation barriers (1 GPU)
ZOO=zoo_b015_mini_n100/gpt2_zoo_mini ARCH=gpt2_zoo_mini TAG=mini_b015 \
    sbatch slurm/population_fn.sbatch
```

Outputs go to `$ARTIFACT_DIR/population/<tag>/`. `reports/build_population_figures.py`
collects them into `reports/population_payload.json`.

The leave-one-mixture-out regression needs only the sealed 100 × 100 displacement
Gram, so it also runs locally from the shipped copy:

```bash
python scripts/diag_pi_regression.py --help   # reads reports/data/population/<tag>/{displacement_gram.npy,zoo_meta.json}
```

## 6. Conditional generation (paper §4)

All generative results use **k = 50** codes, obtained by truncating the k = 99 PCA fit
that the launcher already produced. All population spectra use k = 99.

**6.1 Flows.** `slurm/flow_arm.sbatch` encodes the paper's operating point: width
(64, 128, 64), π-conditioning, 20,000 epochs always run in full, 15% of rows held out
for checkpoint selection, lr 3e-4, batch 64, wd 1e-4, conditioning dropout 0.15,
50 Euler steps. The script defaults in `scripts/train_flow.py` are **not** the paper's
settings, so use the sbatch or copy its flags. Each arm takes minutes on CPU.

```bash
RUN=zoo_b015_mini_k99   # likewise zoo_b015_{small,medium}_k99

# whole-zoo flow (tables: density, CRPS, dose-response, continuity). At Medium the
# paper uses the e20k retrain: MANIFEST=reports/data/manifests/e20k_b015.tsv --array=0
RUN=$RUN K=50 sbatch slurm/flow_arm.sbatch
# unconditional null with the same architecture
RUN=$RUN K=50 COND_MODE=family MANIFEST=reports/data/manifests/null_b015.tsv \
    sbatch --array=1 slurm/flow_arm.sbatch
# leave-one-mixture-out: 85 flows, each blinded to one mixture
RUN=$RUN K=50 MANIFEST=reports/data/manifests/lomo_def_b015.tsv sbatch --array=0-84 slurm/flow_arm.sbatch
# exclusion-radius arms behind the MCD table (r = 0.4 at four targets; Mini, Medium)
RUN=$RUN K=50 MANIFEST=reports/data/manifests/radius_def_b015.tsv \
    sbatch --array=4,9,14,19 slurm/flow_arm.sbatch
# vertices-only flow: trained on the 20 anchors, queried in the interior
RUN=$RUN K=50 MANIFEST=reports/data/manifests/anchoronly_b015.tsv sbatch --array=0 slurm/flow_arm.sbatch
```

Manifests are generated by `scripts/make_flow_manifest.py --design {lomo,radius,sweep}`
from a zoo's `zoo_meta.json`. The committed TSVs are the ones used in the paper.
`sweep*_b015.tsv`, `select*_b015.tsv` and `report_flow_arms.py`/`select_config.py`
reproduce the pre-registered hyperparameter selection (Appendix "Tuning").

**6.2 Functional evaluation.** Decode samples into models and score held-out-domain
perplexity at the target mixture:

```bash
# each LOMO flow at the mixture it was blinded to (32 samples/arm)
RUN=$RUN K=50 ZOO=zoo_b015_mini_n100 ARCH=gpt2_zoo_mini \
  GLOB='flow_k50_codes_pi_def_lomo_*' OUT=gen/lomo_mini sbatch --array=0-84%32 slurm/eval_arm.sbatch
python scripts/merge_gen.py --label lomo --out $ARTIFACT_DIR/gen/merged_lomo_mini.json \
    $ARTIFACT_DIR/gen/lomo_mini/*.json

# baselines, all 85 mixtures per job
for S in soup pca_only; do
  SOURCE=$S RUN=$RUN K=50 ZOO=zoo_b015_mini_n100 ARCH=gpt2_zoo_mini COHORT=48 \
    OUT=gen/${S}_mini.json sbatch slurm/eval_baseline.sbatch
done
SOURCE=trunk TRUNK_FLAT=$ARTIFACT_DIR/population/mini_b015/trunk_flat.npy \
  RUN=$RUN K=50 ZOO=zoo_b015_mini_n100 ARCH=gpt2_zoo_mini OUT=gen/trunk_mini.json \
  sbatch slurm/eval_baseline.sbatch
# a many-model arm (the vertices-only flow, 85 × 16 samples) should be sharded
SOURCE=flow FLOW_DIR=flow_k50_codes_pi_anchoronly N_SAMPLES=16 COHORT=48 SHARDS=9 \
  RUN=$RUN K=50 ZOO=zoo_b015_mini_n100 ARCH=gpt2_zoo_mini OUT=gen/anchoronly_mini/a.json \
  sbatch --array=0-8 slurm/eval_baseline.sbatch

# the ladder: flow vs retrieval (computed here from the zoo itself) vs baselines
A=$ARTIFACT_DIR
python scripts/report_functional.py --arch gpt2_zoo_mini \
    --zoo_dir $A/zoo_b015_mini_n100/gpt2_zoo_mini \
    --gen lomo=$A/gen/merged_lomo_mini.json \
    --gen soup=$A/gen/soup_mini.json --gen trunk=$A/gen/trunk_mini.json \
    --gen pca_only=$A/gen/pca_only_mini.json \
    --json_out $A/gen/fn_phase3c_mini.json
```

The dose-response and continuity inputs come from a dense π-path through the
whole-zoo flow (41 points, barycentre → multilingual):

```bash
python scripts/eval_generated.py --artifact_dir $A --run_name $RUN --k 50 \
    --zoo_dir $A/zoo_b015_mini_n100/gpt2_zoo_mini --source flow --flow_dir flow_k50_codes_pi \
    --cohort 32 --no_wandb --targets none --path 0.2 0.2 0.2 0.2 0.2  0 0 0 0 1 \
    --n_path 41 --n_samples 2 --out $A/gen/path_mult_k50.json
```

The retrieval baseline returns the nearest *training* mixture's member, blinded by the
same exclusion ball as the flow. It is the control that separates generalisation from
memorisation.

**6.3 Density, CRPS, MCD.** For the whole-zoo flow against its unconditional null
(pass `--flow_dir flow_k50_codes_pi_def_rad_dirichlet_<idx>_r0p4 --out mcd_<scale>_<idx>_r0p4.json`
for the MCD table; the flow's own sealed exclusion ball defines the held-out set):

```bash
python scripts/report_density.py --artifact_dir $A --run_name $RUN --k 50 \
    --flow_dir flow_k50_codes_pi --null_flow_dir flow_k50_codes_null_ss1 \
    --zoo_dir $A/zoo_b015_mini_n100/gpt2_zoo_mini --out $A/density_mini_k50.json
```

**6.4 Downstream benchmarks.** Pass `BENCH=<name>` (and `BENCH_NQ=200`) to
`eval_arm.sbatch` / `eval_baseline.sbatch`, then aggregate with
`python scripts/report_bench.py reports/data/bench/*.json`. `bench10_b015.tsv` is the
10-cell benchmark design.

## 7. Diagnostics worth running on any new zoo

- **`code_rms_ratio` / `code_rms_ratio_at_fixed_pi`** (sealed in every `flow_meta.json`).
  A flow that collapses toward the ensemble mean scores ΔPPL ≈ 0 while generating
  nothing, because the zoo is mean-centred. Check sample dispersion before reading any
  generative ΔPPL. `pi_code_distance_corr` (≈ +0.65 here; ≈ 0 means the π↔code rows are
  mis-joined) is a one-epoch pairing check.
- **`--source pca_only`**: the k = 50 truncation floor. Subtract it before attributing
  error to the generator.
- Fold by **mixture**, never by member: the four anchor replicates share a π exactly.

## 8. Re-rendering the paper's tables and figures (laptop, no GPU)

```bash
bash reports/rebuild_tables.sh      # -> outputs/tables/*.tex
bash reports/rebuild_figures.sh     # -> outputs/figs/*.{pdf,png}   (needs matplotlib)
```

Both read only the sealed summaries in `reports/data/`:

| input | produced by | used for |
|---|---|---|
| `fn_lomo_<scale>.json` | `report_functional.py` over the LOMO arms | ladder, strata, CRPS, generalisation/ladder figures |
| `fn_phase3c_<scale>.json` | `report_functional.py` with baselines | baselines table |
| `density_<scale>_k50.json`, `mcd_*_r0p4.json` | `report_density.py` | density, MCD tables; dose-response, continuity figures |
| `spectrum_<scale>_k99.json` | `train_stack.py` (launcher) | scaling table |
| `flowmeta_<scale>.json` | `train_flow.py` (`flow_meta.json`) | scaling table |
| `path_mult_k50.json` | `eval_generated.py --path` (§6.2) | dose-response |
| `arms/mini_sweep.json` | `report_flow_arms.py` | tuning figure |
| `bench/bench_<scale>.json` | `report_bench.py` | benchmark tables |
| `population_payload.json`, `population/*` | `build_population_figures.py` | zoo-design and barrier figures |

## 9. Tests

```bash
python tests/test_report.py                        # stdlib only; runs anywhere
TEST=tests/test_zoo.py sbatch slurm/tests.sbatch   # the rest need torch; any GPU or CPU node
```

`test_zoo`, `test_flow`, `test_bundle`, `test_ensemble`, `test_density`,
`test_functional` and `test_conditioning` use tiny models and finish in seconds to
minutes. The `src/llmzoo/analysis/*/test_*.py` scripts audit the population-statistics
code against synthetic data.

## 10. Appendix: the manufactured noise ensemble

Before the zoo existed, the pipeline was validated on a *manufactured* ensemble made
of Gaussian perturbations of one pretrained checkpoint (gpt2-medium, SmolLM2-360M,
Pythia-410M). The generative models condition on the model family. The result is a
pre-registered null: the leading eigenvalue sits at the median (`ev0/median ≈ 1.00`),
so there is no structure to learn. With enough capacity × training time, the flow
memorises and collapses the marginal (training loss improves while sample RMS falls
from 1.04 to 0.16). `diag_flow_capacity.py` reproduces that table.

```bash
sbatch slurm/calibrate_noise.sbatch   # per-arch perturbation scale (3e-3 is calibrated)
sbatch slurm/stack_smoke.sbatch       # ~1 min end-to-end smoke test, tiny models
sbatch slurm/stack_run.sbatch         # PCA over the three noise ensembles
sbatch slurm/flow_run.sbatch          # codes-space flows + evaluation + report
```

## License

Released for review. A license will be added with the de-anonymised version.

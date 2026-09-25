# Subspace containment: does the mixture span contain the expert span?

Standalone, **NumPy only**. No torch, no transformers, no repository imports, no
network, no downloads. One streaming pass over the saved `w_<i>.npy` vectors.

```bash
python analysis/subspace_containment/containment.py \
  --checkpoint-path /path/to/zoo/gpt2_zoo_mini \
  --output-dir results/containment
```

The checkpoint path is the **architecture directory** holding `zoo_meta.json` and
`w_<index>.npy`. Needs a complete balanced anchor set and at least `D` singleton
mixtures, so N >= 25 for the five-domain zoo; smaller zoos are refused with the
reason.

## The two subspaces

```
A = span{ abar_d - abar }    the D centred anchor domain-mean effects   dim <= D-1
M = span{ y_j  - ybar  }     the centred singleton-mixture displacements
```

`mixture_composition/composition.py` already answers one direction — whether a
mixture lies **inside** A, by regressing it on the anchor means. Nothing answered
the other, because every Gram in this analysis tree is built from anchors only
(`anchor_groups` skips `kind != "anchor"`), so no mixture-side subspace was ever
formed. This script forms both and reports the geometry between them.

## Why the member Gram is sufficient

Principal angles between two spans depend only on `A'A`, `A'B`, `B'B`. Both spans
here are linear combinations of the **same members**, so all three are blocks of
the N-by-N member Gram:

```
G_A = L_A G L_A'      G_AM = L_A G L_M'      G_M = L_M G L_M'
```

Every row of `L_A`, `L_M` and `L_W` sums to zero, so the Gram may be taken of
displacements from any fixed reference — the reference cancels exactly. That is
the conditioning argument `diag_block_spectrum.py` and `diag_zoo_geometry.py`
make: members share a trunk, so a raw Gram would recover a small difference from
large numbers.

## The noise floor is the whole difficulty

Containment **cannot** read 1.0 even when it holds exactly, because `abar_d`
carries sampling noise `Sigma_w/S` and `y_j` carries `Sigma_w`, and neither lies
in the other's span. On a synthetic zoo where mixtures lie in the anchor span *by
construction*, the raw principal cosines top out at 0.988–0.995.

This is the same trap that pins `domain_geometry`'s `effective_rank` at `D-1`
regardless of signal, and the same species as reading a rank-bound tail ratio as a
bulk statistic. So the diagonal blocks
are debiased analytically before any angle is taken. With `t_w = tr(Sigma_w)`
estimated from the within-anchor spread:

```
E[G_A]  = A'A + (I_D - J_D/D) * t_w / S
E[G_M]  = M'M + (I_n - J_n/n) * t_w
E[G_AM] = A'M          <- anchors and mixtures are independent models, so the
                          cross block needs NO correction. Only the diagonals do.
```

Raw and debiased statistics are both written. **Quote the debiased ones.**

## What the answer means

A class is a point on the 5-simplex, so if `pi -> w` were **linear** then A and M
are the same `<= 4`-dimensional space and containment is trivially true in both
directions. The question has content only if the map is nonlinear, in which case
the mixture set is curved, spans more than `dim(Delta^4) = 4`, and properly
contains A.

So the headline column is not really an angle. It is

```
debiased_mixture_dimension_above_noise   vs   simplex_dimension
```

and the boolean `nonlinear_beyond_simplex` derived from it. `diag_block_spectrum.py`
already measured effective rank 4.03 (embeddings) / 7.20 (blocks) on the full
100-member zoo, with 18% and 32% of variance beyond `c4` — this script is what
separates that excess into curvature versus within-anchor noise.

## Reading the energy fractions

`anchor_energy_in_mixture_span` is `||A P_M||_F^2 / ||A||_F^2`. A **random** rank-r
subspace of R^P captures `r/P` of any fixed vector's energy, so the fraction is
reported beside `random_subspace_null_anchor_in_mixture = r/P`. Read it against
that null, never against zero — this is the correction
`composition.py`'s `perpendicular_fraction` still needs.

## Outputs

`containment.csv` (scalar columns) and `containment.json` (adds the eigenvalue
and cosine lists), one row per region: `extra`, each `layer_i`, and
`whole_model`. Regions matter here: at Mini the embedding block is 51% of `D` but
carries 87.5% of the centred variance, so the whole-model row is largely the
embeddings' row.

## Chunking, and what it does and does not touch

`--chunk-size` is an I/O buffer only. A Gram is a **sum over coordinates**, so
accumulating it chunk by chunk partitions that sum exactly, and chunks never
cross a tensor boundary. The subspace is computed from the **finished** N-by-N
Gram — the eigendecomposition and the principal angles never see a chunk.
Verified through the full pipeline at chunk sizes 1, 2, 7, 13, 97, 199 and
whole-tensor: worst relative difference 3e-13, with `dim(A)`, `dim(M)` and
`mixture_dimension_above_noise` bit-identical.

What *would* break under chunking is centring across **coordinates**. Every
centring here is across samples or replicates, where coordinate `p`'s mean
depends only on column `p`. Do not change that.

Note `block_size` (from `zoo_meta.json`: the parameter count of one transformer
layer, and therefore which coordinates a region owns) is unrelated to
`--chunk-size`. They never interact.

## Precision

Forming `G = X X'` squares the condition number. That is the price of never
materialising a P-dimensional basis, and it costs ~1e-4 relative on the principal
cosines against a QR reference — checked in `test_containment.py`, which is why
its tolerances there are 2e-3 and not machine precision. Fine for a statistic
quoted to three decimals; do not read the fourth.

## Assumptions

* A common within-member covariance, estimated from **anchors only**. The five
  corpora differ ~130x in document size, so this is the assumption most worth
  auditing; `anovas/` should report per-domain within-SS to check it.
* Independent replicates. Not verified here — `member_<i>.json` carries
  `throughput.devices` and `throughput.stream_restarts`, and both can break it.

## Tests

```bash
python analysis/subspace_containment/test_containment.py
```

NumPy only. Covers the saved layout, coefficient rows summing to zero, the
Gram-block route against explicit QR bases, chunk invariance, that debiasing
moves the estimate toward the truth, and the two verdicts: mixtures built linear
in `pi` are contained and stay within the simplex dimension, mixtures built with
curvature exceed it.

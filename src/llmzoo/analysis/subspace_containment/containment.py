#!/usr/bin/env python3
"""Does the mixture span CONTAIN the anchor (expert) span, or sit inside it?

Standalone: NumPy only. No torch, no transformers, no repository imports, no
network. One streaming pass over the saved w_<i>.npy vectors.

THE TWO SUBSPACES
-----------------
    A = span{ abar_d - abar }        the D centred anchor domain-mean effects
    M = span{ y_j - ybar }           the centred singleton-mixture displacements

`mixture_composition/composition.py` already answers one direction -- whether a
mixture lies INSIDE A, by regressing it on the anchor means. Nothing answered the
other direction, because every Gram in this analysis tree is built from anchors
only (`anchor_groups` skips `kind != "anchor"`), so no mixture-side subspace was
ever formed.

WHY THE MEMBER GRAM IS SUFFICIENT
---------------------------------
Principal angles between two spans depend only on A'A, A'B and B'B. Both spans
here are linear combinations of the SAME members, so all three are blocks of the
N-by-N member Gram:

    G_A  = L_A G L_A'      G_AM = L_A G L_M'      G_M = L_M G L_M'

Every row of L_A and L_M sums to zero, so the Gram may be taken of displacements
from any fixed reference -- the reference cancels exactly. That is the same
conditioning argument diag_block_spectrum.py and diag_zoo_geometry.py make: the
members share a trunk, so a raw Gram would recover a small difference from large
numbers.

THE NOISE FLOOR, WHICH IS THE WHOLE DIFFICULTY
----------------------------------------------
Containment CANNOT read 1.0 even when it holds exactly, because abar_d carries
sampling noise Sigma_w/S and y_j carries Sigma_w, and neither lies in the other's
span. Measured on a synthetic zoo where mixtures lie in the anchor span BY
CONSTRUCTION, the raw principal cosines topped out at 0.988-0.995, not 1.

This is the same trap that pins domain_geometry's `effective_rank` at D-1
regardless of signal, and the same species as reading a rank-bound tail ratio as a
bulk statistic. So the diagonal blocks
are debiased analytically before any angle is taken. With
t_w = tr(Sigma_w) estimated from the within-anchor spread,

    E[G_A]  = A'A + (I_D - J_D/D) * t_w / S
    E[G_M]  = M'M + (I_n - J_n/n) * t_w
    E[G_AM] = A'M                        <- anchors and mixtures are independent
                                            models, so the cross block needs NO
                                            correction. Only the diagonals do.

Raw and debiased statistics are both reported; quote the debiased ones.

WHAT THE ANSWER MEANS
---------------------
A class is a point on the 5-simplex, so if pi -> w were LINEAR then A and M are
the same <=4-dimensional space and containment is trivially true both ways. The
question has content only if the map is nonlinear, in which case the mixture set
is curved, spans more than dim(Delta^4) = 4, and properly contains A. So the
headline number is not really an angle -- it is
`mixture_dimension_above_noise`: how many debiased mixture directions stand above
the noise floor, against 4.

    python analysis/subspace_containment/containment.py \
        --checkpoint-path /path/to/zoo/gpt2_zoo_mini \
        --output-dir results/containment
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


# ---------------------------------------------------------------------------
# Saved layout
# ---------------------------------------------------------------------------

def tensor_layout(meta):
    """Yield (name, region, start, stop) over the saved [extra | blocks] vector."""
    if meta["layout_version"] != 2:
        raise ValueError("Expected zoo layout_version=2")
    offset, seen = 0, set()
    groups = [("", "extra", meta["extra_schema"], meta["extra_size"])] + [
        (f"transformer.h.{i}.", f"layer_{i}", meta["schema"], meta["block_size"])
        for i in range(meta["n_layers"])]
    for prefix, region, schema, expected in groups:
        begin = offset
        for name, shape in schema:
            name = prefix + name
            if name in seen or not shape or any(int(n) < 1 for n in shape):
                raise ValueError(f"Invalid or duplicate tensor: {name}")
            seen.add(name)
            size = math.prod(shape)
            yield name, region, offset, offset + size
            offset += size
        if offset - begin != expected:
            raise ValueError(f"Schema size disagrees with the saved size for {region}")
    if offset != meta["n_params"]:
        raise ValueError("Schema does not cover n_params")


def open_weights(path, meta):
    w = np.load(path, mmap_mode="r", allow_pickle=False)
    if w.shape != (meta["n_params"],) or w.dtype != np.float32:
        raise ValueError(f"{path}: expected a float32 vector of length {meta['n_params']}")
    return w


# ---------------------------------------------------------------------------
# Member selection
# ---------------------------------------------------------------------------

def member_tables(meta):
    """(order, anchor_groups, mixture_positions, pi) with anchors first.

    `order` is the member index of each Gram row. Anchors must be a COMPLETE
    balanced set of one-hot domains -- they are the only replicates in the zoo
    and therefore the only estimate of the noise floor everything else divides by.
    """
    domains = meta["domains"]
    if len(domains) < 2 or len(set(domains)) != len(domains):
        raise ValueError("Need at least two distinct domains")
    groups, singles, seen = [[] for _ in domains], [], set()
    for member in meta["members"]:
        idx = int(member["idx"])
        if idx in seen:
            raise ValueError(f"Duplicate member index {idx}")
        seen.add(idx)
        pi = np.asarray(member["pi"], dtype=np.float64)
        if pi.shape != (len(domains),) or not np.isfinite(pi).all() or (pi < 0).any() \
                or not np.isclose(pi.sum(), 1.0, atol=1e-8, rtol=0):
            raise ValueError(f"Member {idx} has an invalid mixture")
        if member["kind"] == "anchor":
            d = int(pi.argmax())
            if not np.array_equal(pi, np.eye(len(domains))[d]):
                raise ValueError(f"Anchor {idx} is not one-hot")
            groups[d].append((idx, pi))
        elif member["kind"] == "singleton":
            singles.append((idx, pi))
        else:
            raise ValueError(f"Member {idx} has unknown kind {member['kind']!r}")
    sizes = {len(g) for g in groups}
    if len(sizes) != 1 or min(sizes) < 2:
        raise ValueError(
            f"Need an equal number of anchors per domain, at least two "
            f"(have {[len(g) for g in groups]}). A zoo with N<{4*len(domains)} "
            f"has no complete anchor set and no within-domain noise floor.")
    if sizes != {meta["n_anchor_branches"]}:
        raise ValueError("Anchor count disagrees with n_anchor_branches")
    if len(singles) < len(domains):
        raise ValueError(
            f"Only {len(singles)} singleton mixtures. The mixture span has at most "
            f"{max(len(singles)-1, 0)} dimensions, below dim(simplex)="
            f"{len(domains)-1}, so containment of the anchor span is impossible by "
            f"arithmetic. Use a zoo with at least {len(domains)} singletons.")
    for g in groups:
        g.sort()
    singles.sort()
    order = [idx for g in groups for idx, _ in g] + [idx for idx, _ in singles]
    n_anchor = sum(len(g) for g in groups)
    mixtures = list(range(n_anchor, n_anchor + len(singles)))
    pi = np.array([p for g in groups for _, p in g] + [p for _, p in singles])
    return order, [len(g) for g in groups], mixtures, pi


def coefficient_matrices(n, group_sizes, mixtures):
    """L_A, L_M and L_W as coefficient matrices over Gram rows.

    Every row sums to zero, which is what makes the Gram reference cancel.
      L_A : D centred domain-mean effects          -> A = L_A X
      L_M : centred mixture displacements          -> M = L_M X
      L_W : within-anchor residuals                -> tr(Sigma_w) estimate
    """
    d, s = len(group_sizes), group_sizes[0]
    la = np.zeros((d, n))
    at = 0
    for i, size in enumerate(group_sizes):
        la[i, at:at + size] = 1.0 / size
        at += size
    la -= la.mean(axis=0)                                   # centre the effects
    lm = np.zeros((len(mixtures), n))
    lm[np.arange(len(mixtures)), mixtures] = 1.0
    lm -= lm.mean(axis=0)                                   # centre the mixtures
    lw, at, rows = [], 0, []
    for size in group_sizes:
        for j in range(size):
            row = np.zeros(n)
            row[at + j] = 1.0
            row[at:at + size] -= 1.0 / size
            rows.append(row)
        at += size
    lw = np.array(rows)
    return la, lm, lw, d, s


# ---------------------------------------------------------------------------
# Gram
# ---------------------------------------------------------------------------

def region_grams(zoo_dir, meta, order, chunk_size=65536):
    """Streaming N-by-N Gram of displacements, per region and whole-model.

    The reference is member `order[0]`; it cancels because every coefficient row
    used downstream sums to zero. Taking the Gram of displacements rather than of
    raw vectors avoids recovering a small difference from large numbers, since
    all members descend from one trunk.
    """
    arrays = [open_weights(Path(zoo_dir) / f"w_{idx}.npy", meta) for idx in order]
    n = len(arrays)
    grams, widths = {}, {}
    for _name, region, start, stop in tensor_layout(meta):
        for key in (region, "whole_model"):
            grams.setdefault(key, np.zeros((n, n)))
            widths[key] = widths.get(key, 0) + (stop - start)
        block = np.zeros((n, n))
        for left in range(start, stop, chunk_size):
            right = min(left + chunk_size, stop)
            x = np.array([w[left:right] for w in arrays], dtype=np.float64)
            if not np.isfinite(x).all():
                raise ValueError(f"Nonfinite weights in {region}")
            # Reference member order[0], taken from THIS chunk, so the subtraction
            # stays inside the chunk. Every coefficient row downstream sums to
            # zero, so the reference cancels exactly.
            x -= x[0].copy()
            # np.errstate: Accelerate BLAS (macOS, numpy 2.x) raises spurious
            # divide/overflow/invalid flags on finite matmuls.
            with np.errstate(all="ignore"):
                block += x @ x.T
        grams[region] += block
        grams["whole_model"] += block
    return grams, widths


# ---------------------------------------------------------------------------
# Subspace geometry, entirely from Gram blocks
# ---------------------------------------------------------------------------

def _whitener(matrix, rtol=1e-10):
    """W = V L^{-1/2} over resolved eigenvalues, and the rank.

    Rectangular (n-by-rank) rather than the symmetric inverse square root: it is
    what both statistics actually need, it drops a matrix product, and it is
    better conditioned -- the symmetric form cost ~1e-4 on the principal cosines.
    Rows of W' A are an orthonormal basis of the span, and G^+ = W W'.
    Eigenvalues are clipped at the resolved scale, which also absorbs the small
    negative values debiasing can leave behind.
    """
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    scale = max(float(values[-1]), 0.0)
    keep = values > scale * rtol
    if not keep.any():
        return np.zeros((len(matrix), 0)), 0
    return vectors[:, keep] * values[keep] ** -0.5, int(keep.sum())


def _projection_energy(cross, target_whitener, source_trace):
    """||A P_M||_F^2 / ||A||_F^2 == ||G_AM W_M||_F^2 / tr(G_A), since G_M^+ = W W'."""
    if not target_whitener.shape[1] or source_trace <= 0:
        return None
    with np.errstate(all="ignore"):               # see region_grams
        energy = float(np.square(cross @ target_whitener).sum())
    return min(max(energy / source_trace, 0.0), 1.0)


def containment(ga, gm, gam, ambient, rtol=1e-10):
    """Principal cosines and both energy fractions, from the three blocks."""
    wa, rank_a = _whitener(ga, rtol)
    wb, rank_m = _whitener(gm, rtol)
    with np.errstate(all="ignore"):               # see region_grams
        cosines = (np.clip(np.linalg.svd(wa.T @ gam @ wb, compute_uv=False), 0.0, 1.0)[
            :min(rank_a, rank_m)] if rank_a and rank_m else np.zeros(0))
    a_in_m = _projection_energy(gam, wb, float(np.trace(ga)))
    m_in_a = _projection_energy(gam.T, wa, float(np.trace(gm)))
    return dict(
        dim_anchor_span=rank_a, dim_mixture_span=rank_m,
        principal_cosines=[float(c) for c in cosines],
        smallest_principal_cosine=float(cosines[-1]) if len(cosines) else None,
        anchor_energy_in_mixture_span=a_in_m,
        mixture_energy_in_anchor_span=m_in_a,
        # A random rank-r subspace of R^ambient captures r/ambient of any fixed
        # vector's energy. Quote the fraction against THIS, never against zero --
        # the same correction composition.py's perpendicular_fraction needs.
        random_subspace_null_anchor_in_mixture=(rank_m / ambient) if ambient else None,
        random_subspace_null_mixture_in_anchor=(rank_a / ambient) if ambient else None)


def debias(ga, gm, trace_within, n_branches):
    """Remove the sampling-noise floor from the DIAGONAL blocks only.

    E[G_A] = A'A + (I - J/D) t_w / S     and     E[G_M] = M'M + (I - J/n) t_w.
    The cross block is unbiased already: anchors and mixtures are independent
    models, so their noise is uncorrelated.
    """
    d, n = len(ga), len(gm)
    pa = np.eye(d) - np.full((d, d), 1.0 / d)
    pm = np.eye(n) - np.full((n, n), 1.0 / n)
    return ga - pa * (trace_within / n_branches), gm - pm * trace_within


def spectrum_above_noise(gm_debiased, trace_within, n_mixtures):
    """How many mixture directions stand above the noise floor, against dim(simplex).

    Under the null that mixtures carry no structure at all, every centred
    direction has expected energy t_w, so that is the threshold.
    """
    values = np.sort(np.linalg.eigvalsh((gm_debiased + gm_debiased.T) / 2))[::-1]
    positive = values[values > 0]
    total = float(positive.sum())
    return dict(
        # The FULL spectrum, not a head. n_mixtures is 4 in the random control
        # but 80 in a real N=100 zoo, so a [:16] truncation dropped 64 of the
        # very directions mixture_dimension_above_noise counts -- and the tail
        # against the noise floor is the whole point of the statistic.
        mixture_eigenvalues=[float(v) for v in values],
        noise_floor_per_direction=float(trace_within),
        mixture_dimension_above_noise=int((values > trace_within).sum()),
        mixture_participation_ratio=(float(total ** 2 / float((positive ** 2).sum()))
                                     if len(positive) else None))


# ---------------------------------------------------------------------------

def analyze(zoo_dir, meta, chunk_size=65536, rtol=1e-10):
    order, group_sizes, mixtures, pi = member_tables(meta)
    la, lm, lw, d, s = coefficient_matrices(len(order), group_sizes, mixtures)
    grams, widths = region_grams(zoo_dir, meta, order, chunk_size)
    simplex = len(meta["domains"]) - 1
    rows = []
    for region, gram in grams.items():
        ambient = widths[region]
        with np.errstate(all="ignore"):            # see region_grams
            ga, gm, gam = la @ gram @ la.T, lm @ gram @ lm.T, la @ gram @ lm.T
            # tr(Sigma_w) from the within-anchor residuals: the only replicates.
            ss_within = float(np.trace(lw @ gram @ lw.T))
        trace_within = ss_within / (d * (s - 1))
        gad, gmd = debias(ga, gm, trace_within, s)
        row = dict(region=region, level="model" if region == "whole_model" else
                   ("extra" if region == "extra" else "layer"),
                   n_params=ambient, n_anchor_members=sum(group_sizes),
                   n_mixture_members=len(mixtures), n_domains=d, n_branches=s,
                   simplex_dimension=simplex, ss_within=ss_within,
                   trace_within=trace_within)
        for tag, a, m in (("raw", ga, gm), ("debiased", gad, gmd)):
            for key, value in containment(a, m, gam, ambient, rtol).items():
                row[f"{tag}_{key}"] = value
        row.update({f"debiased_{k}": v for k, v in
                    spectrum_above_noise(gmd, trace_within, len(mixtures)).items()})
        row["nonlinear_beyond_simplex"] = bool(
            row["debiased_mixture_dimension_above_noise"] > simplex)
        rows.append(row)
    return rows, order, mixtures


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", "--zoo-dir", dest="zoo_dir", type=Path,
                        required=True,
                        help="One architecture directory with zoo_meta.json and w_<i>.npy")
    parser.add_argument("--output-dir", type=Path, default=Path("containment_results"))
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--rank-rtol", type=float, default=1e-10,
                        help="Relative eigenvalue cutoff for resolved subspace rank")
    args = parser.parse_args()
    if args.chunk_size < 1 or not 0 < args.rank_rtol < 1:
        parser.error("chunk-size must be positive and 0 < rank-rtol < 1")
    meta = read_json(args.zoo_dir / "zoo_meta.json")
    rows, order, mixtures = analyze(args.zoo_dir, meta, args.chunk_size, args.rank_rtol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scalar = [{k: v for k, v in row.items() if not isinstance(v, list)} for row in rows]
    with (args.output_dir / "containment.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar[0]))
        writer.writeheader()
        writer.writerows(scalar)
    write_json(args.output_dir / "containment.json", dict(
        checkpoint_path=str(args.zoo_dir.resolve()), arch=meta["arch"],
        beta=meta.get("beta"), trunk_seed=meta.get("trunk_seed"),
        domains=meta["domains"], member_order=order, mixture_rows=mixtures, rows=rows,
        interpretation=(
            "A = span of centred anchor domain-mean effects; M = span of centred "
            "mixture displacements. Quote the DEBIASED columns: the raw ones carry "
            "a sampling-noise floor that prevents containment reading 1.0 even when "
            "it holds exactly. Energy fractions must be read against the "
            "random_subspace_null columns, not against zero."),
        caveat=(
            "A linear pi -> w map makes A and M the same <=dim(simplex) space, so "
            "containment is then trivial in both directions. The informative column "
            "is debiased_mixture_dimension_above_noise against simplex_dimension. "
            "Assumes a common within-member covariance, estimated from anchors only.")))
    whole = next(r for r in rows if r["region"] == "whole_model")
    print(f"{meta['arch']}: {whole['n_anchor_members']} anchors "
          f"({whole['n_domains']}x{whole['n_branches']}) + "
          f"{whole['n_mixture_members']} mixtures")
    print(f"  dim(A)={whole['debiased_dim_anchor_span']}  "
          f"dim(M)={whole['debiased_dim_mixture_span']}  "
          f"simplex dimension={whole['simplex_dimension']}")
    print(f"  anchor energy inside M : {whole['debiased_anchor_energy_in_mixture_span']}"
          f"   (random-subspace null {whole['debiased_random_subspace_null_anchor_in_mixture']:.3g})")
    print(f"  mixture dims above noise: {whole['debiased_mixture_dimension_above_noise']}"
          f"  -> beyond the simplex: {whole['nonlinear_beyond_simplex']}")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

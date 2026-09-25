#!/usr/bin/env python3
"""Figure 3: mixture prediction residual within/outside the domain-mean subspace."""
import argparse
from pathlib import Path
import tempfile

import numpy as np

from llmzoo.analysis.mixture_composition.zoo import (add_arguments, anchor_groups, load_model, open_weights, read_json,
                 save_flat, tensor_layout, validate_model, weight_dataset, write_results, plot_layers)


def projection_energy(gram, cross, squared_norm, rank_rtol=1e-8):
    """Project using A.T A and A.T r, without materializing a P-by-D basis."""
    values, vectors = np.linalg.eigh((gram + gram.T) / 2)
    keep = values > max(float(values[-1]), 0.0) * rank_rtol
    coordinates = vectors[:, keep].T @ cross
    parallel = float(np.sum(coordinates ** 2 / values[keep])) if keep.any() else 0.0
    if parallel > squared_norm + 1e-7 * squared_norm + 1e-12:
        raise ValueError("Projection energy exceeds residual energy; inconsistent or unstable statistics")
    parallel = min(max(parallel, 0.0), squared_norm)
    return parallel, max(squared_norm - parallel, 0.0), int(keep.sum())


def analyze_composition(root, meta, member_ids=None, reference=None,
                        chunk_size=65536, rank_rtol=1e-8, token_weights=None):
    if chunk_size < 1 or not 0 < rank_rtol < 1:
        raise ValueError("chunk_size must be positive and 0 < rank_rtol < 1")
    if reference is not None and np.shape(reference) != (meta["n_params"],):
        raise ValueError("Reference must be a flat vector matching the complete saved layout")
    groups = anchor_groups(meta)
    d, s = len(groups), len(groups[0])
    anchors = [[open_weights(root / f"w_{idx}.npy", meta) for idx in group] for group in groups]
    candidates = {int(m["idx"]): m for m in meta["members"] if m["kind"] == "singleton"}
    ids = sorted(candidates) if member_ids is None else member_ids
    if not ids or len(set(ids)) != len(ids) or any(i not in candidates for i in ids):
        raise ValueError("Select distinct singleton member IDs present in zoo_meta.json")
    mixtures = [open_weights(root / f"w_{idx}.npy", meta) for idx in ids]
    pi = np.array([token_weights[str(i)] if token_weights is not None else candidates[i]["pi"]
                   for i in ids], dtype=np.float64)
    if pi.shape != (len(ids), d) or not np.isfinite(pi).all() or (pi < 0).any() or not np.allclose(pi.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("Each mixture must contain D nonnegative finite proportions summing to 1")
    # Remove accepted decimal-rounding error before using translation invariance.
    pi /= pi.sum(axis=1, keepdims=True)
    accumulators = {}
    for name, shape, start, stop, group in tensor_layout(meta):
        gram = np.zeros((d, d))
        cross = np.zeros((d, len(ids)))
        residual_ss = np.zeros(len(ids))
        displacement_ss = np.zeros(len(ids))
        within = 0.0
        for left in range(start, stop, chunk_size):
            right = min(left + chunk_size, stop)
            x = np.array([[w[left:right] for w in domain] for domain in anchors], dtype=np.float64)
            y = np.array([w[left:right] for w in mixtures], dtype=np.float64)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"Nonfinite checkpoint weights in {name}")
            offset = x[0, 0].copy()
            x -= offset
            y -= offset
            means = x.mean(axis=1)
            effects = means - means.mean(axis=0)
            residual = y - pi @ means
            gram += effects @ effects.T
            cross += effects @ residual.T
            residual_ss += np.square(residual).sum(axis=1)
            within += float(np.square(x - means[:, None]).sum())
            if reference is not None:
                ref = np.asarray(reference[left:right], dtype=np.float64)
                if not np.isfinite(ref).all():
                    raise ValueError("Nonfinite reference weights")
                displacement_ss += np.square(y - (ref - offset)).sum(axis=1)
        for key in (("tensor", name), ("layer", group), ("model", "whole_model")):
            if key not in accumulators:
                accumulators[key] = [0, np.zeros_like(gram), np.zeros_like(cross),
                                     np.zeros_like(residual_ss), np.zeros_like(displacement_ss), 0.0]
            acc = accumulators[key]
            for i, value in enumerate((stop - start, gram, cross, residual_ss, displacement_ss, within)):
                acc[i] += value
    rows = []
    for (level, name), (size, gram, cross, residual_ss, displacement_ss, within) in accumulators.items():
        ms_within = within / (d * (s - 1))
        for j, idx in enumerate(ids):
            parallel, perpendicular, rank = projection_energy(gram, cross[:, j], float(residual_ss[j]), rank_rtol)
            expected_noise = ms_within * (1 + float(pi[j] @ pi[j]) / s)
            rows.append(dict(level=level, name=name, member_idx=idx, n_params=size,
                             projection_rank=rank, residual_squared=float(residual_ss[j]),
                             parallel_squared=parallel, perpendicular_squared=perpendicular,
                             perpendicular_fraction=perpendicular / float(residual_ss[j]) if residual_ss[j] else None,
                             relative_residual=float(np.sqrt(residual_ss[j] / displacement_ss[j])) if reference is not None and displacement_ss[j] else None,
                             expected_noise_squared=expected_noise,
                             residual_to_noise=float(np.sqrt(residual_ss[j] / expected_noise)) if expected_noise else None))
    return rows, {str(i): pi[j].tolist() for j, i in enumerate(ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, "composition_results")
    parser.add_argument("--members", type=int, nargs="+", help="Singleton member indices; default: all")
    parser.add_argument("--reference-checkpoint", type=Path,
                        help="Trusted trunk.pt or compatible flat .npy for update normalization; auto-detect trunk.pt")
    parser.add_argument("--token-weights", type=Path,
                        help='Optional JSON of measured token proportions: {"20": [0.2, ...]}')
    parser.add_argument("--rank-rtol", type=float, default=1e-8,
                        help="Relative Gram eigenvalue cutoff for stable projection")
    args = parser.parse_args()
    with weight_dataset(args, mixtures=True) as (root, meta, source):
        reference_path = args.reference_checkpoint
        if reference_path is None:
            # Auto-detect a FLAT reference only. trunk.pt used to be picked up
            # here and torch.load(weights_only=False)'d without the caller
            # asking, which pulls torch and transformers into an otherwise
            # NumPy-only analysis that is meant to run without torch.
            # scripts/diag_zoo_geometry.py --dump_trunk_flat writes trunk_flat.npy
            # through the same flatten_model path that wrote every w_i.npy.
            for name in ("reference.npy" if source == "random_initialization"
                         else "trunk_flat.npy",):
                if (root / name).exists():
                    reference_path = root / name
            if reference_path is None and (root / "trunk.pt").exists():
                print(f"Note: {root / 'trunk.pt'} exists but is NOT auto-loaded "
                      f"(it would require torch). Run\n"
                      f"  python scripts/diag_zoo_geometry.py --zoo_dir {root} "
                      f"--dump_trunk_flat\n"
                      f"once, or pass --reference-checkpoint explicitly. "
                      f"Proceeding with no reference: relative_residual will be "
                      f"null and residuals are measured about the domain-mean "
                      f"affine space rather than about the trunk.", flush=True)
        with tempfile.TemporaryDirectory(prefix="composition_reference_", dir=args.work_dir) as work:
            reference = None
            if reference_path is not None:
                if reference_path.suffix == ".pt":
                    model = load_model(checkpoint_path=reference_path, config_path=args.config)
                    validate_model(model, meta)
                    flat = Path(work) / "reference.npy"
                    save_flat(model, flat, meta)
                    del model
                    reference = open_weights(flat, meta)
                elif reference_path.suffix == ".npy":
                    reference = open_weights(reference_path, meta)
                else:
                    raise ValueError("Reference must be .pt or flat .npy")
            token_weights = read_json(args.token_weights) if args.token_weights else None
            rows, weights = analyze_composition(root, meta, args.members, reference,
                                                args.chunk_size, args.rank_rtol, token_weights)
            write_results(args, meta, source, "composition", rows,
                          mixture_weights=weights, coefficient_source="measured_tokens" if token_weights else "document_sampling",
                          reference_checkpoint=str(reference_path.resolve()) if reference_path and (args.reference_checkpoint or source != "random_initialization") else None,
                          reference_kind=("checkpoint" if args.reference_checkpoint else
                                          "independent_random_initialization" if source == "random_initialization" and reference_path else
                                          "checkpoint" if reference_path else "none"),
                          rank_rtol=args.rank_rtol,
                          interpretation="Parallel error tests mixture coefficients; perpendicular error tests departures from the estimated domain-mean affine space. Noise reference assumes common covariance, not a significance test.")
            plot_layers(args.output_dir, rows, "Mean mixture prediction error / noise scale", source,
                        ["residual_to_noise"])


if __name__ == "__main__":
    main()

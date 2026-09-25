#!/usr/bin/env python3
"""Figure 2: spectrum of centered domain means, per tensor/layer/model."""
import argparse

import numpy as np

from llmzoo.analysis.domain_geometry.zoo import (add_arguments, anchor_groups, open_weights, tensor_layout,
                 weight_dataset, write_results, plot_layers)


def spectrum(gram):
    """Eigenvalues are squared singular values of the P-by-D effect matrix."""
    values = np.maximum(np.linalg.eigvalsh((gram + gram.T) / 2)[::-1], 0)
    # Centered D domain means have at most D-1 directions.
    values = values[:len(gram) - 1]
    total = float(values.sum())
    fractions = values / total if total else np.zeros_like(values)
    return values, fractions, float(1 / (fractions @ fractions)) if total else None


def analyze_geometry(root, meta, chunk_size=65536):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    groups = anchor_groups(meta)
    weights = [[open_weights(root / f"w_{idx}.npy", meta) for idx in group] for group in groups]
    d, s = len(groups), len(groups[0])
    accumulators = {}
    for name, shape, start, stop, group in tensor_layout(meta):
        gram = np.zeros((d, d))
        within = 0.0
        for left in range(start, stop, chunk_size):
            right = min(left + chunk_size, stop)
            x = np.array([[w[left:right] for w in domain] for domain in weights], dtype=np.float64)
            if not np.isfinite(x).all():
                raise ValueError(f"Nonfinite weights in {name}")
            x -= x[0, 0].copy()
            means = x.mean(axis=1)
            effects = means - means.mean(axis=0)
            gram += effects @ effects.T
            within += float(np.square(x - means[:, None]).sum())
        for key in (("tensor", name), ("layer", group), ("model", "whole_model")):
            if key not in accumulators:
                accumulators[key] = [0, np.zeros((d, d)), 0.0]
            accumulators[key][0] += stop - start
            accumulators[key][1] += gram
            accumulators[key][2] += within
    rows = []
    for (level, name), (size, gram, within) in accumulators.items():
        values, fractions, rank = spectrum(gram)
        total = float(values.sum())
        between = s * total
        row = dict(level=level, name=name, n_params=size,
                   effective_rank=rank, rank_bound=d - 1,
                   k90=int(np.searchsorted(np.cumsum(fractions), 0.9) + 1) if total else None,
                   ss_between=between, ss_within=within,
                   between_fraction=between / (between + within) if between + within else None)
        for i, (value, fraction) in enumerate(zip(values, fractions), 1):
            row[f"eigenvalue_{i}"] = float(value)
            row[f"explained_fraction_{i}"] = float(fraction) if total else None
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, "geometry_results")
    args = parser.parse_args()
    with weight_dataset(args) as (root, meta, source):
        rows = analyze_geometry(root, meta, args.chunk_size)
        write_results(args, meta, source, "geometry", rows,
                      interpretation="Spectrum of estimated centered domain means; finite-run noise remains. Rank <= D-1 is structural, not evidence of universality.")
        plot_layers(args.output_dir, rows, "Effective rank of centered domain means", source,
                    ["effective_rank"])


if __name__ == "__main__":
    main()

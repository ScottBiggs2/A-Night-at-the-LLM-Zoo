#!/usr/bin/env python3
"""Figure 1: between-domain versus within-domain weight variation."""
import argparse

from llmzoo.analysis.anovas.zoo import (add_arguments, analyze, anchor_groups, load_model, sums_of_squares,
                 summarize, weight_dataset, write_results, plot_layers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, "anova_results")
    args = parser.parse_args()
    with weight_dataset(args) as (root, meta, source):
        groups, rows = analyze(root, meta, args.chunk_size)
        write_results(args, meta, source, "anova", rows, member_indices=groups,
                      df_between=len(groups) - 1,
                      df_within=len(groups) * (len(groups[0]) - 1),
                      interpretation="Observed between-domain and within-domain fractions; corrected estimates assume independent repetitions with common covariance.")
        plot_layers(args.output_dir, rows, "Observed variance fractions", source,
                    ["between_fraction", "within_fraction"])


if __name__ == "__main__":
    main()

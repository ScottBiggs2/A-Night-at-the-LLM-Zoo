"""Shared NumPy/sklearn plotting utilities; no repository imports."""
import csv
import json
from pathlib import Path

import numpy as np


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def centered_gram(features, chunk_size=65536):
    """Exact feature centering; no feature standardization or normalization."""
    gram = np.zeros((len(features), len(features)), dtype=np.float64)
    for start in range(0, features.shape[1], chunk_size):
        x = np.array(features[:, start:start + chunk_size], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite feature values")
        x -= x[0].copy()
        x -= x.mean(axis=0)
        gram += x @ x.T
    return gram


def pca_from_gram(gram):
    """Return all numerically resolved centered PCA scores and eigenvalues."""
    values, vectors = np.linalg.eigh((gram + gram.T) / 2)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    scale = max(float(values[0]), 0.0)
    if values[-1] < -max(scale * 1e-9, 1e-12):
        raise ValueError("Gram matrix is not positive semidefinite")
    keep = values > scale * 1e-12
    values, vectors = values[keep], vectors[:, keep]
    scores = vectors * np.sqrt(values)[None, :]
    return scores, values


def centroid_geometry(gram, labels, output):
    """Distances and centered cosines between domain/run feature centroids."""
    for name, keys in (("domains", [(r["domain"],) for r in labels]),
                       ("domain_runs", [(r["domain"], str(r["run_id"])) for r in labels])):
        groups = list(dict.fromkeys(keys))
        averaging = np.zeros((len(groups), len(labels)))
        for i, key in enumerate(groups):
            mask = np.array([k == key for k in keys])
            averaging[i, mask] = 1 / mask.sum()
        # np.errstate: Accelerate BLAS (macOS, numpy 2.x) raises spurious
        # divide/overflow/invalid flags on this matmul. Inputs and result are
        # finite and match an explicit per-group mean Gram exactly.
        with np.errstate(all="ignore"):
            g = averaging @ gram @ averaging.T
        diagonal = np.maximum(np.diag(g), 0)
        squared = np.maximum(diagonal[:, None] + diagonal[None, :] - 2*g, 0)
        np.fill_diagonal(squared, 0)
        norms = np.sqrt(diagonal[:, None] * diagonal[None, :])
        cosine = np.divide(g, norms, out=np.full_like(g, np.nan), where=norms > 0)
        cosine = np.clip(cosine, -1, 1)
        np.savez(output / f"{name}_geometry.npz", distances=np.sqrt(squared),
                 centered_cosines=cosine, groups=np.array(["/".join(g) for g in groups]))


def plot_coordinates(coordinates, labels, output, method, scope, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    domains = list(dict.fromkeys(row["domain"] for row in labels))
    runs = list(dict.fromkeys(str(row["run_id"]) for row in labels))
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]
    colors = plt.get_cmap("tab20")
    def scatter(ax, chosen):
        for d, domain in enumerate(domains):
            for r, run in enumerate(runs):
                mask = np.array([row["domain"] == domain and str(row["run_id"]) == run
                                 and i in chosen for i, row in enumerate(labels)])
                if mask.any():
                    ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=25,
                               alpha=.65, c=[colors(d % 20)], marker=markers[r % len(markers)],
                               label=f"{domain} / run {run}")
        ax.set_xlabel("PC1" if method == "pca" else "t-SNE 1")
        ax.set_ylabel("PC2" if method == "pca" else "t-SNE 2")
        ax.grid(alpha=.15)
    if scope in ("both", "cross-domain"):
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter(ax, set(range(len(labels))))
        ax.set_title(title)
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, .5))
        fig.savefig(output / f"{method}_cross_domain.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    if scope in ("both", "within-domain"):
        fig, axes = plt.subplots(1, len(domains), figsize=(5*len(domains), 4), squeeze=False,
                                 sharex=True, sharey=True)
        for ax, domain in zip(axes[0], domains):
            scatter(ax, {i for i, row in enumerate(labels) if row["domain"] == domain})
            ax.set_title(domain)
            ax.legend(fontsize=7)
        fig.suptitle(title + " — within-domain views of the SAME projection")
        fig.tight_layout()
        fig.savefig(output / f"{method}_within_domain.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def reduce_and_save(gram, labels, output, *, method="both", seed=0,
                    perplexity=30.0, tsne_pca_dim=50, scope="both", title="Embeddings"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    n = len(labels)
    if n < 2 or gram.shape != (n, n) or not np.isfinite(gram).all():
        raise ValueError("Need at least two samples and a finite N-by-N Gram matrix")
    scores, values = pca_from_gram(gram)
    np.save(output / "centered_gram.npy", gram)
    np.save(output / "pca_scores.npy", scores)
    np.save(output / "pca_eigenvalues.npy", values)
    write_json(output / "labels.json", labels)
    centroid_geometry(gram, labels, output)
    metadata = dict(n_samples=n, resolved_rank=len(values), seed=seed,
                    centering="global feature mean; no per-feature standardization",
                    explained_variance_ratio=(values/values.sum()).tolist() if len(values) else [])
    if not len(values):
        metadata["status"] = "No resolved variation; zero PCA coordinates, t-SNE skipped."
    methods = ["pca", "tsne"] if method == "both" else [method]
    for chosen in methods:
        if chosen == "pca":
            coordinates = np.zeros((n, 2))
            coordinates[:, :min(2, scores.shape[1])] = scores[:, :2]
        else:
            if not len(values):
                continue
            from sklearn.manifold import TSNE
            if perplexity <= 0 or tsne_pca_dim < 0:
                raise ValueError("perplexity must be positive; tsne-pca-dim must be nonnegative")
            # Adapt for small checkpoint ensembles; record the actual value.
            actual_perplexity = min(perplexity, max(1., (n-1)/3))
            dimensions = min(tsne_pca_dim, scores.shape[1]) if tsne_pca_dim else scores.shape[1]
            data = scores[:, :dimensions]
            coordinates = TSNE(n_components=2, perplexity=actual_perplexity,
                               init="random", learning_rate="auto", random_state=seed).fit_transform(data)
            metadata.update(tsne_perplexity=actual_perplexity, tsne_pca_dimensions=dimensions,
                            tsne_retained_variance=float(values[:dimensions].sum()/values.sum()))
        np.save(output / f"{chosen}_coordinates.npy", coordinates)
        with (output / f"{chosen}_coordinates.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*labels[0], "x", "y"])
            writer.writeheader()
            for label, (x, y) in zip(labels, coordinates):
                writer.writerow(dict(label, x=float(x), y=float(y)))
        try:
            plot_coordinates(coordinates, labels, output, chosen, scope, title)
        except ImportError as exc:            # never discard encoded features
            metadata.setdefault("plot_errors", {})[chosen] = str(exc)
            print(f"WARNING: {chosen} plot skipped ({exc}); "
                  f"coordinates and Gram are still saved in {output}", flush=True)
    write_json(output / "projection.json", metadata)


def require_projection_dependencies(method):
    """Check plotting imports BEFORE expensive encoding, not after it.

    data_embeddings.py streams datasets, loads an encoder and GPU-encodes
    thousands of documents before it ever reaches a plot. matplotlib was an
    undeclared dependency imported inside plot_coordinates, so a missing install
    surfaced only at the very end and threw the whole run away.
    """
    missing = []
    try:
        import matplotlib                                         # noqa: F401
    except ImportError:
        missing.append("matplotlib")
    if method in ("tsne", "both"):
        try:
            from sklearn.manifold import TSNE                     # noqa: F401
        except ImportError:
            missing.append("scikit-learn")
    if missing:
        raise SystemExit(
            "Missing projection dependencies: " + ", ".join(missing)
            + "\nInstall them (pip install " + " ".join(missing) + "), or pass"
              " --method pca if only scikit-learn is missing.")


def add_projection_arguments(parser):
    parser.add_argument("--method", choices=["pca", "tsne", "both"], default="both")
    parser.add_argument("--scope", choices=["cross-domain", "within-domain", "both"], default="both")
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--tsne-pca-dim", type=int, default=50,
                        help="PCA dimensions before t-SNE; 0 keeps all resolved dimensions")


def project(gram, labels, args, title):
    reduce_and_save(gram, labels, args.output_dir, method=args.method,
                    seed=args.projection_seed, perplexity=args.perplexity,
                    tsne_pca_dim=args.tsne_pca_dim, scope=args.scope, title=title)

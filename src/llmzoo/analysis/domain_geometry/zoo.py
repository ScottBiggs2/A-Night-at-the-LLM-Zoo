#!/usr/bin/env python3
"""One-way weight-space ANOVA for a zoo's repeated single-domain branches.

Standalone: no llmzoo imports, network access, PCA, or training.
"""

import argparse
import csv
import html
import json
import math
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np


ARCHITECTURES = {
    "gpt2_zoo_mini": (8, 512, 8),
    "gpt2_zoo_small": (12, 768, 12),
    "gpt2_zoo_medium": (24, 1024, 16),
}


def read_json(path):
    return json.loads(Path(path).read_text())


def tensor_layout(meta):
    """Yield (parameter name, shape, start, stop, group) in saved order."""
    if meta["layout_version"] != 2:
        raise ValueError("Expected zoo layout_version=2")
    offset = 0
    names = set()
    schemas = [("", "extra", meta["extra_schema"])] + [
        (f"transformer.h.{i}.", f"layer_{i}", meta["schema"])
        for i in range(meta["n_layers"])
    ]
    for prefix, group, schema in schemas:
        begin = offset
        for name, shape in schema:
            name = prefix + name
            if name in names or not shape or any(int(n) <= 0 for n in shape):
                raise ValueError(f"Invalid or duplicate tensor: {name}, {shape}")
            names.add(name)
            size = math.prod(shape)
            yield name, tuple(shape), offset, offset + size, group
            offset += size
        expected = meta["extra_size"] if group == "extra" else meta["block_size"]
        if offset - begin != expected:
            raise ValueError(f"Schema size mismatch for {group}")
    if offset != meta["n_params"]:
        raise ValueError("Schema does not cover n_params")


def open_weights(path, meta):
    weights = np.load(path, mmap_mode="r", allow_pickle=False)
    if weights.shape != (meta["n_params"],) or weights.dtype != np.float32:
        raise ValueError(f"{path}: expected float32 vector of length {meta['n_params']}")
    return weights


def validate_model(model, meta):
    """Check metadata against the same parameter set the zoo exporter uses."""
    actual = {name: tuple(p.shape) for name, p in model.named_parameters()
              if (not meta["exclude_1d"] or p.ndim > 1)
              and (meta["include_extra"] or name.startswith("transformer.h."))}
    saved = {name: shape for name, shape, *_ in tensor_layout(meta)}
    if actual != saved:
        raise ValueError("Model configuration and saved tensor schema disagree")


def load_model(arch="gpt2_zoo_mini", checkpoint_path=None, *,
               config_path=None, seed=1234):
    """Build the zoo GPT-2; optionally restore a w_i.npy or trusted trunk.pt.

    Without a checkpoint this uses the same HF GPT-2 initialization as zoo
    pretraining. A .npy file requires zoo_meta.json beside it. A custom config
    is a Hugging Face GPT2Config JSON file. Imports stay lazy for NumPy-only use.
    """
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    path = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpoint = None
    meta = None
    if path is not None:
        if path.suffix == ".npy":
            meta = read_json(path.parent / "zoo_meta.json")
            arch = meta["arch"]
        elif path.suffix == ".pt":
            # Zoo trunk checkpoints include optimizer and non-tensor metadata.
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            arch = checkpoint["arch"]
        else:
            raise ValueError("Checkpoint must be a zoo w_i.npy or trunk.pt")
    if config_path is not None:
        config = GPT2Config.from_dict(read_json(config_path))
    elif checkpoint is not None:
        config = GPT2Config.from_dict(checkpoint["config"])
    elif meta is not None and meta.get("config"):
        config = GPT2Config.from_dict(meta["config"])
    else:
        layers, width, heads = ARCHITECTURES[arch]
        config = GPT2Config(
            vocab_size=50257, n_positions=1024, n_embd=width,
            n_layer=layers, n_head=heads, n_inner=4 * width,
            activation_function="gelu_new", resid_pdrop=0.0,
            embd_pdrop=0.0, attn_pdrop=0.0, layer_norm_epsilon=1e-5,
            initializer_range=0.02, tie_word_embeddings=True,
        )
    torch.manual_seed(seed)
    model = GPT2LMHeadModel(config)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
    elif meta is not None:
        validate_model(model, meta)
        weights = open_weights(path, meta)
        params = dict(model.named_parameters())
        with torch.no_grad():
            for name, shape, start, stop, _ in tensor_layout(meta):
                value = np.array(weights[start:stop]).reshape(shape)
                if not np.isfinite(value).all():
                    raise ValueError(f"Nonfinite weights in {path}: {name}")
                params[name].copy_(torch.from_numpy(value))
    return model.eval()


def anchor_groups(meta):
    """Require a complete balanced set of repeated one-hot domains."""
    domains = meta["domains"]
    if len(domains) < 2 or len(set(domains)) != len(domains):
        raise ValueError("Need at least two distinct domains")
    groups = [[] for _ in domains]
    seen = set()
    for member in meta["members"]:
        idx = member["idx"]
        if type(idx) is not int or idx < 0 or idx in seen:
            raise ValueError(f"Invalid or duplicate member index: {idx}")
        seen.add(idx)
        if member["kind"] != "anchor":
            continue
        pi = np.asarray(member["pi"], dtype=np.float64)
        if pi.shape != (len(domains),) or not np.isfinite(pi).all():
            raise ValueError(f"Invalid mixture for anchor {idx}")
        domain = int(pi.argmax())
        if not np.array_equal(pi, np.eye(len(domains))[domain]):
            raise ValueError(f"Invalid one-hot or duplicate anchor {idx}")
        groups[domain].append(idx)
    sizes = {len(group) for group in groups}
    if len(sizes) != 1 or min(sizes) < 2:
        raise ValueError("Need equal numbers of anchors per domain, at least two")
    if sizes != {meta["n_anchor_branches"]}:
        raise ValueError("Anchor count disagrees with n_anchor_branches")
    return [sorted(group) for group in groups]


def sums_of_squares(values):
    """values has shape (domains, repetitions, coordinates); return B, W, T."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or min(values.shape[:2]) < 2:
        raise ValueError("Expected (D, S, P), with D and S >= 2")
    if not np.isfinite(values).all():
        raise ValueError("Weights contain NaN or infinity")
    # Translation leaves ANOVA unchanged; subtract a common vector for stability.
    values = values - values[0, 0]
    means = values.mean(axis=1)
    grand = means.mean(axis=0)
    between = values.shape[1] * np.square(means - grand).sum()
    within = np.square(values - means[:, None, :]).sum()
    total = np.square(values - grand).sum()
    return np.array([between, within, total], dtype=np.float64)


def summarize(name, level, n_params, ss, d, s):
    between, within, total = map(float, ss)
    if not math.isclose(between + within, total, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"ANOVA identity failed for {name}")
    ms_within = within / (d * (s - 1))
    domain_variance = (between - (d - 1) * ms_within) / (d * s)
    denominator = domain_variance + ms_within
    return dict(
        level=level, name=name, n_params=n_params,
        ss_between=between, ss_within=within, ss_total=total,
        between_fraction=between / total if total else None,
        within_fraction=within / total if total else None,
        ms_within=ms_within,
        domain_variance_corrected=domain_variance,
        domain_fraction_corrected=domain_variance / denominator if denominator > 0 else None,
    )


def analyze(zoo_dir, meta, chunk_size=65536):
    """Read saved coordinates directly; memory is O(D*S*chunk_size)."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    groups = anchor_groups(meta)
    layout = list(tensor_layout(meta))
    arrays = [[open_weights(Path(zoo_dir) / f"w_{idx}.npy", meta)
               for idx in group] for group in groups]
    d, s = len(groups), len(groups[0])
    rows = []
    aggregates = {}
    for name, shape, start, stop, group in layout:
        ss = np.zeros(3, dtype=np.float64)
        for left in range(start, stop, chunk_size):
            right = min(left + chunk_size, stop)
            values = np.array([[w[left:right] for w in domain] for domain in arrays],
                              dtype=np.float64)
            ss += sums_of_squares(values)
        rows.append(summarize(name, "tensor", stop - start, ss, d, s))
        for key in (group, "whole_model"):
            if key not in aggregates:
                aggregates[key] = [0, np.zeros(3, dtype=np.float64)]
            aggregates[key][0] += stop - start
            aggregates[key][1] += ss
    for name, (size, ss) in aggregates.items():
        rows.append(summarize(name, "model" if name == "whole_model" else "layer",
                              size, ss, d, s))
    return groups, rows


def add_arguments(parser, output_name):
    parser.add_argument("--checkpoint-path", "--dataset-path", "--zoo-dir",
                        dest="zoo_dir", type=Path,
                        help="Saved weight dataset: directory with zoo_meta.json and w_i.npy")
    parser.add_argument("--arch", choices=ARCHITECTURES, default="gpt2_zoo_mini")
    parser.add_argument("--config", type=Path, help="Optional GPT2Config JSON")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, default=Path(output_name))
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--work-dir", type=Path,
                        help="Parent of temporary random-weight files; needs several GB for default Mini")
    parser.add_argument("--random-mixtures", type=int, default=4,
                        help="Number of random control mixtures (composition only)")


def model_metadata(model, arch, seed, n_mixtures):
    """Derive exactly the zoo's [extra | transformer blocks] layout."""
    extra = [[name, list(p.shape)] for name, p in model.named_parameters()
             if not name.startswith("transformer.h.")]
    schema = [[name, list(p.shape)] for name, p in model.transformer.h[0].named_parameters()]
    domains = ["web", "code", "math", "books", "multilingual"]
    members = [dict(idx=d * 4 + r, kind="anchor", pi=np.eye(5)[d].tolist(),
                    mixture_id=f"anchor_{domain}", branch=r)
               for d, domain in enumerate(domains) for r in range(4)]
    rng = np.random.default_rng(seed)
    for i in range(n_mixtures):
        members.append(dict(idx=20 + i, kind="singleton", pi=rng.dirichlet(np.ones(5)).tolist(),
                            mixture_id=f"random_mixture_{i}", branch=0))
    extra_size = sum(math.prod(shape) for _, shape in extra)
    block_size = sum(math.prod(shape) for _, shape in schema)
    return dict(layout_version=2, arch=arch, beta=None, trunk_seed=None,
                n_layers=len(model.transformer.h), block_size=block_size,
                extra_size=extra_size, n_params=extra_size + len(model.transformer.h) * block_size,
                schema=schema, extra_schema=extra, exclude_1d=False, include_extra=True,
                domains=domains, members=members, n_anchor_branches=4,
                source="random_initialization", initialization_seed=seed,
                config=model.config.to_dict())


def save_flat(model, path, meta):
    params = dict(model.named_parameters())
    weights = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                       shape=(meta["n_params"],))
    for name, shape, start, stop, _ in tensor_layout(meta):
        weights[start:stop] = params[name].detach().cpu().float().numpy().reshape(-1)
    weights.flush()
    del weights


@contextmanager
def weight_dataset(args, mixtures=False):
    """Use a real zoo, or sequentially initialize independent null-control models."""
    if args.chunk_size <= 0 or args.random_mixtures < 0:
        raise ValueError("chunk-size must be positive and random-mixtures nonnegative")
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.zoo_dir is not None:
        meta = read_json(args.zoo_dir / "zoo_meta.json")
        # The saved schema is sufficient for numeric analysis; no model download.
        list(tensor_layout(meta))
        anchor_groups(meta)
        if args.config:
            model = load_model(meta["arch"], config_path=args.config, seed=args.seed)
            validate_model(model, meta)
            del model
        source = meta.get("source", "pretrained_zoo")
        if source not in ("pretrained_zoo", "random_initialization"):
            raise ValueError(f"Unknown weight dataset source: {source}")
        yield args.zoo_dir, meta, source
        return
    with tempfile.TemporaryDirectory(prefix="anova_random_", dir=args.work_dir) as directory:
        root = Path(directory)
        n_mixtures = args.random_mixtures if mixtures else 0
        model = load_model(args.arch, config_path=args.config, seed=args.seed)
        meta = model_metadata(model, args.arch, args.seed, n_mixtures)
        save_flat(model, root / "reference.npy", meta)
        del model
        print("Random-initialization control: domain/mixture labels are artificial; no training.", flush=True)
        for member in meta["members"]:
            seed = args.seed + 1 + member["idx"]
            model = load_model(args.arch, config_path=args.config, seed=seed)
            validate_model(model, meta)
            save_flat(model, root / f"w_{member['idx']}.npy", meta)
            del model
            print(f"Initialized {member['idx'] + 1}/{len(meta['members'])}", flush=True)
        (root / "zoo_meta.json").write_text(json.dumps(meta))
        yield root, meta, "random_initialization"


def write_results(args, meta, source, name, rows, **details):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(source=source, checkpoint_path=str(args.zoo_dir.resolve()) if args.zoo_dir else None,
                  arch=meta["arch"], beta=meta.get("beta"), trunk_seed=meta.get("trunk_seed"),
                  initialization_seed=meta.get("initialization_seed") if source == "random_initialization" else None,
                  config_validation_seed=args.seed if args.config else None,
                  domains=meta["domains"],
                  config=read_json(args.config) if args.config else meta.get("config"),
                  members=meta["members"], rows=rows, **details)
    (args.output_dir / f"{name}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if rows:
        with (args.output_dir / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Results: {args.output_dir.resolve()}")


def plot_layers(output_dir, rows, title, source, fields):
    """Small dependency-free SVG; repeated mixture rows are averaged per layer."""
    layer_rows = [r for r in rows if r["level"] == "layer" and r["name"].startswith("layer_")]
    indices = sorted({int(r["name"].split("_")[1]) for r in layer_rows})
    series = {}
    for field in fields:
        series[field] = []
        for index in indices:
            values = [r[field] for r in layer_rows if r["name"] == f"layer_{index}"
                      and r[field] is not None]
            series[field].append(float(np.mean(values)) if values else None)
    values = [v for line in series.values() for v in line if v is not None]
    ymax = max(max(values, default=1.0) * 1.1, 1e-12)
    width, height = 900, 500
    def x(i):
        return 80 + 740 * i / max(len(indices) - 1, 1)
    def y(v):
        return 370 - 270 * v / ymax
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" fill="#172033">',
             f'<text x="80" y="32" font-size="21">{html.escape(title)}</text>',
             f'<text x="80" y="58" font-size="13">{html.escape(source)} · layer indices start at 0</text>']
    for tick in range(6):
        value = ymax * tick / 5
        parts.append(f'<path d="M80 {y(value):.2f} H820" stroke="#e2e5eb"/>')
        parts.append(f'<text x="70" y="{y(value)+4:.2f}" text-anchor="end" font-size="12">{value:.3g}</text>')
    for i, index in enumerate(indices):
        parts.append(f'<text x="{x(i):.2f}" y="392" text-anchor="middle" font-size="12">{index}</text>')
    for j, (field, line) in enumerate(series.items()):
        color = ["#2563eb", "#d97706", "#0f766e"][j % 3]
        commands = []
        connected = False
        for i, value in enumerate(line):
            if value is None:
                connected = False
                continue
            commands.append(f'{"L" if connected else "M"}{x(i):.2f},{y(value):.2f}')
            connected = True
            parts.append(f'<circle cx="{x(i):.2f}" cy="{y(value):.2f}" r="3" fill="{color}"/>')
        parts.append(f'<path d="{" ".join(commands)}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="80" y="{425 + j*20}" font-size="13" fill="{color}">{html.escape(field)}</text>')
    if not values:
        parts.append('<text x="180" y="210">No defined values (zero variation or unavailable baseline).</text>')
    parts.append('</g></svg>')
    (Path(output_dir) / "figure.svg").write_text("\n".join(parts) + "\n")

#!/usr/bin/env python3
"""PCA/t-SNE of zoo checkpoints: each point is one model, never one layer."""
import argparse
import fnmatch
import json
import math
from pathlib import Path

import numpy as np

from llmzoo.analysis.Embeddings.embedding_utils import add_projection_arguments, require_projection_dependencies, project, write_json


def tensor_layout(meta):
    if meta["layout_version"] != 2:
        raise ValueError("Expected zoo layout_version=2")
    offset, seen = 0, set()
    groups = [(None, "", meta["extra_schema"], meta["extra_size"])] + [
        (i, f"transformer.h.{i}.", meta["schema"], meta["block_size"])
        for i in range(meta["n_layers"])]
    for layer, prefix, schema, expected_size in groups:
        begin = offset
        for name, shape in schema:
            name = prefix+name
            if name in seen or not shape or any(type(n) is not int or n < 1 for n in shape):
                raise ValueError(f"Invalid schema entry {name}")
            seen.add(name)
            size = math.prod(shape)
            yield name, layer, offset, offset+size
            offset += size
        if offset-begin != expected_size:
            raise ValueError("Saved schema and block sizes disagree")
    if offset != meta["n_params"]:
        raise ValueError("Saved schema and n_params disagree")


def select_members(meta, domains, member_ids, include_mixtures):
    selected, labels, seen = [], [], set()
    for member in meta["members"]:
        idx = member["idx"]
        if type(idx) is not int or idx < 0 or idx in seen:
            raise ValueError("Invalid or duplicate member IDs in metadata")
        seen.add(idx)
        if member_ids is not None and idx not in member_ids:
            continue
        if member["kind"] == "anchor":
            pi = np.asarray(member["pi"])
            if pi.shape != (len(meta["domains"]),) or not np.isfinite(pi).all():
                raise ValueError(f"Invalid anchor mixture {idx}")
            d = int(pi.argmax())
            if not np.array_equal(pi, np.eye(len(meta["domains"]))[d]):
                raise ValueError(f"Anchor {idx} is not single-domain")
            domain = meta["domains"][d]
        elif include_mixtures:
            domain = member["mixture_id"]
        else:
            continue
        if domains is not None and domain not in domains:
            continue
        selected.append(idx)
        labels.append(dict(row=len(labels), member_idx=idx, domain=domain,
                           run_id=member["branch"], sampling_seed=10000+idx,
                           kind=member["kind"], mixture_id=member["mixture_id"]))
    if member_ids is not None and set(member_ids) != set(selected):
        raise ValueError("Some requested members were missing or excluded by domain/kind filters; use include-mixtures if needed")
    if domains is not None and set(domains) != {r["domain"] for r in labels}:
        raise ValueError("Some requested domains have no selected checkpoints")
    if len(selected) < 2:
        raise ValueError("Select at least two checkpoints for PCA/t-SNE")
    return selected, labels


def weight_gram(arrays, ranges, chunk_size):
    gram = np.zeros((len(arrays), len(arrays)), dtype=np.float64)
    for _, _, start, stop in ranges:
        for left in range(start, stop, chunk_size):
            x = np.array([w[left:min(left+chunk_size, stop)] for w in arrays], dtype=np.float64)
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite checkpoint weights")
            x -= x[0].copy()
            x -= x.mean(axis=0)
            gram += x@x.T
    return gram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", "--zoo-path", type=Path, required=True,
                        help="One architecture directory with zoo_meta.json and w_<index>.npy")
    parser.add_argument("--layers", type=int, nargs="+", help="Zero-based transformer block indices; default full saved model")
    parser.add_argument("--include-extra", action="store_true", help="Include embeddings/final norm when selecting specific layers")
    parser.add_argument("--per-layer", action="store_true", help="Also project each selected layer separately; its axes are independent")
    parser.add_argument("--tensor-pattern", nargs="+", help="Optional glob patterns for full tensor names")
    parser.add_argument("--domains", nargs="+", help="Domain labels to include; default all pure-domain anchors")
    parser.add_argument("--members", type=int, nargs="+", help="Explicit member IDs")
    parser.add_argument("--include-mixtures", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--output-dir", type=Path, default=Path("weight_embeddings"))
    add_projection_arguments(parser)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("chunk-size must be positive")
    require_projection_dependencies(args.method)
    meta = json.loads((args.checkpoint_path/"zoo_meta.json").read_text())
    if args.layers is not None and (len(set(args.layers)) != len(args.layers) or
                                   any(i < 0 or i >= meta["n_layers"] for i in args.layers)):
        parser.error("layers must be distinct valid zero-based transformer block indices")
    if args.members is not None and len(set(args.members)) != len(args.members):
        parser.error("members must be distinct")
    selected, labels = select_members(meta, args.domains, args.members, args.include_mixtures)
    layout = list(tensor_layout(meta))
    ranges = [entry for entry in layout
              if (args.layers is None or entry[1] in args.layers or (entry[1] is None and args.include_extra))
              and (args.tensor_pattern is None or any(fnmatch.fnmatchcase(entry[0], p) for p in args.tensor_pattern))]
    if not ranges:
        parser.error("No saved tensors match the selection")
    arrays = []
    for idx in selected:
        array = np.load(args.checkpoint_path/f"w_{idx}.npy", mmap_mode="r", allow_pickle=False)
        if array.shape != (meta["n_params"],) or array.dtype != np.float32:
            raise ValueError(f"w_{idx}.npy does not match the saved float32 layout")
        arrays.append(array)
    output_root = args.output_dir
    selections = [("combined", ranges)]
    if args.per_layer:
        layers = list(dict.fromkeys(entry[1] for entry in ranges))
        selections += [("extra" if layer is None else f"layer_{layer}",
                        [entry for entry in ranges if entry[1] == layer]) for layer in layers]
    for name, selected_ranges in selections:
        args.output_dir = output_root if name == "combined" else output_root/name
        args.output_dir.mkdir(parents=True, exist_ok=True)
        gram = weight_gram(arrays, selected_ranges, args.chunk_size)
        write_json(args.output_dir/"weights.json", dict(
            checkpoint_path=str(args.checkpoint_path.resolve()), arch=meta["arch"],
            beta=meta.get("beta"), trunk_seed=meta.get("trunk_seed"),
            source=meta.get("source", "pretrained_zoo"), member_ids=selected,
            selected_tensors=[e[0] for e in selected_ranges],
            selected_parameter_count=sum(e[3]-e[2] for e in selected_ranges),
            note="Each point is one checkpoint. All coordinates selected for a point are concatenated conceptually. Centering cancels a shared trunk; base sampling seed is not a replay of resumed training."))
        project(gram, labels, args, f"Weight space: {meta['arch']} / {name}")
        print(f"Saved {len(labels)} checkpoint embeddings to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

"""
Whole-stack pipeline: per-family Gram PCA -> codes.

One PCA sample is one complete decoder stack (DeepWeightFlow framing), so:

  Stage 1  build the per-architecture ensembles (w_0 plus noise-augmented members)
  Stage 2  fit ONE DualGramPCA per architecture at k = N-1 (the rank bound)
  Stage 3  assemble the code matrix at the requested rank k

Stage 2 fits at the maximum rank ONCE. Lower-rank arms are a column prefix of the
same fit -- components are variance-ordered and orthogonal, so a k-prefix IS the
rank-k PCA. That makes the whole k sweep free after one fit.

    python scripts/train_stack.py --k 99 --n_samples 100 --noise_scale 1e-2
    python scripts/train_stack.py --k 50            # reuses the Stage 2 fits
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from llmzoo.artifacts.io import (
    ensemble_fingerprint, pca_fingerprint, provenance_block, read_json,
)
from llmzoo.data.ensemble import ENSEMBLE_SOURCES, EnsembleDataset
from llmzoo.pca.gram import DualGramPCA, spectrum_stats
from llmzoo.models.registry import N_FAMILIES
from llmzoo.artifacts.bundle import CodeStats, rebuild_manifest
import llmzoo.wandb_utils as wb


def ts() -> str:
    return time.strftime("[%H:%M:%S]")


# ---------------------------------------------------------------------------
# Member selection
# ---------------------------------------------------------------------------

def resolve_members(args) -> Optional[List[int]]:
    """
    --members -> an explicit sorted member list, or None for "the whole zoo".

    'train' reads the split that train_zoo.py already sealed rather than taking a
    list on the command line. The holdout is a property OF THE ZOO -- it was chosen
    when the plan was built (mixtures.holdout_split) and written into
    zoo_meta.json -- so retyping it into an sbatch is one transcription away from a
    basis that quietly contains the members it is supposed to exclude.
    """
    spec = str(getattr(args, "members", "all")).strip()
    if spec in ("", "all"):
        return None
    if args.source != "zoo":
        raise SystemExit(f"--members {spec!r} needs --source zoo.")
    if args.zoo_dir is None:
        raise SystemExit(f"--members {spec!r} needs --zoo_dir.")

    if spec == "train":
        if len(args.arch_list) != 1:
            raise SystemExit(
                "--members train reads one zoo_meta.json, so it needs exactly one "
                f"--arch_list entry; got {args.arch_list}.")
        zpath = os.path.join(args.zoo_dir, args.arch_list[0], "zoo_meta.json")
        z = read_json(zpath)
        if z is None:
            raise SystemExit(f"--members train: no zoo_meta.json at {zpath}.")
        holdout = z.get("holdout") or {}
        idxs = holdout.get("train")
        if not idxs:
            raise SystemExit(
                f"{zpath} has no holdout['train']. It predates the sealed split; "
                f"re-run scripts/train_zoo.py --mode plan for this zoo.")
        held = sorted(set(holdout.get("interior", []))
                      | set(holdout.get("vertex", [])))
        print(f"  holdout   : interior={holdout.get('interior')} "
              f"vertex={holdout.get('vertex')} -> {len(held)} members excluded")
        return sorted(int(i) for i in idxs)

    try:
        return sorted({int(t) for t in spec.replace(",", " ").split()})
    except ValueError:
        raise SystemExit(
            f"--members {spec!r}: expected 'all', 'train', or a comma-separated "
            f"list of integers.")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def stage_pca(args, ds: EnsembleDataset, pca_root: str) -> Dict[str, DualGramPCA]:
    """Fit (or load) one DualGramPCA per architecture at the maximum rank."""
    out: Dict[str, DualGramPCA] = {}
    k_max = ds.n_samples - 1
    for arch in ds.arch_list:
        d = os.path.join(pca_root, arch)
        meta = os.path.join(d, "gram_pca_meta.json")
        if os.path.exists(meta) and not args.force_pca:
            print(f"{ts()} PCA for {arch} exists — loading …")
            pca = DualGramPCA.load(d)
            if pca.n_samples_ != ds.n_samples:
                raise RuntimeError(
                    f"{arch}: cached PCA was fit on N={pca.n_samples_} but the "
                    f"ensemble now has N={ds.n_samples}. Re-fit with --force_pca.")
            # D changes whenever the stack layout changes -- most sharply when the
            # extra segment is switched on or off. Without this check a stale
            # decoder-only basis is silently reused against a wider ensemble and
            # every code means something different.
            if pca.n_params_ != ds.stacks[arch].n_params:
                raise RuntimeError(
                    f"{arch}: cached PCA was fit on D={pca.n_params_:,} but the "
                    f"ensemble now has D={ds.stacks[arch].n_params:,} "
                    f"(include_extra={ds.include_extra}). Re-fit with --force_pca.")
        else:
            print(f"\n{ts()} Fitting PCA for {arch} at k={k_max} …")
            pca = DualGramPCA(n_components=k_max).fit(ds, arch)
            pca.save(d)
        out[arch] = pca
    return out


def stage_codes(ds: EnsembleDataset, pcas: Dict[str, DualGramPCA], k: int,
                codes_dir: str, provenance: dict):
    """
    Assemble the code matrix at rank k and SEAL it as an artifact.

    Returns (codes (M,k) tensor, family_idxs (M,) tensor, arch_of_row, CodeStats).

    Writing codes_k<k>/ is what lets train_flow.py train on 40 KB without ever
    constructing an EnsembleDataset or touching DualGramPCA. That is the property
    that makes the flow stack re-runnable on a different ensemble source (Pythia
    checkpoint revisions): swap the source, re-run stages 1-3, and the flow trainer
    is unchanged and unaware.
    """
    avail = min(p.n_components for p in pcas.values())
    if k > avail:
        raise ValueError(f"--k {k} exceeds the smallest fitted rank ({avail}). "
                         f"Re-fit with a larger ensemble, or lower --k.")

    blocks, fidxs, arch_of_row = [], [], []
    seen: Dict[int, str] = {}
    arch_to_family: Dict[str, int] = {}
    for arch in ds.arch_list:
        c = torch.from_numpy(pcas[arch].codes(k).copy()).float()
        fi = int(ds.stacks[arch].family_idx)
        # Two archs sharing a family_idx would silently overwrite each other's
        # statistics. Unreachable with today's registry, but the failure is silent,
        # so it is checked rather than assumed.
        if fi in seen:
            raise ValueError(
                f"{arch} and {seen[fi]} both claim family_idx={fi}. Per-family code "
                f"statistics are keyed by family_idx, so one would clobber the "
                f"other. Renumber ARCH_CONFIGS contiguously.")
        seen[fi] = arch
        arch_to_family[arch] = fi
        blocks.append(c)
        fidxs.append(torch.full((c.shape[0],), fi, dtype=torch.long))
        arch_of_row.extend([arch] * c.shape[0])

    codes = torch.cat(blocks)
    fidx = torch.cat(fidxs)

    # CodeStats.from_codes is the single implementation of the reduction: per-dim
    # mean, and one scalar scale per family (RMS over dimensions of the per-dim
    # stds). Each family has its OWN basis, so pooling across families would compare
    # coefficients on unrelated directions.
    cs = CodeStats.from_codes(codes.numpy(), fidx.numpy(), arch_to_family,
                              N_FAMILIES, k=k)
    for arch, fi in arch_to_family.items():
        print(f"  {arch:16s} codes ({int((fidx == fi).sum())}, {k})  "
              f"family_idx={fi}  scale={float(cs.stds[fi, 0]):.4g}")
    cs.save(codes_dir, provenance=provenance)
    return codes, fidx, arch_of_row, cs


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch_list", nargs="+",
                   default=["gpt2_medium", "smollm2_360m", "pythia_410m"])
    p.add_argument("--mode", choices=["tiny", "full"], default="full")
    p.add_argument("--run_name", default="perfam")
    p.add_argument("--artifact_dir",
                   default=os.environ.get("ARTIFACT_DIR", "./artifacts"),
                   help="Run artifacts root. slurm/env.sh exports "
                        "ARTIFACT_DIR; this flag overrides it.")

    # Ensemble
    p.add_argument("--n_samples", type=int, default=100,
                   help="N, ensemble size per architecture (sample 0 = real model)")
    p.add_argument("--noise_scale", type=float, default=1e-2,
                   help="Augmentation std as a fraction of the stack's weight std. "
                        "Calibrate with calibrate_noise.py — too small and the Gram "
                        "matrix is roundoff, too large and the members are broken.")
    p.add_argument("--include_extra", action="store_true", default=True,
                   help="Include embeddings, final norm and untied LM head in the "
                        "stack. On by default: DeepWeightFlow flattens whole "
                        "networks, and gpt2_medium's embedding alone is ~17%% of "
                        "its parameters.")
    p.add_argument("--no_include_extra", dest="include_extra",
                   action="store_false",
                   help="Decoder blocks only — the layout_version 1 behaviour. "
                        "Needs a fresh --run_name; the ensemble cache gate refuses "
                        "to mix the two.")
    p.add_argument("--exclude_1d", action="store_true", default=True)
    p.add_argument("--no_exclude_1d", dest="exclude_1d", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk_budget_mb", type=int, default=512)

    # Where the members come from.
    #
    # A zoo run needs THREE flags together, and each is separately enforced:
    #   --source zoo --zoo_dir <parent>  --noise_scale 0.0  --no_exclude_1d
    # `--noise_scale 0.0` because EnsembleDataset refuses a zoo with augmentation
    # noise, and `--no_exclude_1d` because train_zoo writes exclude_1d=False
    # (GPT-2 has biases everywhere and a generated model must not inherit
    # member 0's) while this script defaults to True.
    p.add_argument("--source", default="noise", choices=list(ENSEMBLE_SOURCES),
                   help="'noise' = the manufactured ensemble (the beta->0 "
                        "reference and the machinery test); 'zoo' = real "
                        "branched models from scripts/train_zoo.py.")
    p.add_argument("--zoo_dir", default=None,
                   help="Required with --source zoo. This is the PARENT of the "
                        "arch directory: members are read from "
                        "<zoo_dir>/<arch>/w_<i>.npy. train_zoo.py's own "
                        "--zoo_dir is the arch-level dir, one level down.")

    p.add_argument("--members", default="all",
                   help="Which zoo members the basis is fit on. 'all' (default) "
                        "uses 0..n_samples-1. 'train' reads "
                        "<zoo_dir>/<arch>/zoo_meta.json['holdout']['train'], so "
                        "the sealed interior + vertex holdout is genuinely "
                        "OUTSIDE the span rather than merely outside the flow's "
                        "training rows -- which is what 'held out' has to mean "
                        "before a generative number can be quoted against a "
                        "retrieval baseline. Or pass a "
                        "comma-separated index list. N and k follow the subset, "
                        "so --members train on a 100-member zoo gives N=92 and a "
                        "rank bound of 91.")

    # Rank
    p.add_argument("--k", type=int, default=None,
                   help="Code dimension. Default N-1 (the rank bound).")

    p.add_argument("--force_extract", action="store_true")
    p.add_argument("--force_pca", action="store_true")
    p.add_argument("--no_wandb", action="store_true")
    wb.add_wandb_args(p)
    args = p.parse_args()

    run_root = os.path.join(args.artifact_dir, "runs", args.run_name)
    pca_root = os.path.join(run_root, "pca")
    os.makedirs(run_root, exist_ok=True)

    member_idxs = resolve_members(args)
    if member_idxs is not None:
        # N follows the subset. Resolved BEFORE k's default so --members train
        # gives k = 91 rather than 99 -- the rank bound of the ensemble that is
        # actually fit, not of the zoo it was drawn from.
        args.n_samples = len(member_idxs)

    k = args.k if args.k is not None else args.n_samples - 1

    # name_suffix matters: slurm/stack_run.sbatch calls this script once per rank in a
    # single job, and each call is its own process, so without it both runs land
    # under the identical name train_stack_<jobid>.
    # --wandb_group matters here specifically. train_zoo.py collects a zoo's trunk,
    # branches and gate under `zoo_<arch>_<tag>_n<N>`, and the spectrum stage must
    # join them; with no group a run falls back to `job_<jobid>`.
    wb.init_run(job_type="train_stack", config=vars(args),
                tags=[args.mode, f"k{k}", args.run_name] + args.arch_list,
                enabled=not args.no_wandb, artifact_dir=args.artifact_dir,
                name_suffix=f"{args.run_name}_k{k}",
                project=args.wandb_project, group=args.wandb_group)

    print("=" * 68)
    print("Whole-stack pipeline (per-family Gram PCA)")
    if member_idxs is not None:
        print(f"  members   : {args.members} -> N={len(member_idxs)} "
              f"{member_idxs[:6]}{' …' if len(member_idxs) > 6 else ''}")
    print(f"  run       : {args.run_name}   -> {run_root}")
    print(f"  archs     : {args.arch_list}")
    print(f"  N         : {args.n_samples}   noise_scale={args.noise_scale}")
    print(f"  source    : {args.source}"
          + (f"   zoo_dir={args.zoo_dir}" if args.source != "noise" else ""))
    print(f"  k         : {k}  (rank bound is N-1 = {args.n_samples - 1})")
    print(f"  extra seg : include_extra={args.include_extra} "
          f"(embeddings / final norm / untied LM head)")
    print("=" * 68)

    print(f"\n{ts()} Stage 1: ensembles")
    ds = EnsembleDataset(arch_list=args.arch_list, n_samples=args.n_samples,
                         noise_scale=args.noise_scale, exclude_1d=args.exclude_1d,
                         include_extra=args.include_extra,
                         mode=args.mode, artifact_dir=run_root, seed=args.seed,
                         chunk_budget_bytes=args.chunk_budget_mb * 1024 * 1024,
                         force_extract=args.force_extract,
                         source=args.source, zoo_dir=args.zoo_dir,
                         member_idxs=member_idxs)
    print(ds.summary())

    print(f"\n{ts()} Stage 2: per-family Gram PCA")
    pcas = stage_pca(args, ds, pca_root)

    # Fingerprints bind every downstream artifact to the ensemble and bases it came
    # from, so a stale codes/flow directory is DETECTED rather than reused.
    ens_meta = read_json(os.path.join(run_root, "ensemble", "ensemble_meta.json"))
    ens_fp = ensemble_fingerprint(ens_meta)
    pca_fps = {
        a: pca_fingerprint(
            read_json(os.path.join(pca_root, a, "gram_pca_meta.json")) or {})
        for a in args.arch_list
    }
    print(f"  ensemble fingerprint : {ens_fp}")

    print(f"\n{ts()} Stage 3: codes at k={k}")
    codes_dir = os.path.join(run_root, f"codes_k{k}")
    codes, fidxs, arch_of_row, code_stats = stage_codes(
        ds, pcas, k, codes_dir,
        provenance_block(args.run_name, k, ens_fp, pca_fps))
    print(f"  code matrix: {tuple(codes.shape)} across {len(set(arch_of_row))} families")

    summary = {
        "run_name": args.run_name, "k": k, "n_samples": args.n_samples,
        "noise_scale": args.noise_scale, "arch_list": args.arch_list,
        "exclude_1d": args.exclude_1d,
        "include_extra": args.include_extra,
        "per_arch": {a: {"n_params": ds.stacks[a].n_params,
                          "n_layers": ds.stacks[a].n_layers,
                          "block_size": ds.stacks[a].block_size,
                          "extra_size": ds.stacks[a].extra_size,
                          "fitted_k": pcas[a].n_components,
                          "variance_captured_at_k": float(np.sum(
                              pcas[a].explained_variance_ratio_[:k])),
                          **{f"spectrum_{key}": val for key, val in
                             spectrum_stats(pcas[a].gram_evals_,
                                            min(k, pcas[a].n_components)).items()}}
                      for a in ds.arch_list},
    }
    summary["ensemble_fingerprint"] = ens_fp
    summary["pca_fingerprints"] = pca_fps
    summary["code_stats_fingerprint"] = code_stats.fingerprint()
    with open(os.path.join(run_root, f"pipeline_summary_k{k}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Derived index over the per-directory metas. Rebuilt by scanning rather than
    # incrementally maintained, so the two train_stack invocations in one Slurm job
    # cannot race each other into a lost section.
    rebuild_manifest(run_root)

    print(f"\n{ts()} Done. Artifacts under {run_root}")
    for a, d in summary["per_arch"].items():
        print(f"  {a:16s} D={d['n_params']:>13,}  fitted_k={d['fitted_k']:>4d}  "
              f"var@k={d['variance_captured_at_k']:.4%}  "
              f"ev0/med={d['spectrum_ev0_over_median']:.4g}  "
              f"effrank={d['spectrum_effective_rank_ratio']:.3f}")

    # Spectrum flatness is the honest caveat, made visible at the end of every run
    # rather than left in a JSON nobody opens. A flow
    # over codes whose aggregate distribution is already Gaussian learns the prior
    # and adds nothing. Reported on the BULK statistic -- ev0/ev[k-1] reads ~12 on a
    # pure-noise ensemble at k=N-1 and would hide exactly this.
    ratios = {a: d["spectrum_ev0_over_median"]
              for a, d in summary["per_arch"].items()}
    effs = {a: d["spectrum_effective_rank_ratio"]
            for a, d in summary["per_arch"].items()}
    if ratios and min(ratios.values()) < 2.0:
        print(f"\n  NOTE the retained spectrum is FLAT "
              f"(min ev0/median = {min(ratios.values()):.3g}, "
              f"max effective-rank ratio = {max(effs.values()):.3f} of 1.0). For a "
              f"manufactured noise ensemble that is the expected result, not a "
              f"problem — but it means the per-family code distribution is "
              f"near-Gaussian by construction, so any flow fit to it may learn only "
              f"the prior. Compare flow_codes against gauss_codes, not against "
              f"pca_only.")

    wb.summary({"pipeline/ensemble_fingerprint": ens_fp,
                "pipeline/code_stats_fingerprint": code_stats.fingerprint(),
                "pipeline/min_spectrum_ev0_over_median":
                    min(ratios.values()) if ratios else None,
                "pipeline/max_effective_rank_ratio":
                    max(effs.values()) if effs else None,
                "pipeline/k": k,
                "pipeline/include_extra": args.include_extra})
    wb.finish()


if __name__ == "__main__":
    main()

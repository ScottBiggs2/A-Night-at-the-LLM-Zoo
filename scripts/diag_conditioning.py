"""
Dump everything needed to judge whether a flow is CONDITIONED, per PCA component.

Why this exists
---------------
Every conditioning statistic the repo had was a single scalar pooled over all k
components: `_rms_ratio` in train_flow.py applies `.pow(2).mean().sqrt()` across the
whole (n, k) tensor. Normalization divides by ONE scalar, so PCA's variance ordering
survives, and c0-c3 carry ~80% of the trace. A pooled statistic is therefore
implicitly eigenvalue-weighted and near-blind to everything else -- and it cannot be
read off, because the weights never appear.

Worse, the pooled numbers mix two subspaces. Measured on zoo_b015_mini_k99, the
within-mixture variance of the five anchor groups lives in nearly DISJOINT blocks of
components:

    anchor_web           99.1% of its within-variance in c25+
    anchor_code          98.5% in c10-c24
    anchor_math          99.2% in c10-c24
    anchor_multilingual  47.5% in c5-c9, 51.9% in c10-c24
    anchor_books         40.1% in c10-c24, 59.2% in c25+

and only 0.4% of ALL within-variance is in c0-c4 -- the components carrying 84% of the
total. So `at_fixed_pi` divides a numerator that is mostly leading-block by a
denominator that has essentially no leading-block content, and the answer depends on
which group happened to be selected. That is why the published rho moved 1.51x when
pooled over five groups instead of one, and 15x at k=25.

This script writes the SUFFICIENT STATISTICS -- per-component mean and variance of the
real codes and of each flow's samples at each requested mixture -- and lets
report_conditioning.py own every formula. The split is deliberate: this half needs the
artifacts, that half needs the argument, and only that half has to stay reproducible
off-cluster forever.

Two paths, and the first needs no torch at all:

    # data only: the zoo's own conditional structure, no flow involved. Runs locally.
    python scripts/diag_conditioning.py --run_name zoo_b015_mini_k99 --k 99 50

    # plus sealed flows, each landing in the payload for its own rank
    python scripts/diag_conditioning.py --run_name zoo_b015_mini_k99 --k 99 50 \
        --flow_dirs diag_cond/rank_k99 diag_cond/steps_e8000

`--k` is VARIADIC here, matching eval_stack.py rather than train_flow.py -- a payload
is per-rank, and a flow is assigned to the payload matching its own `dim`, since a
codes-space flow's width IS k and one flow cannot span ranks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from llmzoo.artifacts.bundle import CodeStats, load_pi_matrix  # noqa: E402
from llmzoo.artifacts.io import read_json  # noqa: E402

LAYOUT_VERSION = 1

# Rounded on the way out. These are float32 reductions; the last bits are not
# reproducible across BLAS versions, and a committed payload that churns on a library
# upgrade is worse than one that is 7 digits wide.
ROUND = 9


def _r(a) -> list:
    return np.round(np.asarray(a, dtype=np.float64), ROUND).tolist()


def normalize(codes: np.ndarray, family_idxs: np.ndarray,
              cs: CodeStats) -> np.ndarray:
    """
    Mirror of CodeStats.normalize (bundle.py:233) in numpy rather than torch.

    Duplicated on purpose: the whole point of the data path is that it runs with no
    torch, so it can be driven from a laptop against copied-down files and on the
    `cpu` partition. The per-family mean is per-DIMENSION and the scale is ONE
    SCALAR, which is what keeps the variance ordering intact -- see bundle.py:83-87
    for why per-dimension whitening was rejected.
    """
    k = cs.k
    return ((codes[:, :k] - cs.means[family_idxs][:, :k])
            / cs.stds[family_idxs]).astype(np.float64)


def dirichlet_probe(n: int, n_domains: int, seed: int) -> np.ndarray:
    """
    A deterministic probe set for the round-trip curve.

    Drawn from Dirichlet(1) to match how the zoo's singletons were drawn
    (mixtures.sample_simplex), NOT from the empirical member distribution: this set
    exists to measure the conditioning map at mixtures nothing was trained on, so it
    must not be concentrated where the training members already are.
    """
    return np.random.default_rng(seed).dirichlet([1.0] * n_domains, size=n)


def build_targets(members: List[dict], holdout: dict, pi: np.ndarray,
                  n_dirichlet: int, seed: int,
                  all_members: Optional[List[dict]] = None) -> List[dict]:
    """
    The mixtures to sample each flow at.

    Three kinds, and they answer different questions:

      anchor    -- the only mixtures with REPLICATES, so the only ones where a
                   within-mixture reference exists at all. `member_rows` is carried
                   so the report can form the empirical group mean, which is the
                   unbiased reference the regression surface is not.
      holdout   -- the sealed cells. `interior` is interpolation, `vertex` is
                   extrapolation, and the design provisions them as separate
                   difficulties on purpose.

                   On a MEMBER-SUBSET run these members are not basis rows at all, so
                   `all_members` supplies their pi from the zoo metadata. Without it
                   the sealed cells silently vanish from the payload -- which is the
                   one evaluation a subset basis exists to make possible, since it is
                   the only place the holdout is genuinely held out. `member_rows` is
                   then empty and the report scores the round-trip against the
                   REQUESTED pi, which is the honest comparison: there is no in-basis
                   reference member to compare against, by construction.
      dirichlet -- novel mixtures, for the round-trip curve. Cheap, so there are
                   many: no decode and no perplexity, which is the whole reason a
                   code-space control can run at hundreds of pi where the functional
                   one runs at five.
    """
    by_idx = {int(m["idx"]): m for m in members}
    by_idx_all = {int(m["idx"]): m for m in (all_members or members)}
    groups: Dict[str, List[int]] = {}
    for row, m in enumerate(members):
        groups.setdefault(str(m["mixture_id"]), []).append(row)

    out: List[dict] = []
    for mid, rows in sorted(groups.items()):
        if len(rows) > 1:
            out.append({"label": mid, "kind": "anchor", "pi": _r(pi[rows[0]]),
                        "member_rows": rows})

    seen = {t["label"] for t in out}
    for nm in ("interior", "vertex"):
        for idx in holdout.get(nm, []):
            m = by_idx.get(int(idx))
            if m is None:
                # Not a basis row -- a genuinely sealed member on a subset run. Emit
                # it from the zoo metadata so the flow still gets sampled there.
                m = by_idx_all.get(int(idx))
                if m is None:
                    continue
                mid = str(m["mixture_id"])
                if mid in seen:
                    continue
                seen.add(mid)
                out.append({"label": mid, "kind": "sealed_" + nm,
                            "pi": _r(m["pi"]), "member_rows": [], "holdout": nm,
                            "in_basis": False})
                continue
            mid = str(m["mixture_id"])
            rows = groups.get(mid, [])
            if mid in seen:
                # The held-out vertex IS an anchor group; do not duplicate it, just
                # record that it is sealed.
                for t in out:
                    if t["label"] == mid:
                        t["holdout"] = nm
                continue
            seen.add(mid)
            out.append({"label": mid, "kind": "holdout_" + nm,
                        "pi": _r(pi[rows[0]] if rows else m["pi"]),
                        "member_rows": rows, "holdout": nm})

    for i, p in enumerate(dirichlet_probe(n_dirichlet, pi.shape[1], seed)):
        out.append({"label": f"probe_{i:03d}", "kind": "probe", "pi": _r(p),
                    "member_rows": []})
    return out


def data_block(run_root: str, k: int, zoo_dir: Optional[str]) -> dict:
    """The zoo's own conditional structure. No flow, no torch, no GPU."""
    codes_dir = os.path.join(run_root, f"codes_k{k}")
    cs = CodeStats.load(codes_dir, expect_k=k)
    if cs.codes is None or cs.family_idxs is None:
        raise SystemExit(f"{codes_dir} has no codes.npy/family_idxs.npy.")
    x = normalize(cs.codes, cs.family_idxs, cs)

    ens = read_json(os.path.join(run_root, "ensemble", "ensemble_meta.json"))
    if ens is None:
        raise SystemExit(f"No ensemble_meta.json under {run_root}/ensemble/.")
    if zoo_dir:
        # Offline override, so a payload can be rebuilt from copied-down files with
        # no cluster. The join itself still goes through load_pi_matrix, because a
        # reimplemented row->member->pi walk is exactly the defect
        # pi_code_distance_corr exists to catch (bundle.py:255-261).
        ens = dict(ens, zoo_dir=zoo_dir)
    pim = load_pi_matrix(ens)
    pi = pim["pi"].astype(np.float64)
    if pi.shape[0] != x.shape[0]:
        raise SystemExit(
            f"pi has {pi.shape[0]} rows and codes have {x.shape[0]}. The "
            f"row->member join is broken; refusing to emit a scrambled pairing.")

    arch = ens["arch_list"][0]
    zmeta = read_json(os.path.join(pim["zoo_dir"], arch, "zoo_meta.json")) or {}
    members = list(zmeta.get("members", []))
    sel = ens.get("member_idxs")
    if sel is not None:
        by = {int(m["idx"]): m for m in members}
        members = [by[int(i)] for i in sel]
    if len(members) != x.shape[0]:
        raise SystemExit(f"zoo_meta has {len(members)} members for {x.shape[0]} rows.")

    ev_p = os.path.join(run_root, "pca", arch, "gram_evals.npy")
    ev = np.load(ev_p).astype(np.float64) if os.path.exists(ev_p) else None
    summ = read_json(os.path.join(run_root, f"pipeline_summary_k{k}.json")) or {}

    return {
        "layout_version": LAYOUT_VERSION,
        "all_members": [{"idx": int(m["idx"]), "pi": _r(m["pi"]),
                         "kind": m.get("kind"),
                         "mixture_id": m.get("mixture_id")}
                        for m in list(zmeta.get("members", []))],
        "k": int(k),
        "arch": arch,
        "n_members": int(x.shape[0]),
        "domains": list(pim["domains"]),
        "code_scale": float(cs.stds[int(cs.family_idxs[0]), 0]),
        "eigenvalues": _r(ev[:k]) if ev is not None else None,
        "fingerprints": {
            "pi": pim["fingerprint"],
            "code_stats": cs.fingerprint(),
            "ensemble": summ.get("ensemble_fingerprint"),
            "pca": summ.get("pca_fingerprints"),
        },
        "spectrum": (summ.get("per_arch", {}) or {}).get(arch),
        "members": [{"row": i, "idx": int(m["idx"]), "pi": _r(pi[i]),
                     "kind": m.get("kind"), "mixture_id": m.get("mixture_id"),
                     "branch": m.get("branch")} for i, m in enumerate(members)],
        "holdout": zmeta.get("holdout", {}),
        "codes": [_r(row) for row in x],
        "cells": [],
    }


def sample_cells(run_root: str, payloads: Dict[int, dict], flow_dirs: List[str],
                 n_samples: int, n_steps: int, n_raw: int, seed: int) -> None:
    """
    Sample each sealed flow at every target and append its sufficient statistics.

    Samples are taken in FLOW SPACE, which for a codes-space flow is exactly the
    normalized code space `train_flow.py:_rms_ratio` measures in. Going through
    `sample_codes` instead would denormalize and force the report to undo it, and any
    mismatch there would look like a conditioning result.
    """
    import torch  # noqa: PLC0415  -- lazy, so the data path stays torch-free
    from llmzoo.gen.flow import load_flow  # noqa: PLC0415

    dev = torch.device("cpu")
    for rel in flow_dirs:
        d = rel if os.path.isabs(rel) else os.path.join(run_root, rel)
        meta = read_json(os.path.join(d, "flow_meta.json"))
        if meta is None:
            print(f"  SKIP {rel} — no flow_meta.json")
            continue
        dim = int(meta["dim"])
        if dim not in payloads:
            raise SystemExit(
                f"{rel} is a dim={dim} flow but --k does not include {dim}. A "
                f"codes-space flow's width IS k, so it belongs to that rank's "
                f"payload; add --k {dim}.")
        if meta.get("space") != "codes":
            print(f"  SKIP {rel} — space={meta.get('space')!r}; only 'codes' "
                  f"samples directly in normalized code space.")
            continue

        cs = CodeStats.load(os.path.join(run_root, f"codes_k{dim}"), expect_k=dim)
        fm = load_flow(d, code_stats=cs, device=str(dev))
        mode = fm.cond_mode
        fam = int(cs.family_idxs[0]) if cs.family_idxs is not None else 0
        pay = payloads[dim]
        cell = {"name": os.path.basename(d.rstrip("/")), "dir": rel, "dim": dim,
                "cond_mode": mode, "n_steps": int(n_steps),
                "n_samples": int(n_samples),
                "net": meta.get("net"), "train": meta.get("train"),
                "cond": meta.get("cond"), "targets": []}

        for t in pay["targets"]:
            g = torch.Generator(device="cpu").manual_seed(seed)
            fv = torch.full((n_samples,), fam, dtype=torch.long, device=dev)
            kw = {}
            if mode == "pi":
                pv = torch.tensor(t["pi"], dtype=torch.float32,
                                  device=dev).reshape(1, -1).expand(n_samples, -1)
                kw["pi"] = pv.contiguous()
            with torch.no_grad():
                x0 = torch.randn(n_samples, dim, generator=g).to(dev) * fm.flow.source_std
                s = fm.flow.sample(n_samples, fv, n_steps=n_steps, x0=x0, **kw)
            s = s.double().numpy()
            cell["targets"].append({
                "label": t["label"], "kind": t["kind"], "n": int(n_samples),
                "mean": _r(s.mean(axis=0)),
                "var": _r(s.var(axis=0, ddof=1)),
                # A bounded raw subsample, for the nearest-neighbour / data-copying
                # screen. Copying lives in the DIRECTION of a residual, which needs
                # individual samples; means and variances cannot show it.
                "samples": [_r(r) for r in s[:n_raw]],
            })
            if mode != "pi":
                # A family-conditioned flow ignores pi entirely, so every target
                # would be the same draw. Record one and say so.
                cell["targets"][-1]["label"] = "marginal"
                break
        pay["cells"].append(cell)
        print(f"  {cell['name']:22s} dim={dim:3d} cond={mode:6s} "
              f"targets={len(cell['targets'])}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact_dir",
                   default=os.environ.get("ARTIFACT_DIR", "./artifacts"))
    p.add_argument("--run_name", required=True)
    p.add_argument("--k", nargs="+", type=int, required=True,
                   help="VARIADIC, unlike train_flow.py --k. One payload per rank.")
    p.add_argument("--zoo_dir", default=None,
                   help="Override ensemble_meta's zoo_dir, for offline rebuilds.")
    p.add_argument("--flow_dirs", nargs="*", default=[],
                   help="Sealed flow directories, absolute or relative to the run "
                        "root. Taken explicitly rather than discovered: load_run "
                        "only globs flow_k<k>_<space>, so a pi flow in "
                        "flow_k<k>_<space>_pi/ is invisible to it.")
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument("--n_raw", type=int, default=16,
                   help="Raw sample rows kept per target. Needed for the "
                        "per-sample round-trip and the data-copying screen. This "
                        "dominates the payload: n_raw*k floats per target per cell, "
                        "so 32 over 73 targets and 12 cells was 13 MB at k=50. "
                        "Only the data-only payload (no --flow_dirs) is small "
                        "enough to commit; cell payloads regenerate deterministically "
                        "from the sealed flows and --seed.")
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--n_dirichlet", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="Output DIRECTORY; one cond_<run>_k<k>.json per rank. "
                        "Defaults to <run_root>/conditioning/.")
    args = p.parse_args()

    run_root = os.path.join(args.artifact_dir, "runs", args.run_name)
    if not os.path.isdir(run_root):
        raise SystemExit(f"No run at {run_root}.")

    payloads: Dict[int, dict] = {}
    print("=" * 78)
    print(f"Conditioning payload — run={args.run_name} ranks={sorted(set(args.k))}")
    print("=" * 78)
    for k in sorted(set(args.k)):
        blk = data_block(run_root, k, args.zoo_dir)
        blk["run_name"] = args.run_name
        blk["targets"] = build_targets(blk["members"], blk["holdout"],
                                       np.array([m["pi"] for m in blk["members"]]),
                                       args.n_dirichlet, args.seed,
                                       all_members=blk.get("all_members"))
        payloads[k] = blk
        n_anch = sum(1 for t in blk["targets"] if t["kind"] == "anchor")
        print(f"  k={k:3d}  members={blk['n_members']}  targets={len(blk['targets'])}"
              f" ({n_anch} anchor)  pi_fp={blk['fingerprints']['pi'][:12]}")

    if args.flow_dirs:
        print("\nSampling sealed flows (CPU):")
        sample_cells(run_root, payloads, args.flow_dirs, args.n_samples,
                     args.n_steps, args.n_raw, args.seed)

    out_dir = args.out or os.path.join(run_root, "conditioning")
    os.makedirs(out_dir, exist_ok=True)
    for k, blk in sorted(payloads.items()):
        dest = os.path.join(out_dir, f"cond_{args.run_name}_k{k}.json")
        tmp = dest + ".tmp"
        with open(tmp, "w") as f:
            json.dump(blk, f, separators=(",", ":"))
        os.replace(tmp, dest)
        print(f"-> {dest}  ({os.path.getsize(dest) / 1024:.0f} KB, "
              f"{len(blk['cells'])} cells)")
    print("\nScore it with:  python scripts/report_conditioning.py --data_dir "
          f"{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

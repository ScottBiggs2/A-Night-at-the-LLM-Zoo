#!/usr/bin/env python3
"""
Weight-space geometry of a zoo: is it a real manifold or a ball around the trunk?

    python scripts/diag_zoo_geometry.py --zoo_dir $ARTIFACT_DIR/zoo_b030/gpt2_zoo_mini

The domain-separation gate answers "is mixture identity DETECTABLE" and nothing else. It is
maximised by a tight within-anchor spread -- which is precisely the regime where
the ensemble mean is already a good model, `gauss_codes` becomes a strong null, and
a flow can memorise the code distribution -- the collapse-toward-the-mean
mechanism -- and no number in domain_separation.json sees it.

So measure the geometry instead. The quantities that matter:

  disp        ||w_i - trunk|| / ||trunk||     how far members actually moved.
              If this is ~1e-4 the zoo is the trunk with rounding on top, and
              "we generated a working model" is unfalsifiable.

  spread/disp mean ||w_i - w_j|| / mean ||w_i - trunk||
              THE diagnostic. Members that all drifted the same direction give
              ~0; members that moved independently give ~sqrt(2)=1.41. Low means
              the ensemble has one direction and the flow has nothing to learn.

  mean_frac   ||w_i - mean|| / mean ||w_i - w_j||
              sqrt((N-1)/N / 2) for an i.i.d. spread -- 0.677 at N=12, ->0.707
              as N grows. Much lower means every member sits near the centroid,
              so decoding the mean scores like a member and dPPL cannot police
              the generative arm.

  between/within   inter-group distance / intra-group distance, over the groups
              that HAVE a within (i.e. the anchors). The weight-space analogue
              of the gate's SNR, on the quantity a PCA basis is actually built
              from.

--------------------------------------------------------------------------------
HOW THIS SCALES, AND WHY IT HAD TO BE REWRITTEN FOR N=100
--------------------------------------------------------------------------------
The first version materialised `blk[i] - blk[j]` for every pair in every chunk.
At N=12 that is 66 pairs and it ran in 35 s. At N=100 it is 4950 pairs over
D=51.5M, i.e. ~255 G float64 element-ops in pure numpy -- hours, and it also
held N chunks of float64 at once (3.2 GB at chunk=4M, N=100).

Every quantity above comes out of an N x N Gram instead, in ONE streaming pass
and one BLAS call per chunk.

The Gram is taken of the DISPLACEMENTS d_i = w_i - ref, not of w_i, and that is
a numerical choice rather than a convenience. Every member descends from one
trunk, so ||w_i - w_j|| is 16-56% of ||w_i|| (measured, N=12 calibration zoo). Recovering a
small difference from `G_ii + G_jj - 2 G_ij` on RAW vectors is catastrophic
cancellation; on displacements the Gram entries are already the scale of the
answer, so nothing cancels. Identities used, all exact:

    ||w_i - w_j||^2 = G_ii + G_jj - 2 G_ij                    (ref cancels)
    ||w_i - ref||^2 = G_ii
    ||w_i - mean||^2 = G_ii - (2/N) sum_j G_ij + (1/N^2) sum_jk G_jk
    ||w_i||^2       = G_ii + 2 t_i + ||ref||^2 ,  t_i = d_i . ref

Reads .npy members with mmap and streams in chunks, so peak RSS is one (N, chunk)
float64 block rather than the whole ensemble.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np


def flat_trunk(trunk_path: str, arch: str, exclude_1d: bool, include_extra: bool):
    """Flatten trunk.pt through the SAME code path that wrote the members."""
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from llmzoo.models.registry import build_zoo_model
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tz", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_zoo.py"))
    tz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tz)

    ck = torch.load(trunk_path, map_location="cpu", weights_only=False)
    model = build_zoo_model(arch, seed=None)
    model.load_state_dict(ck["model"])
    model.eval()
    w, _frag = tz.flatten_model(model, arch, exclude_1d=exclude_1d,
                               include_extra=include_extra)
    return w


def displacement_cosines(G, idx, group, kind, D):
    """Branch-delta orthogonality, the answer to "they all share one trunk".

    d_i = w_i - trunk is this member's task vector. Ilharco et al. 2023 report
    task vectors from a common init as near-orthogonal; the null for two
    independent directions in R^D is cos ~ N(0, 1/D), i.e. |cos| ~ 1/sqrt(D) =
    1.4e-4 at Mini. A zoo whose members merely followed the trunk's momentum
    would instead show cos -> 1.

    Reported in FOUR disjoint pools, because mixing them is exactly the error
    that makes zoo_geometry.json's own scalars look impossible (between_group_mean
    80.02 > 2 x to_trunk_mean 31.19 cannot happen inside one pool -- the existing
    `between` is anchor-only while `pairwise_mean` is all 4950 pairs).

    Then the honest version: remove the mean displacement -- the drift EVERY
    member inherited from the shared trunk -- and re-measure. The centred Gram
    P G P with P = I - 11^T/N is exactly that projection, so this costs nothing.
    """
    N = G.shape[0]
    null = 1.0 / np.sqrt(D)

    def pools(M):
        s = np.sqrt(np.clip(np.diag(M), 1e-300, None))
        C = M / np.outer(s, s)
        acc = {"within_anchor": [], "between_anchor": [],
               "singleton_singleton": [], "anchor_singleton": []}
        for i, j in itertools.combinations(range(N), 2):
            a, b = idx[i], idx[j]
            ka, kb = kind[a], kind[b]
            if ka == "anchor" and kb == "anchor":
                key = ("within_anchor" if group[a] == group[b]
                       else "between_anchor")
            elif ka == "singleton" and kb == "singleton":
                key = "singleton_singleton"
            else:
                key = "anchor_singleton"
            acc[key].append(float(C[i, j]))
        res = {k: (float(np.mean(v)) if v else None) for k, v in acc.items()}
        res["n_pairs"] = {k: len(v) for k, v in acc.items()}
        allc = [c for v in acc.values() for c in v]
        res["all_pairs"] = float(np.mean(allc)) if allc else None
        res["min"] = float(np.min(allc)) if allc else None
        res["max"] = float(np.max(allc)) if allc else None
        ev = np.sort(np.linalg.eigvalsh((M + M.T) / 2))[::-1]
        pos = ev[ev > 0]
        res["leading_eigenvalue_share"] = (float(ev[0] / pos.sum())
                                           if len(pos) else None)
        res["effective_rank"] = (float(pos.sum() ** 2 / (pos ** 2).sum())
                                 if len(pos) else None)
        return res

    P = np.eye(N) - 1.0 / N
    return {
        "cos_null_random_directions": float(null),
        "cos_raw": pools(G),
        "cos_shared_drift_removed": pools(P @ G @ P),
        "gram_member_idx": [int(i) for i in idx],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zoo_dir", required=True, help="the ARCH-level dir")
    p.add_argument("--chunk_bytes", type=float, default=1.5e9,
                   help="peak size of the (N, chunk) float64 working block. The "
                        "chunk length is derived from this and N, so N=12 and "
                        "N=100 both stay inside one --mem=200G job.")
    p.add_argument("--chunk", type=int, default=None,
                   help="override the derived chunk length in ELEMENTS")
    p.add_argument("--json_out", default=None)
    p.add_argument("--dump_gram", default=None, nargs="?", const="",
                   help="write the N x N float64 displacement Gram as .npy "
                        "(default <zoo_dir>/displacement_gram.npy). It is 80 KB "
                        "at N=100 and this pass is the only thing that computes "
                        "it: every pairwise cosine, the shared-drift eigenvector "
                        "and any MDS embedding come out of it offline, forever.")
    p.add_argument("--dump_trunk_flat", default=None, nargs="?", const="",
                   help="write the flattened trunk as float32 .npy (default "
                        "<zoo_dir>/trunk_flat.npy). This is the ONLY supported "
                        "way to hand a reference to analysis/mixture_composition "
                        "without it silently torch.load()ing trunk.pt itself.")
    args = p.parse_args()

    with open(os.path.join(args.zoo_dir, "zoo_meta.json")) as f:
        zmeta = json.load(f)
    arch = zmeta["arch"]
    paths = sorted(glob.glob(os.path.join(args.zoo_dir, "w_*.npy")),
                   key=lambda s: int(s.rsplit("_", 1)[1].split(".")[0]))
    N = len(paths)
    if N < 2:
        print("need >= 2 members"); return 2
    members = [np.load(p, mmap_mode="r") for p in paths]
    D = members[0].shape[0]
    idx = [int(p.rsplit("_", 1)[1].split(".")[0]) for p in paths]
    plan = {m["idx"]: m for m in zmeta["members"]}
    group = {i: plan[i]["mixture_id"] for i in idx}
    kind = {i: plan[i]["kind"] for i in idx}

    trunk = None
    tp = os.path.join(args.zoo_dir, "trunk.pt")
    if os.path.exists(tp):
        print(f"[geom] flattening {tp} …", flush=True)
        trunk = flat_trunk(tp, arch, zmeta["exclude_1d"], zmeta["include_extra"])
        if trunk.shape[0] != D:
            print(f"  trunk D={trunk.shape[0]} != member D={D}; ignoring trunk")
            trunk = None
        elif args.dump_trunk_flat is not None:
            # Write BEFORE the streaming pass. The flatten above is the only
            # torch in this script; if the Gram pass then dies we still keep the
            # artifact, and nobody has to re-import torch to get it back.
            dest = args.dump_trunk_flat or os.path.join(args.zoo_dir,
                                                        "trunk_flat.npy")
            os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
            # float32, exactly like w_*.npy -- analysis/*/zoo.py:open_weights
            # rejects any other dtype, and a float64 copy would double the file.
            np.save(dest, np.ascontiguousarray(trunk, dtype=np.float32))
            print(f"[geom] -> {dest}  ({os.path.getsize(dest) / 1e6:.0f} MB)")

    # Reference for the displacement Gram. The trunk when we have it (it is also
    # the thing `disp` is measured against); member 0 otherwise, which still
    # gives exact pairwise distances because the reference cancels in
    # G_ii + G_jj - 2 G_ij -- it just cannot give `disp` or `spread/disp`.
    ref = trunk if trunk is not None else np.asarray(members[0], dtype=np.float64)
    ref_name = "trunk" if trunk is not None else "member_0"

    chunk = args.chunk or max(1, int(args.chunk_bytes // (8 * N)))
    n_chunks = (D + chunk - 1) // chunk
    print(f"[geom] N={N} D={D:,} ref={ref_name}  {n_chunks} chunks of "
          f"<= {chunk:,} elems ({8 * N * chunk / 1e9:.2f} GB working block)",
          flush=True)

    G = np.zeros((N, N), dtype=np.float64)   # Gram of the displacements
    t = np.zeros(N, dtype=np.float64)        # d_i . ref
    ref_sq = 0.0

    for ci, s in enumerate(range(0, D, chunk)):
        e = min(s + chunk, D)
        r = np.asarray(ref[s:e], dtype=np.float64)
        Dm = np.empty((N, e - s), dtype=np.float64)
        for i, m in enumerate(members):
            # copy out of the memmap, then subtract in place -- one temporary,
            # not one per member per pair as the old inner loop had.
            Dm[i, :] = m[s:e]
        Dm -= r
        G += Dm @ Dm.T
        t += Dm @ r
        ref_sq += float(r @ r)
        del Dm, r
        if n_chunks <= 20 or ci % max(1, n_chunks // 10) == 0:
            print(f"  chunk {ci + 1}/{n_chunks}", flush=True)

    if args.dump_gram is not None:
        dest = args.dump_gram or os.path.join(args.zoo_dir,
                                              "displacement_gram.npy")
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        # Rows are in the order of `idx` below (members sorted numerically),
        # NOT necessarily 0..N-1 -- a partial zoo has holes. Read the order out
        # of gram_member_idx in zoo_geometry.json, never assume it.
        np.save(dest, G)
        print(f"[geom] -> {dest}  ({G.nbytes / 1024:.0f} KB, "
              f"{N}x{N} float64, ref={ref_name})")

    diag = np.diag(G).copy()
    to_ref = np.sqrt(np.clip(diag, 0.0, None))
    # ||w_i||^2 = ||d_i + ref||^2
    nrm = np.sqrt(np.clip(diag + 2.0 * t + ref_sq, 0.0, None))
    # ||w_i - mean||^2, from the Gram alone
    rowsum = G.sum(axis=1)
    total = float(G.sum())
    to_mean = np.sqrt(np.clip(diag - 2.0 * rowsum / N + total / (N * N),
                              0.0, None))

    pairs = list(itertools.combinations(range(N), 2))
    pd = {(i, j): float(np.sqrt(max(diag[i] + diag[j] - 2.0 * G[i, j], 0.0)))
          for i, j in pairs}

    # between/within over the groups that HAVE a within. Singletons each carry a
    # unique mixture_id, so counting singleton-singleton pairs as "between"
    # would silently redefine the statistic between N=12 (all anchors) and
    # N=100 (20 anchors + 80 unique singletons) and destroy comparability with
    # the N=12 calibration table. Restricting to multi-member groups reproduces
    # that table exactly and stays meaningful at N=100.
    sizes = {}
    for i in idx:
        sizes[group[i]] = sizes.get(group[i], 0) + 1
    multi = {g for g, c in sizes.items() if c > 1}
    within = [pd[(i, j)] for i, j in pairs
              if group[idx[i]] == group[idx[j]] and group[idx[i]] in multi]
    between = [pd[(i, j)] for i, j in pairs
               if group[idx[i]] != group[idx[j]]
               and group[idx[i]] in multi and group[idx[j]] in multi]
    allpd = list(pd.values())

    out = {
        "zoo_dir": args.zoo_dir, "arch": arch, "beta": zmeta.get("beta"),
        "n_members": N, "D": D,
        "reference": ref_name, "n_chunks": n_chunks, "chunk": chunk,
        "member_norm_mean": float(np.mean(nrm)),
        "pairwise_mean": float(np.mean(allpd)),
        "pairwise_min": float(np.min(allpd)),
        "within_group_mean": float(np.mean(within)) if within else None,
        "between_group_mean": float(np.mean(between)) if between else None,
        "between_over_within": (float(np.mean(between) / np.mean(within))
                                if within and between else None),
        "n_within_pairs": len(within),
        "n_between_pairs": len(between),
        "n_multi_member_groups": len(multi),
        "n_anchor_members": sum(1 for i in idx if kind[i] == "anchor"),
        "n_singleton_members": sum(1 for i in idx if kind[i] == "singleton"),
        "to_mean_mean": float(np.mean(to_mean)),
        "mean_frac": float(np.mean(to_mean) / np.mean(allpd)),
        # Per member, in `gram_member_idx` order. N floats, and they are what
        # makes the report reproducible from JSON alone: displacement turns out
        # to track how CONCENTRATED the member's mixture is (corr with max(pi)
        # +0.73 at Mini), which is the thing that explains the composition
        # residual. A pure-stdlib figure builder cannot read the Gram .npy, so
        # if these are not here that panel needs NumPy and a cluster round-trip.
        "to_trunk": [float(v) for v in to_ref],
        "to_mean": [float(v) for v in to_mean],
        "member_norm": [float(v) for v in nrm],
    }
    # Only a TRUNK reference makes these branch deltas. Against member_0 the
    # same arithmetic measures displacement from an arbitrary member, which is a
    # different quantity with the same name -- so label it rather than hide it.
    out.update(displacement_cosines(G, idx, group, kind, D))
    out["cos_reference"] = ref_name
    out["cos_is_branch_delta"] = trunk is not None
    if trunk is not None:
        out.update({
            "trunk_norm": float(np.sqrt(ref_sq)),
            "to_trunk_mean": float(np.mean(to_ref)),
            "disp_rel": float(np.mean(to_ref) / np.sqrt(ref_sq)),
            "spread_over_disp": float(np.mean(allpd) / np.mean(to_ref)),
        })

    print("\n" + "=" * 70)
    print(f"zoo geometry — {arch}  beta={out['beta']}  N={N}  D={D:,}")
    print("=" * 70)
    if trunk is not None:
        print(f"  displacement from trunk   ||w_i-trunk||/||trunk||   "
              f"{out['disp_rel']:.5f}   ({100*out['disp_rel']:.3f}% of the weights moved)")
        print(f"  spread / displacement     mean||w_i-w_j|| / mean||w_i-trunk||   "
              f"{out['spread_over_disp']:.3f}")
        print(f"      sqrt(2)=1.414 means members moved INDEPENDENTLY.")
        print(f"      near 0 means they all moved the SAME direction -> one-dimensional zoo.")
    else:
        print(f"  no trunk.pt here, so displacement is undefined; pairwise "
              f"distances are still exact (the reference cancels).")
    print(f"  centroid fraction         ||w_i-mean||/mean||w_i-w_j||   "
          f"{out['mean_frac']:.3f}   (i.i.d. spread at N={N} would be "
          f"{((N-1)/N/2)**0.5:.3f})")
    if out["between_over_within"]:
        print(f"  between/within (weights)  {out['between_over_within']:.3f}"
              f"    within {out['within_group_mean']:.2f}  between {out['between_group_mean']:.2f}")
        print(f"      over {out['n_multi_member_groups']} multi-member groups "
              f"({out['n_within_pairs']} within / {out['n_between_pairs']} between "
              f"pairs). Unique-pi singletons are excluded by construction, so "
              f"this stays comparable across N.")
    else:
        print(f"  between/within (weights)  undefined -- "
              f"{out['n_multi_member_groups']} multi-member group(s). An "
              f"all-singleton probe has no within-group pair.")
    cr, cd = out["cos_raw"], out["cos_shared_drift_removed"]
    label = "branch-delta cosine" if trunk is not None else (
        "cosine vs member_0  (NOT a branch delta -- no trunk.pt here)")
    print(f"  {label:<26}random-direction null 1/sqrt(D) = "
          f"{out['cos_null_random_directions']:.2e}")
    print(f"      {'pool':<22}{'raw':>10}{'drift removed':>16}{'pairs':>9}")
    for key in ("within_anchor", "between_anchor", "singleton_singleton",
                "anchor_singleton", "all_pairs"):
        n = cr["n_pairs"].get(key, sum(cr["n_pairs"].values()))
        if cr[key] is None:
            continue
        print(f"      {key:<22}{cr[key]:>10.4f}{cd[key]:>16.4f}{n:>9}")
    print(f"      leading eigenvalue share {cr['leading_eigenvalue_share']:.3f} "
          f"raw -> {cd['leading_eigenvalue_share']:.3f} once the drift every "
          f"member inherited from the trunk is removed")
    print("=" * 70)

    dest = args.json_out or os.path.join(args.zoo_dir, "zoo_geometry.json")
    # Create the parent. Both of these scripts do a full streaming pass over the
    # ensemble first -- minutes at Mini, and every member read twice at Medium -- so
    # dying on the WRITE throws all of that away for a missing mkdir.
    _parent = os.path.dirname(os.path.abspath(dest))
    if _parent:
        os.makedirs(_parent, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

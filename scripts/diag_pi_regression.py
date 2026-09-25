#!/usr/bin/env python3
"""
Leave-one-mixture-out predictive R^2 of the branch delta on pi: the INTERIOR
analogue of the anchors-only ANOVA.

    # offline, from the copied-down Grams (seconds, no GPU, no weight reads)
    python scripts/diag_pi_regression.py \
        --gram_dir reports/data/population/mini_b015/weight_embeddings \
        --meta     reports/data/population/mini_b015/zoo_meta.json \
        --geometry reports/data/population/mini_b015/zoo_geometry.json \
        --disagreement reports/data/population/mini_b015/disagreement.npy \
        --json_out reports/data/population/mini_b015/pi_regression.json

    # on-cluster, streaming w_*.npy (what run_population_analysis.py calls)
    python scripts/diag_pi_regression.py --zoo_dir $ARTIFACT_DIR/zoo_b015_mini_n100/gpt2_zoo_mini

WHY THIS EXISTS
---------------
The eta^2 table (tab:pop-anova) is measured on the TWENTY ANCHORS, which sit at
the five one-hot VERTICES of Delta^4 -- the most mutually distinct mixtures the
design contains. It is an upper bound on what an
interior mixture would give. But the conditional generator is asked to work in the INTERIOR, where there are no replicates and so
no within-group term and so no ANOVA. The out-of-sample analogue that does exist
there is a predictive R^2: hold out a mixture, predict its delta from pi alone,
score against predicting the training mean.

THE FOLDS ARE LEAVE-ONE-MIXTURE-OUT, NOT LEAVE-ONE-ROW-OUT
----------------------------------------------------------
This is correctness, not style. The four anchor branches share a pi exactly, so
dropping one row leaves three siblings at the IDENTICAL key and the "prediction"
is retrieval -- the same leak scripts/report_conditioning.py:545 documents for
the decoder direction. 85 folds over 100 members: 5 anchor groups of 4, plus 80
singletons. For a singleton, leave-one-mixture-out and leave-one-row-out coincide.

WHY IT NEEDS NO WEIGHT READS
----------------------------
OLS is a linear smoother, so every prediction is a fixed linear combination of
the training rows and the held-out residual sum of squares is

    SSE_F = tr( R_F  G  R_F^T ),   R_F = E_F - X_F (X_tr^T X_tr)^+ X_tr^T S_tr

which depends on the D-dimensional deltas ONLY through the N x N Gram G. A 100x100
matrix is the entire sufficient statistic; the laptop does the whole ladder in
about a second.

The Gram splits exactly along the layout w = [ extra | block_0 | ... ], because a
Gram is a sum over coordinates -- the same identity diag_block_spectrum.py uses.
So whole / embeddings / blocks are three separate regressions, not one regression
and two subtractions. (R^2 is NOT additive across regions even though the Grams
are: never compute the blocks number as whole minus extra.)

CENTRED OR UNCENTRED GRAM GIVES THE SAME ANSWER, PROVABLY
---------------------------------------------------------
pi sums to 1, so the all-ones vector lies in the column space of every design
here (including the quadratic one, since sum_{j<=k} pi_j pi_k terms recombine to
1). Adding any fixed vector to every delta -- e.g. changing the reference from the
trunk to member 0, or centring -- therefore shifts prediction and baseline
identically and cancels. --self_test asserts it to 1e-12.

WHAT THE DESIGNS ARE
--------------------
The ladder used by the conditioning report, via
designs() at scripts/report_conditioning.py:341:

    pi        5 cols   the linear blend
    pi + H    6 cols   plus the entropy of pi, the one scalar nonlinear feature
    quad     15 cols   { pi_j pi_k : j <= k }; contains the linear span

No intercept anywhere: pi already spans the constant.

A NUMBER THIS SCRIPT DELIBERATELY DOES NOT EMIT
-----------------------------------------------
A "noise ceiling" from the pooled within-anchor spread. It is computed and sealed
as `within_anchor_reference` for the record, but it lands BELOW the achieved
interior R^2 (0.16 against 0.31 at Mini), i.e. replicates at a VERTEX overstate
the seed noise of an interior mixture. It is a caveat, not a ceiling, and quoting
it as one would be wrong in the direction that flatters the result.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

DESIGN_ORDER = ("pi", "pi_entropy", "pi_quadratic")
EVAL_SETS = ("all", "interior", "anchor")
# A fourth cell, scored separately because it changes the TRAINING set and not
# only the evaluation set: restrict the whole problem to the 80 singletons, so
# the anchors neither inform the fit nor anchor the baseline. It is what
# separates "the map is curved" from "the map is curved AT THE VERTICES".
INTERIOR_FIT = "interior_fit"


# ----------------------------------------------------------------- designs
def design_matrix(P: np.ndarray, name: str) -> np.ndarray:
    """P is (N, 5) on the simplex. No intercept: pi spans the constant."""
    if name == "pi":
        return P.copy()
    if name == "pi_entropy":
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.where(P > 0, np.log(np.where(P > 0, P, 1.0)), 0.0)
        return np.column_stack([P, -(P * lg).sum(axis=1)])
    if name == "pi_quadratic":
        d = P.shape[1]
        return np.column_stack([P[:, j] * P[:, k]
                                for j in range(d) for k in range(j, d)])
    raise ValueError(f"unknown design {name!r}")


# ----------------------------------------------------------------- scoring
def _ss(R: np.ndarray, G: np.ndarray) -> float:
    """tr(R G R^T) -- the sum of squares of the rows of R @ D, from the Gram."""
    return float(np.einsum("ij,jk,ik->", R, G, R))


def cv_r2(G: np.ndarray, X: np.ndarray, folds, eval_rows) -> dict:
    """Leave-one-fold-out R^2. Baseline is the TRAINING mean, refit per fold."""
    N = G.shape[0]
    keep = np.zeros(N, dtype=bool)
    keep[list(eval_rows)] = True
    sse = sst = 0.0
    n_eval = n_folds = 0
    for fold in folds:
        te = np.array([i for i in fold if keep[i]], dtype=int)
        if te.size == 0:
            continue
        drop = set(int(i) for i in fold)          # drop the WHOLE mixture group
        tr = np.array([j for j in range(N) if j not in drop], dtype=int)
        Xtr = X[tr]
        A = np.linalg.pinv(Xtr.T @ Xtr) @ Xtr.T   # (p, n_tr)
        E = np.zeros((te.size, N))
        E[np.arange(te.size), te] = 1.0
        B = np.zeros_like(E)
        B[:, tr] = X[te] @ A                      # prediction as a row combo
        M = np.zeros_like(E)
        M[:, tr] = 1.0 / tr.size                  # training-mean baseline
        sse += _ss(E - B, G)
        sst += _ss(E - M, G)
        n_eval += te.size
        n_folds += 1
    return {"r2": 1.0 - sse / sst, "sse": sse, "sst": sst,
            "n_eval": n_eval, "n_folds": n_folds}


def in_sample_r2(G: np.ndarray, X: np.ndarray, eval_rows) -> dict:
    """Fit on all N rows; score the eval subset. The overfit reference for cv_r2."""
    N = G.shape[0]
    rows = np.array(sorted(eval_rows), dtype=int)
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    C = np.eye(N) - np.ones((N, N)) / N
    E = np.zeros((rows.size, N))
    E[np.arange(rows.size), rows] = 1.0
    sse = _ss(E @ (np.eye(N) - H), G)
    sst = _ss(E @ C, G)
    return {"r2": 1.0 - sse / sst, "sse": sse, "sst": sst, "n_eval": int(rows.size)}


# ----------------------------------------------------------------- Grams
def region_gram(members, lo: int, hi: int, N: int, chunk: int) -> np.ndarray:
    """Gram of displacements w_i - w_0 over coordinates [lo, hi). Streaming."""
    G = np.zeros((N, N), dtype=np.float64)
    Dm = np.empty((N, chunk), dtype=np.float64)
    for s in range(lo, hi, chunk):
        e = min(s + chunk, hi)
        blk = Dm[:, : e - s]
        for i, m in enumerate(members):
            blk[i, :] = m[s:e]
        blk -= blk[0]
        G += blk @ blk.T
    return G


def load_region_grams(gram_dir: str) -> dict:
    """Read the region Grams weight_embeddings already wrote.

    Verified against ladder/blockspec_*.json at all three rungs: the traces agree
    to ~1e-15 and extra + blocks == whole to 1e-16.
    """
    whole = np.load(os.path.join(gram_dir, "centered_gram.npy"))
    extra = np.load(os.path.join(gram_dir, "extra", "centered_gram.npy"))
    layers = sorted(glob.glob(os.path.join(gram_dir, "layer_*")),
                    key=lambda s: int(s.rsplit("_", 1)[1]))
    if not layers:
        raise SystemExit(f"no layer_* Grams under {gram_dir}")
    blocks = sum(np.load(os.path.join(L, "centered_gram.npy")) for L in layers)
    rel = abs(np.trace(extra) + np.trace(blocks) - np.trace(whole)) / abs(np.trace(whole))
    if rel > 1e-10:
        raise SystemExit(f"region Grams are not additive (rel {rel:.2e}) -- "
                         f"layer set under {gram_dir} is incomplete")
    return {"whole": whole, "embeddings": extra, "blocks": blocks,
            "n_layers": len(layers)}


def mds_gram(delta: np.ndarray) -> tuple:
    """Classical MDS / dbRDA: turn a dissimilarity matrix into a Gram."""
    N = delta.shape[0]
    C = np.eye(N) - np.ones((N, N)) / N
    G = -0.5 * C @ (delta ** 2) @ C
    G = 0.5 * (G + G.T)
    ev = np.linalg.eigvalsh(G)
    neg = float(abs(ev[ev < 0].sum()) / ev[ev > 0].sum()) if (ev > 0).any() else float("nan")
    return G, neg


# ----------------------------------------------------------------- pi geometry
def pi_geometry(P: np.ndarray, kind: np.ndarray) -> dict:
    N = P.shape[0]
    iu = np.triu_indices(N, 1)
    L1 = np.abs(P[:, None, :] - P[None, :, :]).sum(-1)[iu]
    a = kind == "anchor"
    pool = np.where(a[:, None] & a[None, :], "AA",
                    np.where(~a[:, None] & ~a[None, :], "SS", "AS"))[iu]
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(P > 0, np.log(np.where(P > 0, P, 1.0)), 0.0)
    H = -(P * lg).sum(axis=1)
    out = {}
    for tag, m in (("singleton_singleton", pool == "SS"),
                   ("anchor_singleton", pool == "AS"),
                   ("between_anchor", (pool == "AA") & (L1 > 0)),
                   ("within_anchor", (pool == "AA") & (L1 == 0))):
        v = L1[m]
        out[tag] = {"n_pairs": int(m.sum()),
                    "l1_mean": float(v.mean()) if v.size else None,
                    "l1_median": float(np.median(v)) if v.size else None,
                    "l1_max": float(v.max()) if v.size else None}
    s = kind == "singleton"
    out["max_pi_singleton_mean"] = float(P[s].max(axis=1).mean())
    out["max_pi_singleton_median"] = float(np.median(P[s].max(axis=1)))
    out["entropy_singleton_mean"] = float(H[s].mean())
    out["entropy_uniform"] = float(np.log(P.shape[1]))
    return out


def disagreement_vs_l1(P: np.ndarray, kind: np.ndarray, delta: np.ndarray) -> dict:
    """Fit disagreement ~ a + b*L1 on ANCHOR-INVOLVING pairs, extrapolate to the
    singleton pool. The gap is the saturation the pool ordering hides."""
    N = P.shape[0]
    iu = np.triu_indices(N, 1)
    L1 = np.abs(P[:, None, :] - P[None, :, :]).sum(-1)[iu]
    y = delta[iu]
    a = kind == "anchor"
    ss = (~a[:, None] & ~a[None, :])[iu]
    fit = ~ss
    A = np.column_stack([np.ones(fit.sum()), L1[fit]])
    b0, b1 = np.linalg.lstsq(A, y[fit], rcond=None)[0]
    # The pool means as (pi-distance, disagreement) points, and the secant slopes
    # between them. A single linear slope is the wrong summary: the response is
    # CONVEX in pi-distance, so the interior pool sits where the curve is
    # flattest and is doubly favoured -- close-packed in pi AND in the shallow
    # part of the response. Quoting the pool means alone hides the second effect.
    #
    # NO RATIO OF THE SECANTS IS EMITTED. The first one runs 0.0334, 0.0040,
    # -0.0054 along the ladder -- it passes through zero, because the within-
    # anchor/singleton-singleton floor inverts at Medium -- so any ratio against
    # it has an unstable denominator and changes sign for reasons that have
    # nothing to do with the quantity of interest. Quote the slopes themselves.
    a_ = kind == "anchor"
    pools = {
        "within_anchor": (a_[:, None] & a_[None, :])[iu] & (L1 == 0),
        "singleton_singleton": ss,
        "anchor_singleton": (a_[:, None] ^ a_[None, :])[iu],
        "between_anchor": (a_[:, None] & a_[None, :])[iu] & (L1 > 0),
    }
    pts = [(float(L1[m].mean()), float(y[m].mean())) for m in pools.values()]
    pts.sort()
    secants = [{"l1_from": pts[i][0], "l1_to": pts[i + 1][0],
                "slope": (pts[i + 1][1] - pts[i][1]) / (pts[i + 1][0] - pts[i][0])}
               for i in range(len(pts) - 1)]
    return {"intercept": float(b0), "slope": float(b1),
            "n_fit_pairs": int(fit.sum()),
            "singleton_predicted": float((b0 + b1 * L1[ss]).mean()),
            "singleton_observed": float(y[ss].mean()),
            "pool_points": [{"l1": a, "disagreement": b} for a, b in pts],
            "secant_slopes": secants}


def within_anchor_reference(G: np.ndarray, mixture_id: np.ndarray,
                            kind: np.ndarray) -> dict:
    """Pooled within-anchor SS per member, and the 'ceiling' it implies for the
    interior. Sealed for the record; see the module docstring -- DO NOT QUOTE IT
    AS A CEILING, it lands below the achieved interior R^2."""
    N = G.shape[0]
    a = np.where(kind == "anchor")[0]
    ssw = 0.0
    groups = sorted(set(mixture_id[a]))
    for g in groups:
        ix = np.where(mixture_id == g)[0]
        E = np.zeros((ix.size, N))
        E[np.arange(ix.size), ix] = 1.0
        M = np.zeros_like(E)
        M[:, ix] = 1.0 / ix.size
        ssw += _ss(E - M, G)
    dof = a.size - len(groups)
    per_member = ssw / dof
    s = np.where(kind == "singleton")[0]
    E = np.zeros((s.size, N))
    E[np.arange(s.size), s] = 1.0
    M = np.zeros_like(E)
    M[:, s] = 1.0 / s.size
    sst_s = _ss(E - M, G)
    return {"within_ss_pooled": float(ssw), "dof": int(dof),
            "within_ss_per_member": float(per_member),
            "singleton_ss_about_singleton_mean": float(sst_s),
            "implied_ceiling_NOT_A_CEILING": float(1.0 - per_member * s.size / sst_s)}


# ----------------------------------------------------------------- driver
def analyse(G: np.ndarray, P: np.ndarray, folds, eval_rows: dict,
            shuffle_seed: int | None = 0) -> dict:
    out = {"trace_centred": float(np.trace(
        (np.eye(G.shape[0]) - np.ones(G.shape) / G.shape[0]) @ G
        @ (np.eye(G.shape[0]) - np.ones(G.shape) / G.shape[0]))), "designs": {}}
    for dn in DESIGN_ORDER:
        X = design_matrix(P, dn)
        cell = {"n_cols": int(X.shape[1]),
                "rank": int(np.linalg.matrix_rank(X))}
        for es in EVAL_SETS:
            rows = eval_rows[es]
            cv = cv_r2(G, X, folds, rows)
            ins = in_sample_r2(G, X, rows)
            cell[es] = {"r2_lomo": cv["r2"], "r2_in_sample": ins["r2"],
                        "n_eval": cv["n_eval"], "n_folds": cv["n_folds"]}
        # Interior-only fit: sub-Gram on the singleton rows, folds re-indexed.
        sub = np.asarray(eval_rows["interior"], dtype=int)
        Gs = G[np.ix_(sub, sub)]
        Xs = design_matrix(P[sub], dn)
        sfolds = [np.array([i]) for i in range(sub.size)]   # each pi is unique
        cvs = cv_r2(Gs, Xs, sfolds, np.arange(sub.size))
        cell[INTERIOR_FIT] = {
            "r2_lomo": cvs["r2"],
            "r2_in_sample": in_sample_r2(Gs, Xs, np.arange(sub.size))["r2"],
            "n_eval": cvs["n_eval"], "n_folds": cvs["n_folds"]}
        if shuffle_seed is not None:
            # The pairing check: permute the pi <-> row join and the signal must
            # vanish. Fails in milliseconds where a real defect would read as a
            # plausible null. Same discipline as pi_code_distance_corr.
            rng = np.random.default_rng(shuffle_seed)
            perm = rng.permutation(P.shape[0])
            cell["shuffled_r2_lomo_interior"] = cv_r2(
                G, design_matrix(P[perm], dn), folds, eval_rows["interior"])["r2"]
        out["designs"][dn] = cell
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("one of these two")
    src.add_argument("--gram_dir", default=None,
                     help="a weight_embeddings/ dir holding the region Grams "
                          "(the offline path; no weight reads)")
    src.add_argument("--zoo_dir", default=None,
                     help="the ARCH-level dir; streams w_*.npy (the cluster path)")
    p.add_argument("--meta", default=None, help="zoo_meta.json (default: beside the source)")
    p.add_argument("--geometry", default=None,
                   help="zoo_geometry.json, read ONLY for gram_member_idx -- the "
                        "Gram row order. Never assume 0..N-1; a partial zoo has holes.")
    p.add_argument("--disagreement", default=None,
                   help="disagreement.npy; adds the function-space region via classical MDS")
    p.add_argument("--chunk_bytes", type=float, default=1.5e9)
    p.add_argument("--shuffle_seed", type=int, default=0)
    p.add_argument("--json_out", default=None)
    p.add_argument("--self_test", action="store_true",
                   help="run the invariance and pairing assertions and exit")
    args = p.parse_args()

    if args.self_test:
        return self_test()
    if (args.gram_dir is None) == (args.zoo_dir is None):
        p.error("give exactly one of --gram_dir / --zoo_dir")

    meta_path = args.meta
    if meta_path is None:
        base = args.zoo_dir or os.path.dirname(os.path.abspath(args.gram_dir))
        meta_path = os.path.join(base, "zoo_meta.json")
    with open(meta_path) as f:
        z = json.load(f)

    if args.gram_dir is not None:
        regions = load_region_grams(args.gram_dir)
        n_layers = regions.pop("n_layers")
        N = regions["whole"].shape[0]
        order = list(range(N))
    else:
        paths = sorted(glob.glob(os.path.join(args.zoo_dir, "w_*.npy")),
                       key=lambda s: int(s.rsplit("_", 1)[1].split(".")[0]))
        order = [int(q.rsplit("_", 1)[1].split(".")[0]) for q in paths]
        members = [np.load(q, mmap_mode="r") for q in paths]
        N = len(members)
        D = int(members[0].shape[0])
        extra = int(z.get("extra_size", 0))
        n_layers = int(z.get("n_layers", 0))
        chunk = max(1, int(args.chunk_bytes // (8 * N)))
        g_extra = region_gram(members, 0, extra, N, chunk)
        g_block = region_gram(members, extra, D, N, chunk)
        regions = {"whole": g_extra + g_block,
                   "embeddings": g_extra, "blocks": g_block}

    if args.geometry:
        with open(args.geometry) as f:
            order = json.load(f)["gram_member_idx"]
        if len(order) != N:
            raise SystemExit(f"gram_member_idx has {len(order)} rows, Gram has {N}")

    # Key pi by idx. Positional order is a hazard: a partial zoo has holes.
    by_idx = {int(m["idx"]): m for m in z["members"]}
    P = np.array([by_idx[i]["pi"] for i in order], dtype=np.float64)
    kind = np.array([by_idx[i]["kind"] for i in order])
    mixture_id = np.array([by_idx[i]["mixture_id"] for i in order])
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-9) or (P < 0).any():
        raise SystemExit("pi rows are not on the simplex")

    folds = [np.where(mixture_id == g)[0] for g in sorted(set(mixture_id))]
    eval_rows = {"all": np.arange(N),
                 "interior": np.where(kind == "singleton")[0],
                 "anchor": np.where(kind == "anchor")[0]}

    out = {
        "source": args.gram_dir or args.zoo_dir,
        "arch": z.get("arch"), "beta": z.get("beta"),
        "n_members": N, "n_layers": n_layers,
        "n_params": z.get("n_params"), "extra_size": z.get("extra_size"),
        "domains": z.get("domains"),
        "n_folds": len(folds),
        "n_anchor_folds": int(sum(1 for f in folds if kind[f[0]] == "anchor")),
        "n_interior_folds": int(sum(1 for f in folds if kind[f[0]] == "singleton")),
        "fold_rule": "leave-one-mixture-out",
        "gram_member_idx": [int(i) for i in order],
        "regions": {},
    }
    for name, G in regions.items():
        out["regions"][name] = analyse(G, P, folds, eval_rows, args.shuffle_seed)
    out["within_anchor_reference"] = {
        name: within_anchor_reference(G, mixture_id, kind)
        for name, G in regions.items()}
    out["pi_geometry"] = pi_geometry(P, kind)

    if args.disagreement:
        delta = np.load(args.disagreement)
        Gf, neg = mds_gram(delta)
        out["regions"]["function"] = analyse(Gf, P, folds, eval_rows, args.shuffle_seed)
        out["regions"]["function"]["mds_negative_eigenvalue_share"] = neg
        out["disagreement_vs_l1"] = disagreement_vs_l1(P, kind, delta)

    txt = json.dumps(out, indent=2, sort_keys=True)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            f.write(txt + "\n")
        print(f"wrote {args.json_out}")
    report(out)
    return 0


def report(out: dict) -> None:
    print("=" * 78)
    print(f"LOMO predictive R^2 of the branch delta on pi — {out['arch']} "
          f"beta={out['beta']}")
    print(f"  N={out['n_members']}  folds={out['n_folds']} "
          f"({out['n_anchor_folds']} anchor + {out['n_interior_folds']} interior), "
          f"{out['fold_rule']}")
    print("=" * 78)
    cols = EVAL_SETS + (INTERIOR_FIT,)
    hdr = f"{'region':<12}{'design':<14}" + "".join(f"{e:>13}" for e in cols)
    print(hdr)
    print("-" * len(hdr))
    for rn, rv in out["regions"].items():
        for dn in DESIGN_ORDER:
            c = rv["designs"][dn]
            print(f"{rn:<12}{dn:<14}" +
                  "".join(f"{c[e]['r2_lomo']:>+13.4f}" for e in cols))
    print("-" * len(hdr))
    sh = out["regions"]["whole"]["designs"]["pi_quadratic"]["shuffled_r2_lomo_interior"]
    print(f"pairing check: shuffled pi<->row join, whole/quad/interior R^2 = {sh:+.4f}")


# ----------------------------------------------------------------- self test
def self_test() -> int:
    rng = np.random.default_rng(0)
    N, d, D = 100, 5, 40
    P = rng.dirichlet(np.ones(d), size=N)
    P[:20] = np.eye(d)[np.repeat(np.arange(d), 4)]      # 5 anchor groups of 4
    mixture_id = np.array([f"anchor_{i // 4}" if i < 20 else f"s{i}" for i in range(N)])
    kind = np.array(["anchor"] * 20 + ["singleton"] * 80)
    folds = [np.where(mixture_id == g)[0] for g in sorted(set(mixture_id))]
    ev = {"all": np.arange(N), "interior": np.arange(20, N), "anchor": np.arange(20)}
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name} {detail}")

    chk("85 folds over 100 members (5 anchor groups of 4 + 80 singletons)",
        len(folds) == 85)

    # (b) an exactly-quadratic map must be recovered at R^2 = 1
    Xq = design_matrix(P, "pi_quadratic")
    Dm = Xq @ rng.normal(size=(Xq.shape[1], D))
    G = Dm @ Dm.T
    r = cv_r2(G, Xq, folds, ev["all"])["r2"]
    chk("exact quadratic map recovers R^2 = 1", abs(r - 1.0) < 1e-8, f"({r:.10f})")
    rl = cv_r2(G, design_matrix(P, "pi"), folds, ev["all"])["r2"]
    chk("a linear design does NOT recover a quadratic map", rl < 0.999, f"({rl:+.4f})")

    # (e) centred and uncentred Grams must agree: 1 is in the column space of X
    Dn = Dm + rng.normal(size=(N, D)) * 0.5
    Gu = Dn @ Dn.T
    C = np.eye(N) - np.ones((N, N)) / N
    Gc = C @ Gu @ C
    for dn in DESIGN_ORDER:
        X = design_matrix(P, dn)
        a = cv_r2(Gu, X, folds, ev["interior"])["r2"]
        b = cv_r2(Gc, X, folds, ev["interior"])["r2"]
        chk(f"centred == uncentred Gram [{dn}]", abs(a - b) < 1e-10, f"(|d|={abs(a-b):.2e})")

    # (d) in-sample must never be beaten by held-out
    for dn in DESIGN_ORDER:
        X = design_matrix(P, dn)
        a = cv_r2(Gu, X, folds, ev["interior"])["r2"]
        b = in_sample_r2(Gu, X, ev["interior"])["r2"]
        chk(f"in-sample >= LOMO [{dn}]", b >= a - 1e-12, f"({b:+.4f} >= {a:+.4f})")

    # (c) the pairing check: a scrambled join must read as a null
    perm = rng.permutation(N)
    rs = cv_r2(Gu, design_matrix(P[perm], "pi_quadratic"), folds, ev["interior"])["r2"]
    chk("shuffled pi<->row join collapses R^2", rs < 0.05, f"({rs:+.4f})")

    # leave-one-ROW-out on anchors leaks: it must beat leave-one-MIXTURE-out
    rowfolds = [np.array([i]) for i in range(N)]
    a_row = cv_r2(Gu, design_matrix(P, "pi"), rowfolds, ev["anchor"])["r2"]
    a_mix = cv_r2(Gu, design_matrix(P, "pi"), folds, ev["anchor"])["r2"]
    chk("leave-one-ROW-out inflates the anchor score (why folds are by mixture)",
        a_row > a_mix, f"(row {a_row:+.4f} > mixture {a_mix:+.4f})")

    print("self test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

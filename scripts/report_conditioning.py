"""
Score conditioning per PCA component. Pure stdlib, so it reproduces off-cluster.

This script is the conditional-measurement instrument, with the corrections below.

What this replaces, and why
---------------------------
The repo's conditioning statistic was

    a_pi = RMS(S - mean S) / within_rms      within_rms from ONE anchor group
    rho  = within_rms / pooled_rms
    recovery = (1 - (a_pi*rho)^2) / (1 - rho^2)

Three defects, all measured on zoo_b015_mini_k99, none of them cosmetic.

1. `within_rms` CANCELS. a_pi*rho = RMS(S - mean S)/pooled_rms identically, so
   recovery depends on the within-mixture estimate only through the ceiling. a_pi and
   recovery are the same statistic up to a per-rank monotone map; reporting both is
   double counting.

2. `recovery` IS MAXIMISED BY COLLAPSE. It is monotone decreasing in the flow's
   conditional spread with no upper bound: a flow emitting a point mass per mixture
   scores 1/(1-rho^2) = 130%. Three of the twelve published cells already read above
   100%, which is that signature and not a flow beating the data. A statistic whose
   optimum is memorisation cannot police memorisation -- the collapse-toward-the-mean
   mechanism, reappearing inside its own fix.

3. IT IS POOLED OVER k, AND THE TWO SUBSPACES DO NOT OVERLAP. Only 0.4% of all
   within-mixture variance lies in c0-c4, the components carrying 84% of the trace;
   80% of it lies in c5-c24. The numerator of a_pi is mostly leading-block and its
   denominator has almost none, so the ratio depends on which anchor group was
   selected -- and train_flow.py's stable sort always selected the tightest of the
   five. Pooling five groups moved rho 0.320 -> 0.483 at k=99, and 0.031 -> 0.454 at
   k=25.

So: report LOCATION where the structure is (c0-c4) and WIDTH where the noise is
(c5-c24), never one scalar over both, and lead with a statistic that is not optimised
by collapse.

The headline
------------
    pi round-trip L1: decode the generated code back to a mixture and measure
    ||pi_hat - pi_req||_1 on the simplex.

It is the dose-response control test moved into code space, so it costs no decode
and no perplexity and therefore runs at hundreds of mixtures rather than five. It is
sensitive to the conditional MEAN, which every dispersion statistic is blind to, and
it cannot be gamed by collapse: a point mass at the wrong mixture scores badly. The
predictor is projected onto the simplex -- without that the raw error at the held-out
vertex reaches a mean L1 of 3.87 across the (top_m, lambda) grid -- and 7.16 for a
single prediction -- beyond the 2.0 maximum between two simplex points, because the
linear map extrapolates outside it.

Landmarks, measured on b015 k=99: null (best constant predictor, the barycentre)
0.836; leave-one-mixture-out ceiling on real members 0.167; sealed interior 0.107;
sealed vertex 0.269. Quote the null AND the ceiling with every number -- a marginal
draw reads 1.149, which overstates any achievement by 37%.

    python scripts/report_conditioning.py --data_dir reports/data/cond
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

# Hand-copied rather than imported from llmzoo.data.mixtures: this module must stay
# importable with no numpy, no torch and no llmzoo, which tests/test_report.py
# enforces. Order is load-bearing -- it fixes which simplex coordinate is which.
DOMAINS = ("web", "code", "math", "books", "multilingual")

# Ridge floor for the rank-deficient designs. Pi sums to 1, so Pi already spans the
# constant and every quadratic cross-term set contains it again; one linear
# combination is therefore degenerate. A relative floor keeps the solve stable
# without moving R2 in the sixth decimal.
RIDGE_FLOOR = 1e-9

Matrix = List[List[float]]


# ---------------------------------------------------------------------------
# small linear algebra, written out rather than imported
# ---------------------------------------------------------------------------

def _gram(X: Matrix) -> Matrix:
    """X^T X for X given as rows."""
    p = len(X[0])
    G = [[0.0] * p for _ in range(p)]
    for row in X:
        for i in range(p):
            ri = row[i]
            if ri == 0.0:
                continue
            Gi = G[i]
            for j in range(i, p):
                Gi[j] += ri * row[j]
    for i in range(p):
        for j in range(i):
            G[i][j] = G[j][i]
    return G


def _xty(X: Matrix, Y: Matrix) -> Matrix:
    """X^T Y for X, Y given as rows."""
    p, q = len(X[0]), len(Y[0])
    B = [[0.0] * q for _ in range(p)]
    for xr, yr in zip(X, Y):
        for i in range(p):
            xi = xr[i]
            if xi == 0.0:
                continue
            Bi = B[i]
            for j in range(q):
                Bi[j] += xi * yr[j]
    return B


def _solve(A: Matrix, B: Matrix) -> Matrix:
    """Gaussian elimination with partial pivoting. A is p x p, B is p x q."""
    p = len(A)
    M = [list(A[i]) + list(B[i]) for i in range(p)]
    q = len(B[0])
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-300:
            raise ValueError("singular design; raise the ridge floor")
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        Mc = M[c]
        for j in range(c, p + q):
            Mc[j] /= d
        for r in range(p):
            if r == c:
                continue
            f = M[r][c]
            if f == 0.0:
                continue
            Mr = M[r]
            for j in range(c, p + q):
                Mr[j] -= f * Mc[j]
    return [row[p:] for row in M]


def _ridge(X: Matrix, Y: Matrix, lam: float = 0.0,
           G: Optional[Matrix] = None, B: Optional[Matrix] = None) -> Matrix:
    """
    Ridge coefficients. G and B may be passed precomputed.

    Reusing the Gram across a lambda grid is the whole reason they are arguments: the
    leave-one-mixture-out loop below is 85 folds x several lambdas x several
    truncations, and rebuilding X^T X each time would dominate the runtime.
    """
    G = [list(r) for r in (G if G is not None else _gram(X))]
    B = B if B is not None else _xty(X, Y)
    p = len(G)
    if lam or RIDGE_FLOOR:
        scale = sum(G[i][i] for i in range(p)) / max(p, 1)
        add = lam + RIDGE_FLOOR * max(scale, 1e-300)
        for i in range(p):
            G[i][i] += add
    return _solve(G, B)


def _predict(X: Matrix, beta: Matrix) -> Matrix:
    q = len(beta[0])
    out = []
    for row in X:
        acc = [0.0] * q
        for xi, br in zip(row, beta):
            if xi == 0.0:
                continue
            for j in range(q):
                acc[j] += xi * br[j]
        out.append(acc)
    return out


def _col(M: Matrix, j: int) -> List[float]:
    return [r[j] for r in M]


def _slice(M: Matrix, lo: int, hi: int) -> Matrix:
    return [r[lo:hi] for r in M]


def _ss_centred(Y: Matrix) -> List[float]:
    n, q = len(Y), len(Y[0])
    mu = [statistics.fmean(_col(Y, j)) for j in range(q)]
    return [sum((r[j] - mu[j]) ** 2 for r in Y) for j in range(q)]


def _ss_resid(Y: Matrix, P: Matrix) -> List[float]:
    q = len(Y[0])
    return [sum((yr[j] - pr[j]) ** 2 for yr, pr in zip(Y, P)) for j in range(q)]


def proj_simplex(v: Sequence[float]) -> List[float]:
    """
    Euclidean projection onto {p >= 0, sum p = 1}.

    Required, not cosmetic. A linear decoder fit on interior mixtures predicts far
    outside the simplex when asked to extrapolate. At the held-out vertex the raw
    prediction is [-0.524, 0.087, 1.137, -0.232, 0.081] -- two negative coordinates,
    one above 1, summing to 0.549 -- and it projects to exactly [0, 0, 1, 0, 0], the
    correct answer. Across the (top_m, lambda) grid the raw error at that vertex
    reaches a mean L1 of 3.87 and a single-prediction L1 of 7.16, against a two-point
    maximum of 2.0; 15 of the 20 default grid points exceed 2.0. Note the vector above
    and the 3.87 come from DIFFERENT grid points (top_m=32/lam=0.1 and
    top_m=32/lam=1e-3 respectively) and the vector's own raw L1 is 1.061. Quoting them
    as one number would be a conflation.
    """
    d = len(v)
    u = sorted(v, reverse=True)
    css, rho, theta = 0.0, 0, 0.0
    best = None
    for i in range(d):
        css += u[i]
        t = (css - 1.0) / (i + 1)
        if u[i] - t > 0:
            best = (i, t)
    rho, theta = best if best else (0, (sum(v) - 1.0) / d)
    return [max(x - theta, 0.0) for x in v]


def l1(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------

def blocks_for(k: int) -> List[Tuple[str, int, int]]:
    """
    Fixed, stated, and coarse past c4 on purpose.

    Individual components are only identified where the eigenvalue gaps are large.
    Measured gap ratios at b015: 1.60, 1.70, 3.02 across c0-c2, then 1.21 at c3/c4
    and 3.59 at c4/c5, then ~1.0 for the tail. So c3 and c4 are a near-degenerate
    PAIR whose individual axes rotate between runs -- b015 reads R2 0.718/0.368 for
    them and b030 reads 0.310/0.769, while the c0-c4 BLOCK is stable at 0.568/0.557.
    Past ~c10 the axes are arbitrary and a per-component number there will not
    reproduce. This is the rank-bound instability of tail eigenvalues, at the
    level of a single component.
    """
    edges = [("c0-c3", 0, 4), ("c4", 4, 5), ("c5-c9", 5, 10),
             ("c10-c24", 10, 25), ("c25+", 25, k)]
    return [(nm, lo, min(hi, k)) for nm, lo, hi in edges if lo < min(hi, k)]


LEAD = ("c0-c4", 0, 5)     # where the structure is: 84% of the trace
NOISE = ("c5-c24", 5, 25)  # where the within-mixture noise is: 80% of it


# ---------------------------------------------------------------------------
# Layer 0 -- the zoo's own conditional structure. No flow involved.
# ---------------------------------------------------------------------------

def anchor_groups(members: List[dict]) -> Dict[str, List[int]]:
    g: Dict[str, List[int]] = {}
    for m in members:
        g.setdefault(str(m["mixture_id"]), []).append(int(m["row"]))
    return {kk: v for kk, v in sorted(g.items()) if len(v) > 1}


def within_components(X: Matrix, groups: Dict[str, List[int]]
                      ) -> Tuple[List[float], int, Dict[str, List[float]]]:
    """
    Pooled per-component within-mixture variance, and each group's own profile.

    Pooled as a sum of squared deviations over a sum of (n_g - 1), the standard
    pooled-variance estimator. The per-group profiles are returned because pooling
    ASSUMES equal group variances and they are not equal: measured per-group rho runs
    0.369 (web) to 0.600 (multilingual), a 1.63x spread in sd. Reporting only the
    pooled number hides that, and the pooled number is the denominator of everything
    downstream.
    """
    q = len(X[0])
    ss = [0.0] * q
    dof = 0
    per: Dict[str, List[float]] = {}
    for nm, rows in groups.items():
        m = len(rows)
        mu = [statistics.fmean(X[r][j] for r in rows) for j in range(q)]
        g_ss = [sum((X[r][j] - mu[j]) ** 2 for r in rows) for j in range(q)]
        per[nm] = [s / (m - 1) for s in g_ss]
        for j in range(q):
            ss[j] += g_ss[j]
        dof += m - 1
    return [s / dof for s in ss], dof, per


def eta2_anchors(X: Matrix, groups: Dict[str, List[int]]
                 ) -> Tuple[List[float], List[float]]:
    """
    One-way ANOVA over the anchor members only: between and within sums of squares.

    Well-posed in a way a mixed decomposition is not. Pooling a within-variance from
    the 20 anchors against a total variance from all 100 members lets the ratio
    exceed 1 -- measured up to 6.55 -- because the anchors are more dispersed than the
    ensemble in some directions. Restricting the total to the same 20 members bounds
    eta2 in [0, 1] by construction.

    Read it with care: five groups spanning a 4-dimensional simplex saturate a
    one-way ANOVA, so eta2 ~ 1 says the within-variance is tiny, NOT that pi acts
    linearly. The regression below is what measures the latter.
    """
    q = len(X[0])
    rows = [r for v in groups.values() for r in v]
    gmean = [statistics.fmean(X[r][j] for r in rows) for j in range(q)]
    ssb = [0.0] * q
    ssw = [0.0] * q
    for rr in groups.values():
        mu = [statistics.fmean(X[r][j] for r in rr) for j in range(q)]
        for j in range(q):
            ssb[j] += len(rr) * (mu[j] - gmean[j]) ** 2
            ssw[j] += sum((X[r][j] - mu[j]) ** 2 for r in rr)
    return ssb, ssw


def block_r2(design: Matrix, X: Matrix, lo: int, hi: int) -> Tuple[float, List[float]]:
    """Trace-weighted R2 over a block, and the per-component values inside it."""
    Y = _slice(X, lo, hi)
    beta = _ridge(design, Y)
    P = _predict(design, beta)
    res, tot = _ss_resid(Y, P), _ss_centred(Y)
    agg = 1.0 - sum(res) / sum(tot) if sum(tot) > 0 else float("nan")
    per = [1.0 - r / t if t > 0 else float("nan") for r, t in zip(res, tot)]
    return agg, per


def entropy(pi: Sequence[float]) -> float:
    return -sum(p * math.log(p) for p in pi if p > 0.0)


def designs(PI: Matrix) -> Dict[str, Matrix]:
    """
    Nested models, in increasing order of what they are allowed to know about pi.

    No intercept anywhere: pi sums to 1, so Pi already spans the constant, and adding
    one would make every design rank-deficient for no gain.

    The `+entropy` row is not a fishing expedition. Pi lives on Delta^4, so the
    between-mixture MEAN subspace spans at most 4 dimensions -- yet the spectrum has
    five spiked directions with a 3.59x gap at c4/c5, and simulating a 4-dimensional
    mean structure predicts a 6x gap at c3/c4 where 1.21 is observed. One extra
    dimension is what a single scalar nonlinear feature of pi adds; generic curvature
    would add ten. Entropy is that scalar, and it survives deleting all 20 anchors
    (singletons-only dR2 = +0.053, F ~ 12.9), so it is not an artifact of the one-hot
    fast path in mixture_stream.
    """
    p = len(PI[0])
    ent = [[entropy(r)] for r in PI]
    quad = [[r[i] * r[j] for i in range(p) for j in range(i, p)] for r in PI]
    return {
        "pi (linear)": PI,
        "pi + entropy": [a + b for a, b in zip(PI, ent)],
        "pi + quadratic": [a + b for a, b in zip(PI, quad)],
    }


def stratified_permutation(members: List[dict], groups: Dict[str, List[int]],
                           rng: random.Random) -> List[int]:
    """
    A row permutation that respects the anchor/singleton split.

    The 16 non-held-out anchor rows carry 43% of the design leverage of Pi with 17%
    of the rows, and `1[anchor]` is nearly collinear with pi-extremeness. An
    unstratified shuffle would let any anchor-versus-singleton offset be attributed
    to pi and inflate the null. So permute the five anchor MIXTURES among the five
    anchor groups (the exchangeable unit is the group, not the member -- the same
    argument report_singleton_probe.py makes for its exact null) and permute
    singleton rows only among singleton rows.
    """
    n = len(members)
    perm = list(range(n))
    keys = list(groups.keys())
    shuffled = keys[:]
    rng.shuffle(shuffled)
    for src, dst in zip(keys, shuffled):
        a, b = groups[src], groups[dst]
        for i, r in enumerate(a):
            perm[r] = b[i % len(b)]
    singles = [int(m["row"]) for m in members
               if str(m["mixture_id"]) not in groups]
    tgt = singles[:]
    rng.shuffle(tgt)
    for r, t in zip(singles, tgt):
        perm[r] = t
    return perm


def layer0(pay: dict, n_perm: int, seed: int) -> dict:
    k = int(pay["k"])
    X: Matrix = pay["codes"]
    members = pay["members"]
    PI: Matrix = [list(m["pi"]) for m in members]
    n = len(X)
    groups = anchor_groups(members)
    out: dict = {"k": k, "n_members": n, "n_anchor_groups": len(groups)}

    # --- invariants. Cheap, and they localise a bug instantly. ---------------
    s = float(pay["code_scale"])
    var = [v / (n - 1) for v in _ss_centred(X)]
    inv = {"target_rms": math.sqrt(sum(sum(c * c for c in r) for r in X) / (n * k)),
           "target_rms_expected": math.sqrt((n - 1) / n),
           "max_abs_col_mean": max(abs(statistics.fmean(_col(X, j)))
                                   for j in range(k))}
    ev = pay.get("eigenvalues")
    if ev:
        # codes_ = U_k diag(sqrt(lambda)) with U_k orthonormal, so Var_j is fixed by
        # the eigenvalue alone. Which is exactly why a naive "expected vs measured
        # magnitude per component" is vacuous on real codes: it reads 1.000 for every
        # j by construction. It only carries information about GENERATED samples.
        pred = [ev[j] / ((n - 1) * s * s) for j in range(k)]
        inv["max_rel_var_error"] = max(abs(var[j] / pred[j] - 1.0)
                                       for j in range(k) if pred[j] > 0)
    out["invariants"] = inv

    # --- within-mixture reference, with its error bar ------------------------
    s2w, dof, per_group = within_components(X, groups)
    tr = inv["target_rms"]
    out["dof_within"] = dof
    out["rho_pooled"] = math.sqrt(statistics.fmean(s2w)) / tr
    out["rho_per_group"] = {nm: math.sqrt(statistics.fmean(v)) / tr
                            for nm, v in per_group.items()}
    # Leave-one-GROUP-out jackknife: the group is the exchangeable unit.
    jk = []
    for drop in groups:
        keep = {nm: v for nm, v in per_group.items() if nm != drop}
        pooled = [statistics.fmean([keep[nm][j] for nm in keep]) for j in range(k)]
        jk.append(math.sqrt(statistics.fmean(pooled)) / tr)
    if len(jk) > 1:
        mj = statistics.fmean(jk)
        out["rho_jackknife_se"] = math.sqrt(
            (len(jk) - 1) / len(jk) * sum((x - mj) ** 2 for x in jk))
    out["s2w"] = s2w

    # --- per-block: where the variance is, where the noise is, what pi explains
    ssb, ssw = eta2_anchors(X, groups)
    dsg = designs(PI)
    tot_var = sum(var)
    rows = []
    for nm, lo, hi in blocks_for(k) + [LEAD, NOISE]:
        den = sum(ssb[lo:hi]) + sum(ssw[lo:hi])
        blk = {"block": nm, "lo": lo, "hi": hi,
               "lambda_share": sum(var[lo:hi]) / tot_var,
               # NOT a variance share, and it can exceed 1 -- measured 4.18 on
               # c10-c24. It compares a within-variance from the 20 anchors against a
               # total from all 100 members, so a value above 1 says the anchor groups
               # are MORE dispersed than the ensemble in that block. That is a real
               # finding about where each group's private noise directions sit, not an
               # error, but it is why the ceiling below is computed on one population.
               "within_vs_ensemble": (sum(s2w[lo:hi]) / sum(var[lo:hi])
                                      if sum(var[lo:hi]) > 0 else float("nan")),
               "share_of_all_within": (sum(s2w[lo:hi]) / sum(s2w)
                                       if sum(s2w) > 0 else float("nan")),
               # Anchors-only one-way ANOVA, so this is bounded in [0, 1] by
               # construction and IS the ceiling -- but at the five vertices only,
               # and five groups spanning Delta^4 saturate a one-way ANOVA, so read
               # it as "the within-mixture noise is this small", not as evidence that
               # pi acts linearly. The R2 columns measure the latter.
               "ceiling_at_anchors": (sum(ssb[lo:hi]) / den if den > 0
                                      else float("nan"))}
        for dn, D in dsg.items():
            blk["R2 " + dn] = block_r2(D, X, lo, hi)[0]
        rows.append(blk)
    out["blocks"] = rows

    # --- per-component, leading block only (where axes are identified) -------
    _, per_lin = block_r2(PI, X, 0, min(8, k))
    out["per_component"] = [
        {"j": j, "lambda_share": var[j] / tot_var, "R2_pi": per_lin[j],
         "within_share": s2w[j] / var[j] if var[j] > 0 else float("nan"),
         "resid_over_within": ((1 - per_lin[j]) * var[j] / s2w[j]
                               if s2w[j] > 0 else float("nan"))}
        for j in range(min(8, k))]

    # --- stratified permutation null on the aggregate ------------------------
    if n_perm > 0:
        rng = random.Random(seed)
        obs, _ = block_r2(PI, X, LEAD[1], min(LEAD[2], k))
        null = []
        for _ in range(n_perm):
            perm = stratified_permutation(members, groups, rng)
            Pp = [PI[perm[i]] for i in range(n)]
            null.append(block_r2(Pp, X, LEAD[1], min(LEAD[2], k))[0])
        null.sort()
        ge = sum(1 for v in null if v >= obs)
        out["permutation"] = {"n_perm": n_perm, "observed": obs,
                              "null_mean": statistics.fmean(null),
                              "null_q95": null[int(0.95 * (len(null) - 1))],
                              "p_value": (ge + 1) / (n_perm + 1)}
    return out


# ---------------------------------------------------------------------------
# Layer 2 -- the headline. Recover pi from the code.
# ---------------------------------------------------------------------------

def fit_decoder(X: Matrix, PI: Matrix, rows: Sequence[int], m: int,
                lam: float) -> Matrix:
    D = [X[r][:m] + [1.0] for r in rows]
    Y = [PI[r] for r in rows]
    return _ridge(D, Y, lam)


def apply_decoder(X: Matrix, rows: Sequence[int], beta: Matrix, m: int) -> Matrix:
    D = [X[r][:m] + [1.0] for r in rows]
    return _predict(D, beta)


def layer2(pay: dict, m_grid: Sequence[int], lam_grid: Sequence[float]) -> dict:
    k = int(pay["k"])
    X: Matrix = pay["codes"]
    members = pay["members"]
    PI: Matrix = [list(m["pi"]) for m in members]
    p = len(PI[0])
    bary = [1.0 / p] * p
    groups = anchor_groups(members)
    hold = pay.get("holdout", {}) or {}
    idx_to_row = {int(m["idx"]): int(m["row"]) for m in members}
    train = [idx_to_row[int(i)] for i in hold.get("train", [])
             if int(i) in idx_to_row] or list(range(len(X)))
    mid = {int(m["row"]): str(m["mixture_id"]) for m in members}

    out: dict = {
        # Two nulls, and the difference matters. The best CONSTANT predictor is the
        # barycentre; a draw from the marginal pi distribution is much worse. Quoting
        # the latter overstates any achievement by 36%, so the barycentre is the null.
        "null_barycentre": statistics.fmean(l1(r, bary) for r in PI),
        "null_marginal": statistics.fmean(
            l1(PI[i], PI[j]) for i in range(len(PI)) for j in range(len(PI))
            if i != j),
        "max_possible_l1": 2.0,
    }

    # Leave-one-MIXTURE-out, not leave-one-row-out: dropping one branch of an anchor
    # leaves three siblings at the identical pi, which leaks the answer.
    tr_groups: Dict[str, List[int]] = {}
    for r in train:
        tr_groups.setdefault(mid[r], []).append(r)
    best = None
    grid = []
    for m in [x for x in m_grid if x <= k]:
        for lam in lam_grid:
            errs = []
            for g, te in tr_groups.items():
                fit = [r for r in train if mid[r] != g]
                try:
                    beta = fit_decoder(X, PI, fit, m, lam)
                except ValueError:
                    errs = None
                    break
                for r, pr in zip(te, apply_decoder(X, te, beta, m)):
                    errs.append(l1(proj_simplex(pr), PI[r]))
            if not errs:
                continue
            e = statistics.fmean(errs)
            grid.append({"top_m": m, "lam": lam, "lomo_l1": e,
                         "skill": 1.0 - e / out["null_barycentre"]})
            if best is None or e < best["lomo_l1"]:
                best = grid[-1]
    out["grid"] = grid
    out["selected"] = best
    if best is None:
        return out

    m, lam = best["top_m"], best["lam"]
    beta = fit_decoder(X, PI, train, m, lam)
    out["ceiling_lomo_l1"] = best["lomo_l1"]
    for nm in ("interior", "vertex"):
        rows = [idx_to_row[int(i)] for i in hold.get(nm, []) if int(i) in idx_to_row]
        if not rows:
            continue
        raw = apply_decoder(X, rows, beta, m)
        out["sealed_" + nm] = {
            "n": len(rows),
            "l1_projected": statistics.fmean(l1(proj_simplex(r), PI[q])
                                             for r, q in zip(raw, rows)),
            "l1_raw": statistics.fmean(l1(r, PI[q]) for r, q in zip(raw, rows)),
        }
    out["decoder"] = {"top_m": m, "lam": lam, "beta": beta}
    return out


# ---------------------------------------------------------------------------
# Layers 1 and 3 -- per sealed flow. Need `cells` in the payload.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Proper scoring rules
#
# WHY THESE AND NOT THE MOMENT STATISTICS ABOVE
# ---------------------------------------------
# `width` and `location_B` are moments, and `roundtrip_l1` is a mean per-sample
# distance to the target. That last one is the problem: by Jensen, E||f(X) - pi||_1
# is minimised by a POINT MASS at the optimum, so RT is improper by construction and
# nothing tunes it out. Measured on the real Mini code matrix at k=50, leave-one-
# branch-out over the 20 anchor members, M=256 samples per condition:
#
#     generator     width    B        RT       ENERGY   VARIOGRAM
#     G_correct     0.996    0.004    0.157    2.818    164.8
#     G_collapse    0.000    0.000    0.131 <- WINS     74.3   <- also wins
#     G_wide        2.013    0.016    0.197    3.096    360.4
#     G_marginal    0.621    3643     1.632    10.005   403.7
#
# A memorising point mass BEATS the correct conditional on RT by 16%, and `width`
# reads better for the generator that ignores pi entirely (0.621) than for the
# correct one (0.996).
#
# The energy score repairs exactly this. Its -1/2 E||X - X'|| term is the diversity
# credit RT lacks, and it is STRICTLY PROPER: uniquely minimised at the true
# conditional. It also needs exactly ONE observation per condition and as many model
# samples as you like -- which is this regime (80 of 85 mixtures are seen once) and
# is precisely what cheap sampling buys.
#
# It is weakly sensitive to over-dispersion, though (G_wide is only 1.10x), and the
# variogram score is the complement -- 2.19x on G_wide, because it reads the
# CORRELATION structure across components rather than the spread. But variogram
# alone rewards collapse (0.451x). Neither is sufficient; the PAIR is correctly
# ordered on all four constructed generators, which is the property being bought.

def energy_score(S: Matrix, y: Sequence[float], beta: float = 1.0) -> float:
    """
    ES(P, y) = E||X - y||^beta - 1/2 E||X - X'||^beta, strictly proper for
    beta in (0, 2). Lower is better. beta=1 is the multivariate CRPS.

    The second term is computed over ordered pairs including i == j (which
    contribute 0), i.e. divided by M^2 rather than M(M-1). That is the V-statistic
    form; it is slightly biased toward rewarding spread at small M but is the form
    whose minimiser is the true distribution for a FIXED M, which is what a
    finite-sample comparison between generators needs.
    """
    M = len(S)
    if M == 0:
        return float("nan")
    def _nrm(u, v) -> float:
        return math.sqrt(sum((x - z) ** 2 for x, z in zip(u, v)))

    a = statistics.fmean(_nrm(sm, y) ** beta for sm in S)
    tot = 0.0
    for i in range(M):
        for j in range(i + 1, M):
            tot += _nrm(S[i], S[j]) ** beta
    # 2 * sum over i<j == sum over ordered pairs; /M^2 is the V-statistic form.
    return a - 0.5 * (2.0 * tot) / (M * M)


def crps(samples: Sequence[float], y: float) -> float:
    """
    CRPS(F, y) = E|X - y| - 1/2 E|X - X'|, the univariate continuous ranked
    probability score. Strictly proper, lower is better, and in the same units as
    the forecast variable -- which is the reason it is the headline here and the
    multivariate energy score is not.

    It needs exactly ONE observation per condition, so it is the score this zoo
    permits: 80 of its 85 mixtures are seen once.

    THE CLOSED FORM, not the double sum. With samples sorted ascending and i
    one-indexed,
        sum_{i,j} |x_i - x_j| = 2 * sum_i (2i - M - 1) * x_(i),
    so the whole thing is O(M log M) instead of O(M^2). Same V-statistic convention
    as energy_score (divide by M^2, not M(M-1)), so the two agree exactly on 1-D
    input -- tests/test_conditioning.py asserts it.

    WHY THIS AND NOT A ROUND-TRIP DISTANCE. E|f(X) - y| is minimised by a point mass
    at the optimum, by Jensen, for any convex loss -- so a generator that has
    memorised the conditional MEAN beats the correct conditional on it. That is
    measured, not hypothetical: on the real Mini code matrix a collapsed generator
    scores RT 0.131 against the correct conditional's 0.157. The `- 1/2 E|X - X'|`
    term is exactly the diversity credit that repairs this.
    """
    M = len(samples)
    if M == 0:
        return float("nan")
    x = sorted(float(v) for v in samples)
    a = statistics.fmean(abs(v - y) for v in x)
    spread = sum((2 * (i + 1) - M - 1) * x[i] for i in range(M))
    return a - spread / (M * M)


def crps_vector(S: Matrix, y: Sequence[float],
                dims: Optional[Sequence[int]] = None) -> float:
    """Mean per-coordinate CRPS over `dims` (default: all of them).

    Averaged rather than summed so it reads per-coordinate and does not change
    meaning when k changes -- the same convention report_functional.err uses across
    the five domains.
    """
    if not S:
        return float("nan")
    idx = list(range(len(y))) if dims is None else list(dims)
    if not idx:
        return float("nan")
    return statistics.fmean(
        crps([row[j] for row in S], float(y[j])) for j in idx)


def skill(score: float, reference: float) -> float:
    """1 - score/reference. 1.0 = perfect, 0.0 = no better than the reference,
    negative = worse than it. NaN-safe and zero-safe so a degenerate reference
    reads as "no verdict" rather than as a spectacular result."""
    if score != score or reference != reference or abs(reference) < 1e-12:
        return float("nan")
    return 1.0 - score / reference


def variogram_score(S: Matrix, y: Sequence[float], p: float = 0.5,
                    max_dim: int = 24) -> float:
    """
    Scheuerer-Hamill variogram score of order p: sum over component pairs of
    (|y_a - y_b|^p - E|X_a - X_b|^p)^2. Proper, and sensitive to the dependence
    structure the energy score barely sees.

    Capped at `max_dim` components because the cost is O(d^2 M) and the pairs beyond
    the leading blocks are private within-member noise (c25+ carries 19% of all
    within-mixture variance in nearly member-specific directions), which no generator
    can or should reproduce.
    """
    d = min(len(y), max_dim)
    if d < 2 or not S:
        return float("nan")
    tot = 0.0
    for a in range(d):
        for b in range(a + 1, d):
            gy = abs(y[a] - y[b]) ** p
            gs = statistics.fmean(abs(sm[a] - sm[b]) ** p for sm in S)
            tot += (gy - gs) ** 2
    return tot


def pit_ranks(S: Matrix, y: Sequence[float], dims: Sequence[int]) -> List[float]:
    """
    Per-component PIT: the rank of the real value among the generated ones,
    normalised to [0, 1].

    Simulation-based calibration, one observation per condition. Uniform ranks mean
    the conditional is calibrated; ranks piling at 0 and 1 mean the generator is too
    TIGHT (the truth keeps landing outside it) -- the collapse signature -- and ranks
    piling in the middle mean it is too wide. Unlike `width` this is conditional by
    construction and needs no separate noise-floor estimate.
    """
    M = len(S)
    out = []
    for j in dims:
        below = sum(1 for sm in S if sm[j] < y[j])
        ties = sum(1 for sm in S if sm[j] == y[j])
        out.append((below + 0.5 * ties) / M if M else float("nan"))
    return out


def score_cell(pay: dict, l0: dict, l2: dict, cell: dict) -> dict:
    k = int(cell["dim"])
    X: Matrix = pay["codes"]
    members = pay["members"]
    PI: Matrix = [list(m["pi"]) for m in members]
    s2w = l0["s2w"]
    var = [v / (len(X) - 1) for v in _ss_centred(X)]
    tbl = {t["label"]: t for t in pay["targets"]}
    res = {"name": cell["name"], "cond_mode": cell["cond_mode"],
           "dim": k, "targets": []}
    dec = l2.get("decoder")

    for t in cell["targets"]:
        lab = t["label"]
        src = tbl.get(lab)
        row = {"label": lab, "kind": t["kind"], "n": t["n"]}
        mean, v = t["mean"], t["var"]

        # --- Layer 1: marginal profile. Vacuous on real codes by the eigenvalue
        # identity; NOT vacuous here. A flow can put all its energy in c0-c3 and
        # still read pooled RMS 1.0, because those components ARE 80% of the trace.
        for nm, lo, hi in (LEAD, NOISE):
            tv, sv = sum(var[lo:hi]), sum(v[lo:hi])
            row[f"profile_{nm}"] = sv / tv if tv > 0 else float("nan")

        # --- Layer 1: width, read where the within-mixture noise actually lives.
        lo, hi = NOISE[1], min(NOISE[2], k)
        sw = sum(s2w[lo:hi])
        row["width_c5_c24"] = (sum(v[lo:hi]) / sw) ** 0.5 if sw > 0 else float("nan")

        # --- Layer 1: location, read where the structure is. The reference is the
        # anchor group's EMPIRICAL mean where replicates exist -- unbiased, and not
        # the regression surface, which any generator that learned that surface would
        # match by construction.
        if src and src.get("member_rows"):
            rows = src["member_rows"]
            lo, hi = LEAD[1], min(LEAD[2], k)
            ref = [statistics.fmean(X[r][j] for r in rows) for j in range(lo, hi)]
            m_g = len(rows)
            num = sum((mean[lo + i] - ref[i]) ** 2 for i in range(hi - lo))
            den = sum(s2w[lo:hi])
            row["location_B_c0_c4"] = num / den if den > 0 else float("nan")
            # A correct flow does NOT read 0. Both estimates carry error, so
            #
            #     E[B] = 1/n + 1/m_g
            #
            # with NO block-width factor: B is a ratio of sums, sum_j (dmu_j)^2 over
            # sum_j s2w_j, not a sum of per-component ratios. The ratio-of-sums form
            # is chosen deliberately -- the per-component form is a proper chi-square
            # but it up-weights the components where s2w_j is tiny, which in this
            # block is all of them, and those are the least identified axes.
            #
            # The 1/m_g term dominates at m_g = 4, so more SAMPLES barely move this:
            # 1/64 + 1/4 = 0.266 against 1/256 + 1/4 = 0.254. Only more BRANCHES per
            # anchor tighten it, which is a zoo-composition fact, not a measurement
            # one. And note the direction of failure: B far BELOW its expectation at
            # an anchor means the flow reproduced that group's empirical mean too
            # well, i.e. memorisation -- the opposite tail from a flow that missed.
            row["location_B_expected"] = 1.0 / t["n"] + 1.0 / m_g

        # --- Layer 2: the headline, on this flow's samples.
        #
        # TWO versions, and only one is comparable to the ceiling.
        #
        #   roundtrip_l1          decodes each INDIVIDUAL sample and averages the L1.
        #                         This is the honest number: a generated model is one
        #                         draw, and the leave-one-mixture-out ceiling is one
        #                         real member. Like for like.
        #   roundtrip_l1_of_mean  decodes the mean of n samples. Variance-reduced by
        #                         ~sqrt(n), so it is NOT comparable to a per-member
        #                         ceiling -- and it is why the best cells first read
        #                         BELOW the ceiling, which is impossible for an
        #                         honest estimator and was a comparison bug, not a
        #                         result. Kept because the gap between the two
        #                         measures how much of the round-trip error is
        #                         per-sample noise rather than a biased mean.
        if dec:
            m = dec["top_m"]
            if m <= k:
                want = src["pi"] if src else t.get("pi")
                pr = _predict([mean[:m] + [1.0]], dec["beta"])[0]
                row["roundtrip_l1_of_mean"] = (l1(proj_simplex(pr), want)
                                               if want else None)
                raws = t.get("samples") or []
                if raws and want:
                    per = [l1(proj_simplex(_predict([r[:m] + [1.0]],
                                                    dec["beta"])[0]), want)
                           for r in raws]
                    row["roundtrip_l1"] = statistics.fmean(per)
                    row["roundtrip_l1_n"] = len(per)
                else:
                    row["roundtrip_l1"] = row["roundtrip_l1_of_mean"]

        # --- Proper scores. These need SAMPLES, not moments, which is why
        # diag_conditioning dumps a raw subsample at all.
        #
        # Scored on c0-c24: the leading block carries the conditional location and
        # c5-c24 the within-mixture noise. c25+ is excluded deliberately -- it is
        # nearly member-private noise (99.1% of the web anchor's within-variance
        # lives there), so no generator can or should reproduce it, and including it
        # would let tail mismatch swamp the signal.
        #
        # The absolute value of an energy score is scale-dependent and means nothing
        # on its own, so each is reported as SKILL against the marginal: the same
        # score with the real training codes themselves used as the sample set, i.e.
        # the best a generator that ignores pi can do. skill = 1 - ES/ES_marginal.
        # 0 means "no better than ignoring the mixture"; 1 is unreachable.
        raw = t.get("samples") or []
        if raw and src and src.get("member_rows"):
            hi_s = min(NOISE[2], k)
            S = [r[:hi_s] for r in raw]
            Xm = [r[:hi_s] for r in X]
            es, vs, pits, esn, vsn = [], [], [], [], []
            for r in src["member_rows"]:
                y = X[r][:hi_s]
                es.append(energy_score(S, y))
                vs.append(variogram_score(S, y))
                pits += pit_ranks(S, y, range(hi_s))
                # The marginal reference excludes the observation itself, or the
                # null would score itself against a set containing it.
                Xo = [Xm[q] for q in range(len(Xm)) if q != r]
                esn.append(energy_score(Xo, y))
                vsn.append(variogram_score(Xo, y))
            row["energy_score"] = statistics.fmean(es)
            row["energy_null"] = statistics.fmean(esn)
            row["energy_skill"] = (1.0 - statistics.fmean(es) / statistics.fmean(esn)
                                   if statistics.fmean(esn) > 0 else float("nan"))
            row["variogram_score"] = statistics.fmean(vs)
            row["variogram_skill"] = (
                1.0 - statistics.fmean(vs) / statistics.fmean(vsn)
                if statistics.fmean(vsn) > 0 else float("nan"))
            row["n_energy_obs"] = len(src["member_rows"])
            row["n_energy_samples"] = len(S)
            # Calibration: under a calibrated conditional the PIT ranks are uniform,
            # so 10% land outside [0.05, 0.95]. Collapse drives this toward 1.0
            # (the truth always lands outside a too-tight ensemble); an over-wide
            # conditional drives it toward 0 and pushes pit_mean toward 0.5.
            row["pit_mean"] = statistics.fmean(pits)
            row["pit_frac_extreme"] = statistics.fmean(
                1.0 if (q < 0.05 or q > 0.95) else 0.0 for q in pits)

        # --- Layer 3: nearest training code, on the raw subsample.
        if raw:
            lo, hi = LEAD[1], min(LEAD[2], k)
            nn = []
            for sm in raw:
                d = [(sum((sm[j] - X[r][j]) ** 2 for j in range(lo, hi)), r)
                     for r in range(len(X))]
                nn.append(min(d))
            row["nn_dist_mean"] = statistics.fmean(math.sqrt(d) for d, _ in nn)
            hits = [r for _, r in nn]
            row["nn_distinct_rows"] = len(set(hits))
            row["nn_modal_row"] = max(set(hits), key=hits.count)
        res["targets"].append(row)
    return res


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _f(x, w=8, d=4) -> str:
    if x is None:
        return " " * (w - 1) + "-"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return " " * (w - 3) + "n/a"
    return f"{x:{w}.{d}f}"


def render(pay: dict, l0: dict, l2: dict, cells: List[dict]) -> None:
    k = l0["k"]
    print("=" * 96)
    print(f"CONDITIONING — run={pay.get('run_name')} k={k} "
          f"N={l0['n_members']} arch={pay.get('arch')}")
    print(f"  pi fingerprint {str(pay['fingerprints'].get('pi'))[:16]}   "
          f"code_stats {str(pay['fingerprints'].get('code_stats'))[:16]}")
    print("=" * 96)

    inv = l0["invariants"]
    print("\n-- invariants (a failure here localises a bug before any statistic) --")
    print(f"  target_rms {inv['target_rms']:.6f} vs expected "
          f"sqrt((N-1)/N) = {inv['target_rms_expected']:.6f}")
    print(f"  max |column mean|                 {inv['max_abs_col_mean']:.3e}"
          f"   (Gram is centred, so 0)")
    if "max_rel_var_error" in inv:
        print(f"  max |Var_j/(lam_j/(N-1)s^2) - 1|  {inv['max_rel_var_error']:.3e}"
              f"   (exact by construction)")

    print(f"\n-- within-mixture reference: {l0['n_anchor_groups']} groups, "
          f"{l0['dof_within']} dof --")
    se = l0.get("rho_jackknife_se")
    print(f"  rho pooled = {l0['rho_pooled']:.4f}"
          + (f" +/- {se:.4f}  ({100 * se / l0['rho_pooled']:.0f}%, "
             f"leave-one-group-out)" if se else ""))
    print("  per group : " + "  ".join(
        f"{nm.replace('anchor_', '')}={v:.4f}"
        for nm, v in sorted(l0["rho_per_group"].items())))
    print("  Pooling ASSUMES equal group variances. It is reported because it is the")
    print("  denominator of every width statistic, not because it is a good model.")

    print("\n-- per block: where the variance is, where the noise is, what pi explains --")
    print(f"  {'block':9s} {'lam share':>9s} {'wtn/ens':>8s} {'of all wtn':>10s} "
          f"{'ceiling':>8s} {'R2 lin':>8s} {'R2 +ent':>8s} {'R2 quad':>8s}")
    for b in l0["blocks"]:
        star = "  <-" if b["block"] in ("c0-c4", "c5-c24") else ""
        print(f"  {b['block']:9s} {_f(b['lambda_share'],9)} "
              f"{_f(b['within_vs_ensemble'],8,3)} {_f(b['share_of_all_within'],10)} "
              f"{_f(b['ceiling_at_anchors'],8)} {_f(b['R2 pi (linear)'],8)} "
              f"{_f(b['R2 pi + entropy'],8)} {_f(b['R2 pi + quadratic'],8)}{star}")
    print("  ceiling is the anchors-only between share, bounded in [0,1]. wtn/ens")
    print("  compares 20 anchors' within-variance to all 100 members' total, so > 1")
    print("  means the anchor groups are more dispersed than the ensemble there.")
    print("  LOCATION is read on c0-c4 and WIDTH on c5-c24. Those are different")
    print("  subspaces: only ~0.4% of within-mixture variance is in c0-c4.")

    print("\n-- per component, leading block only (axes past ~c10 are not identified) --")
    print(f"  {'j':>3s} {'lam share':>10s} {'R2 on pi':>9s} {'within/var':>11s} "
          f"{'resid/within':>13s}")
    for r in l0["per_component"]:
        print(f"  {r['j']:3d} {_f(r['lambda_share'],10)} {_f(r['R2_pi'],9)} "
              f"{_f(r['within_share'],11,5)} {_f(r['resid_over_within'],13,1)}")
    print("  resid/within >> 1 means the unexplained part is STRUCTURE, not noise.")

    pm = l0.get("permutation")
    if pm:
        print(f"\n-- stratified permutation null on c0-c4 R2, {pm['n_perm']} draws --")
        print(f"  observed {pm['observed']:.4f}   null mean {pm['null_mean']:.4f}   "
              f"null q95 {pm['null_q95']:.4f}   p = {pm['p_value']:.4f}")

    print("\n" + "-" * 96)
    print("LAYER 2 — the headline: recover pi from the code, L1 on the simplex")
    print("-" * 96)
    print(f"  null, best constant predictor (barycentre) {l2['null_barycentre']:.4f}")
    print(f"  null, marginal draw                        {l2['null_marginal']:.4f}"
          f"   (do NOT quote this one)")
    sel = l2.get("selected")
    if sel:
        print(f"  ceiling, leave-one-mixture-out on real members "
              f"{l2['ceiling_lomo_l1']:.4f}   skill {sel['skill']:.3f}"
              f"   (top_m={sel['top_m']}, lam={sel['lam']:.0e})")
    for nm in ("interior", "vertex"):
        d = l2.get("sealed_" + nm)
        if d:
            print(f"  sealed {nm:9s} n={d['n']}  projected {d['l1_projected']:.4f}"
                  f"   raw {d['l1_raw']:.4f}"
                  + ("   <- raw exceeds the 2.0 two-point maximum: the linear map "
                     "left the simplex" if d["l1_raw"] > 2.0 else ""))

    for c in cells:
        print("\n" + "-" * 96)
        print(f"CELL {c['name']}   dim={c['dim']}  cond_mode={c['cond_mode']}")
        print("-" * 96)
        print(f"  {'target':16s} {'kind':14s} {'prof c0-c4':>10s} {'prof c5-c24':>11s} "
              f"{'width':>7s} {'loc B':>8s} {'B exp':>7s} {'rt L1':>7s} "
              f"{'rt(mean)':>8s} {'nn d':>7s} {'uniq':>5s}")
        for t in c["targets"]:
            print(f"  {t['label']:16s} {t['kind']:14s} "
                  f"{_f(t.get('profile_c0-c4'),10,3)} {_f(t.get('profile_c5-c24'),11,3)} "
                  f"{_f(t.get('width_c5_c24'),7,3)} {_f(t.get('location_B_c0_c4'),8,3)} "
                  f"{_f(t.get('location_B_expected'),7,3)} {_f(t.get('roundtrip_l1'),7,3)} "
                  f"{_f(t.get('roundtrip_l1_of_mean'),8,3)} {_f(t.get('nn_dist_mean'),7,3)} "
                  f"{str(t.get('nn_distinct_rows','-')):>5s}")
        print("  width 1.0 = correct conditional spread.  loc B should sit near "
              "'B exp', NOT near 0:")
        print("  B ~ 0 at an anchor means the flow reproduced that group's mean "
              "exactly, i.e. memorisation.")
        print("  rt L1 is PER-SAMPLE and is the number comparable to the ceiling. "
              "rt(mean) is")
        print("  variance-reduced by ~sqrt(n) and can beat the ceiling for that "
              "reason alone.")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True,
                   help="Directory of cond_*.json payloads from "
                        "scripts/diag_conditioning.py.")
    p.add_argument("--k", nargs="*", type=int, default=None)
    p.add_argument("--n_perm", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top_m", nargs="+", type=int, default=[3, 5, 8, 16, 32])
    p.add_argument("--lam", nargs="+", type=float,
                   default=[1e-3, 1e-2, 1e-1, 1.0])
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_dir, "cond_*.json")))
    if not paths:
        raise SystemExit(f"No cond_*.json under {args.data_dir}.")

    results = []
    for path in paths:
        with open(path) as f:
            pay = json.load(f)
        if args.k and int(pay["k"]) not in args.k:
            continue
        l0 = layer0(pay, args.n_perm, args.seed)
        l2 = layer2(pay, args.top_m, args.lam)
        cells = [score_cell(pay, l0, l2, c) for c in pay.get("cells", [])]
        render(pay, l0, l2, cells)
        l0.pop("s2w", None)
        l2.pop("decoder", None)
        results.append({"run_name": pay.get("run_name"), "k": pay["k"],
                        "layer0": l0, "layer2": l2, "cells": cells})

    if args.json_out:
        d = os.path.dirname(os.path.abspath(args.json_out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n-> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

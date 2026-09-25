"""
report_conditioning.py recovers a KNOWN answer, and its controls fire.

Pure stdlib, like the script it tests, so this runs on the `cpu` partition and on a
laptop. tests/test_report.py enforces that purity; this file checks the arithmetic.

The subject is not formatting. Every statistic here replaced one that was wrong in a
way that read as a result: `recovery` is maximised by collapse, so a memorising flow
scored 130%, and its denominator was pooled across two nearly disjoint subspaces. A
replacement instrument is only worth anything if it FIRES on the failures the old one
missed, so the three generator controls at the bottom are the point of this file:

    G_collapse  emits each anchor group's mean exactly     -> must read as memorisation
    G_correct   emits the true conditional                 -> must read as correct
    G_marginal  ignores pi entirely                        -> must read as pi-ignored

A synthetic zoo is used rather than the real one because the answer has to be known
in advance. The construction mirrors the real zoo's shape -- 5 one-hot anchor groups
with replicates plus Dirichlet singletons, a leading block that is a deterministic
function of pi, and a tail that is pure within-mixture noise -- because that shape is
what makes the leading/tail split load-bearing.

    python tests/test_conditioning.py
"""

from __future__ import annotations

import math
import os
import random
import statistics
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "scripts"))
import report_conditioning as rc                                 # noqa: E402

FAILS: List[str] = []

P = 5           # domains
N_BRANCH = 4    # replicates per anchor -- the only source of variation at fixed pi
N_SINGLE = 40
K = 12          # components
LEAD_K = 5      # components that are a function of pi. MUST equal the row count
#                 of true_mean's weight matrix AND match
#                 report_conditioning.LEAD, which is c0-c4 INCLUSIVE. At 4 the
#                 leading block mixed four structured components with one
#                 pure-noise one and 'exactly linear' read R2 = 0.77.
SIG_LEAD = 0.02     # within-mixture sd in the leading block
SIG_TAIL = 1.00     # within-mixture sd in the tail


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def close(name: str, got: float, want: float, tol: float) -> None:
    check(name, abs(got - want) <= tol, f"got {got:.4f}, want {want:.4f} +/-{tol}")


# ---------------------------------------------------------------------------
# a synthetic zoo whose answer is known by construction
# ---------------------------------------------------------------------------

def build_pi(seed: int = 7) -> Tuple[List[List[float]], List[dict]]:
    rng = random.Random(seed)
    pis: List[List[float]] = []
    members: List[dict] = []
    row = 0
    for d in range(P):
        one = [1.0 if i == d else 0.0 for i in range(P)]
        for b in range(N_BRANCH):
            pis.append(one)
            members.append({"row": row, "idx": row, "pi": one, "kind": "anchor",
                            "mixture_id": f"anchor_{d}", "branch": b})
            row += 1
    for s in range(N_SINGLE):
        g = [rng.gammavariate(1.0, 1.0) for _ in range(P)]
        t = sum(g)
        pi = [x / t for x in g]
        pis.append(pi)
        members.append({"row": row, "idx": row, "pi": pi, "kind": "singleton",
                        "mixture_id": f"dirichlet_{s:03d}", "branch": 0})
        row += 1
    return pis, members


def true_mean(pi: List[float]) -> List[float]:
    """
    The conditional mean. LINEAR in pi on the leading block, zero on the tail.

    Deliberately exactly linear: then R2 on the leading block must come back ~1.0,
    and any shortfall is the estimator's fault rather than the data's. The real zoo
    is strongly NONLINEAR there (measured R2 0.568 linear, 0.948 quadratic), which is
    a finding about the zoo, not something a unit test should bake in.
    """
    w = [[3.0, -1.0, 0.5, 0.0, -2.0],
         [0.0, 2.5, -1.5, 1.0, 0.0],
         [-1.0, 0.0, 2.0, -0.5, 1.5],
         [0.5, 0.5, 0.5, -2.0, 0.5],
         [1.0, -1.5, 0.0, 1.5, -1.0]]
    assert len(w) == LEAD_K, f"w has {len(w)} rows but LEAD_K={LEAD_K}"
    return [sum(wj[d] * pi[d] for d in range(P)) for wj in w] + [0.0] * (K - len(w))


def build_payload(shuffle_pi: bool = False, seed: int = 11) -> dict:
    pis, members = build_pi()
    rng = random.Random(seed)
    n = len(pis)

    codes: List[List[float]] = []
    mus: List[List[float]] = []
    for i in range(n):
        mu = true_mean(pis[i])
        mus.append(mu)
        codes.append([mu[j] + rng.gauss(0.0, SIG_LEAD if j < LEAD_K else SIG_TAIL)
                      for j in range(K)])
    # Centre every column: the Gram PCA is mean-centred, so the real codes satisfy
    # sum_i x_ij = 0 exactly and several statistics assume it.
    for j in range(K):
        m = statistics.fmean(r[j] for r in codes)
        for r in codes:
            r[j] -= m
        for r in mus:
            r[j] -= m

    if shuffle_pi:
        # The negative control: a scrambled row->member->pi join. This is the failure
        # bundle.py:255-261 exists to prevent, and after the fact it is
        # indistinguishable from an honest null result -- so the instrument has to
        # separate them.
        order = list(range(n))
        rng.shuffle(order)
        for m_, src in zip(members, order):
            m_["pi"] = pis[src]

    hold = {"train": [m["idx"] for m in members], "interior": [], "vertex": []}
    targets = []
    seen: Dict[str, List[int]] = {}
    for m in members:
        seen.setdefault(str(m["mixture_id"]), []).append(int(m["row"]))
    for mid, rows in sorted(seen.items()):
        if len(rows) > 1:
            targets.append({"label": mid, "kind": "anchor",
                            "pi": members[rows[0]]["pi"], "member_rows": rows})
    return {"layout_version": 1, "k": K, "arch": "synthetic", "run_name": "synth",
            "n_members": n, "domains": list(rc.DOMAINS), "code_scale": 1.0,
            "eigenvalues": None, "spectrum": None,
            "fingerprints": {"pi": "synthetic", "code_stats": "synthetic"},
            "members": members, "holdout": hold, "codes": codes,
            "targets": targets, "cells": [],
            # Test-only. The TRUE conditional mean in centred coordinates, so a
            # "correct" generator can be centred on it rather than on the 4-member
            # empirical mean. Centring on the latter makes the generator and the
            # reference perfectly correlated, which cancels the 1/m term in B's
            # expectation and makes a correct flow indistinguishable from a
            # memorising one -- the exact confusion this file exists to detect.
            "_true_mu": mus}


# ---------------------------------------------------------------------------
# generators, as sufficient statistics -- exactly what diag_conditioning dumps
# ---------------------------------------------------------------------------

def make_cell(pay: dict, kind: str, n: int = 128, seed: int = 3) -> dict:
    rng = random.Random(seed)
    X = pay["codes"]
    tgts = []
    all_mu = [statistics.fmean(r[j] for r in X) for j in range(K)]
    all_var = [statistics.variance([r[j] for r in X]) for j in range(K)]
    for t in pay["targets"]:
        rows = t["member_rows"]
        gm = [statistics.fmean(X[r][j] for r in rows) for j in range(K)]
        mu_true = pay["_true_mu"][rows[0]]
        sig = [SIG_LEAD if j < LEAD_K else SIG_TAIL for j in range(K)]
        if kind == "collapse":
            # Emits each anchor group's empirical mean, exactly. This is what
            # memorising four members at one mixture looks like.
            mean, var = list(gm), [0.0] * K
        elif kind in ("correct", "wide"):
            f = 1.0 if kind == "correct" else 2.0
            s_ = [[mu_true[j] + rng.gauss(0.0, f * sig[j]) for j in range(K)]
                  for _ in range(n)]
            mean = [statistics.fmean(r[j] for r in s_) for j in range(K)]
            var = [statistics.variance([r[j] for r in s_]) for j in range(K)]
        elif kind == "marginal":
            mean, var = list(all_mu), list(all_var)
        else:
            raise ValueError(kind)
        raw = [[mean[j] + rng.gauss(0.0, math.sqrt(var[j])) for j in range(K)]
               for _ in range(8)]
        tgts.append({"label": t["label"], "kind": "anchor", "n": n,
                     "mean": mean, "var": var, "samples": raw})
    return {"name": f"G_{kind}", "dir": f"synthetic/{kind}", "dim": K,
            "cond_mode": "pi", "n_steps": 50, "n_samples": n, "targets": tgts}


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_linear_algebra() -> None:
    print("\n--- pure-stdlib linear algebra ---")
    A = [[4.0, 1.0], [1.0, 3.0]]
    B = [[1.0], [2.0]]
    x = rc._solve(A, B)
    check("solve: A x == B", all(
        abs(sum(A[i][j] * x[j][0] for j in range(2)) - B[i][0]) < 1e-12
        for i in range(2)))
    # OLS on an exactly-linear system must be exact.
    X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]
    Y = [[3.0], [5.0], [8.0], [1.0]]
    beta = rc._ridge(X, Y)
    close("ridge recovers beta[0]=3", beta[0][0], 3.0, 1e-5)
    close("ridge recovers beta[1]=5", beta[1][0], 5.0, 1e-5)


def test_proj_simplex() -> None:
    print("\n--- simplex projection (required: the raw map leaves the simplex) ---")
    for v in ([0.2] * 5, [1.0, 0.0, 0.0, 0.0, 0.0], [-0.524, 0.087, 1.137, -0.232, 0.081],
              [5.0, -3.0, 0.1, 0.0, 0.2]):
        q = rc.proj_simplex(v)
        check(f"sums to 1 and non-negative: {[round(x,2) for x in v]}",
              abs(sum(q) - 1.0) < 1e-9 and min(q) >= -1e-12,
              f"-> {[round(x,3) for x in q]}")
    check("already-in-simplex vector is unchanged",
          max(abs(a - b) for a, b in zip(rc.proj_simplex([0.2] * 5), [0.2] * 5)) < 1e-9)
    # The measured case from the real sealed vertex.
    q = rc.proj_simplex([-0.524, 0.087, 1.137, -0.232, 0.081])
    check("real sealed-vertex prediction projects onto the true one-hot",
          rc.l1(q, [0.0, 0.0, 1.0, 0.0, 0.0]) < 0.35,
          f"L1 = {rc.l1(q, [0.0, 0.0, 1.0, 0.0, 0.0]):.3f} vs raw "
          f"{rc.l1([-0.524, 0.087, 1.137, -0.232, 0.081], [0, 0, 1, 0, 0]):.3f}")


def test_layer0_known_answer() -> None:
    print("\n--- Layer 0 on a zoo whose answer is known ---")
    pay = build_payload()
    l0 = rc.layer0(pay, n_perm=100, seed=0)
    check("all 5 anchor groups found", l0["n_anchor_groups"] == 5,
          f"dof {l0['dof_within']}")
    check("within dof is 5*(4-1) = 15", l0["dof_within"] == 15)

    lead = [b for b in l0["blocks"] if b["block"] == "c0-c4"][0]
    tail = [b for b in l0["blocks"] if b["block"] == "c5-c9"][0]
    check("leading block R2 on pi ~ 1 (it is exactly linear by construction)",
          lead["R2 pi (linear)"] > 0.99, f"{lead['R2 pi (linear)']:.4f}")
    check("tail block R2 on pi ~ 0 (pure noise)",
          tail["R2 pi (linear)"] < 0.20, f"{tail['R2 pi (linear)']:.4f}")
    check("leading block within share is small",
          lead["within_vs_ensemble"] < 0.05, f"{lead['within_vs_ensemble']:.5f}")
    check("tail block within share ~ 1", tail["within_vs_ensemble"] > 0.7,
          f"{tail['within_vs_ensemble']:.4f}")
    check("ceiling is bounded in [0, 1]",
          all(0.0 <= b["ceiling_at_anchors"] <= 1.0 for b in l0["blocks"]),
          "an unbounded ceiling means two populations were mixed")
    check("jackknife SE is positive and smaller than rho",
          0.0 < l0.get("rho_jackknife_se", -1) < l0["rho_pooled"],
          f"rho {l0['rho_pooled']:.4f} +/- {l0.get('rho_jackknife_se'):.4f}")
    pm = l0["permutation"]
    check("stratified permutation null is far below the observed R2",
          pm["observed"] > pm["null_q95"] and pm["p_value"] < 0.05,
          f"obs {pm['observed']:.3f} q95 {pm['null_q95']:.3f} p {pm['p_value']:.3f}")


def test_scrambled_join_is_caught() -> None:
    print("\n--- negative control: a scrambled pi<->code join ---")
    good = rc.layer0(build_payload(shuffle_pi=False), n_perm=0, seed=0)
    bad = rc.layer0(build_payload(shuffle_pi=True), n_perm=100, seed=0)
    g = [b for b in good["blocks"] if b["block"] == "c0-c4"][0]["R2 pi (linear)"]
    b = [b for b in bad["blocks"] if b["block"] == "c0-c4"][0]["R2 pi (linear)"]
    check("shuffling the join collapses leading-block R2", b < 0.5 * g,
          f"{g:.4f} -> {b:.4f}")
    check("and its permutation p-value stops being significant",
          bad["permutation"]["p_value"] > 0.05,
          f"p = {bad['permutation']['p_value']:.3f}")

    l2g = rc.layer2(build_payload(False), [3, 5, 8], [1e-2, 1e-1, 1.0])
    l2b = rc.layer2(build_payload(True), [3, 5, 8], [1e-2, 1e-1, 1.0])
    check("and the decoder's skill collapses toward the null",
          l2b["selected"]["skill"] < 0.5 * l2g["selected"]["skill"],
          f"skill {l2g['selected']['skill']:.3f} -> {l2b['selected']['skill']:.3f}")


def test_layer2_decoder() -> None:
    print("\n--- Layer 2: recover pi from the code ---")
    pay = build_payload()
    l2 = rc.layer2(pay, [3, 5, 8], [1e-3, 1e-2, 1e-1, 1.0])
    p = len(rc.DOMAINS)
    want = statistics.fmean(rc.l1(m["pi"], [1.0 / p] * p) for m in pay["members"])
    close("barycentre null matches a direct computation",
          l2["null_barycentre"], want, 1e-9)
    check("marginal null is strictly worse than the barycentre null",
          l2["null_marginal"] > l2["null_barycentre"],
          f"{l2['null_marginal']:.4f} vs {l2['null_barycentre']:.4f}")
    check("leave-one-mixture-out ceiling beats the null",
          l2["selected"]["skill"] > 0.5,
          f"L1 {l2['ceiling_lomo_l1']:.4f} vs null {l2['null_barycentre']:.4f}")


def test_generator_controls() -> None:
    print("\n--- the point of the file: three generators, three verdicts ---")
    pay = build_payload()
    l0 = rc.layer0(pay, n_perm=0, seed=0)
    l2 = rc.layer2(pay, [3, 5, 8], [1e-2, 1e-1, 1.0])
    null = l2["null_barycentre"]

    got = {}
    for kind in ("collapse", "correct", "wide", "marginal"):
        cell = make_cell(pay, kind)
        sc = rc.score_cell(pay, l0, l2, cell)
        w = statistics.fmean(t["width_c5_c24"] for t in sc["targets"])
        b = statistics.fmean(t["location_B_c0_c4"] for t in sc["targets"])
        be = statistics.fmean(t["location_B_expected"] for t in sc["targets"])
        rt = statistics.fmean(t["roundtrip_l1"] for t in sc["targets"])
        got[kind] = (w, b, be, rt)
        print(f"    G_{kind:9s} width {w:7.3f}   loc B {b:9.4f} (expect {be:.4f})"
              f"   round-trip L1 {rt:.4f}")

    wc, bc, bec, rtc = got["collapse"]
    wk, bk, bek, rtk = got["correct"]
    ww, _, _, _ = got["wide"]
    wm, bm, bem, rtm = got["marginal"]

    check("G_correct: width ~ 1", 0.7 < wk < 1.4, f"{wk:.4f}")
    check("G_correct: loc B is CALIBRATED against its expectation",
          0.25 * bek < bk < 4.0 * bek, f"{bk:.4f} vs expected {bek:.4f}")
    check("G_correct: round-trip far below the null", rtk < 0.2 * null,
          f"{rtk:.4f} vs null {null:.4f}")

    check("G_wide: width ~ 2 — the width statistic is sensitive to over-dispersion",
          1.5 < ww < 2.6, f"{ww:.4f}")

    check("G_collapse: width ~ 0", wc < 0.05, f"{wc:.4f}")
    check("G_collapse: loc B far BELOW its expectation — the memorisation "
          "signature, and the OPPOSITE tail from a bad flow",
          bc < 0.2 * bek, f"{bc:.5f} vs expected {bek:.4f}")
    check("G_collapse: round-trip is still good, so no dispersion statistic can "
          "condemn it on its own", rtc < 0.5 * null, f"{rtc:.4f} vs {null:.4f}")

    check("G_marginal: round-trip at or beyond the null — pi was ignored",
          rtm > 0.6 * null, f"{rtm:.4f} vs null {null:.4f}")
    check("G_marginal: loc B far ABOVE its expectation", bm > 2.0 * bem,
          f"{bm:.4f} vs expected {bem:.4f}")
    # The reason the headline is the round-trip and not a width. Here the tail
    # carries no between-mixture variance, so a pi-ignoring generator and a correct
    # one have the SAME conditional width and the width statistic cannot separate
    # them at all. In the real zoo the tail carries a little (c5-c9 R2 = 0.18), so
    # the separation is weak rather than absent -- which is worse, because it looks
    # like a measurement.
    check("width does NOT separate G_marginal from G_correct (why width is not "
          "the headline)", abs(wm - wk) < 0.35, f"{wm:.4f} vs {wk:.4f}")
    check("the round-trip DOES separate them", rtm > 5.0 * rtk,
          f"{rtm:.4f} vs {rtk:.4f}")


def test_proper_scores() -> None:
    """
    The energy/variogram PAIR must order all four generators; RT must not.

    This is the test that justifies demoting the round-trip. RT is a mean
    per-sample distance to the target, E||f(X) - pi||_1, so by Jensen it is
    minimised by a POINT MASS at the optimum -- it is improper by construction and
    no amount of tuning changes that. Measured on the REAL Mini code matrix (k=50,
    leave-one-branch-out over the 20 anchor members, M=256): G_collapse scores
    RT 0.131 against G_correct's 0.157, i.e. a memoriser WINS by 16%. The synthetic
    zoo here reproduces the same sign.

    The energy score repairs it with the -1/2 E||X - X'|| diversity term, and is
    strictly proper. It is weak on over-dispersion, which is what the variogram
    score covers; the variogram in turn rewards collapse. Neither alone is enough.
    """
    print("\n--- proper scores: the pair must order all four ---")
    pay = build_payload()
    l0 = rc.layer0(pay, n_perm=0, seed=0)
    l2 = rc.layer2(pay, [3, 5, 8], [1e-2, 1e-1, 1.0])

    got = {}
    for kind in ("collapse", "correct", "wide", "marginal"):
        sc = rc.score_cell(pay, l0, l2, make_cell(pay, kind, n=64))
        ts = [t for t in sc["targets"] if t.get("energy_score") is not None]
        if not ts:
            check("proper scores were computed at all", False,
                  "no target carried energy_score")
            return
        got[kind] = {
            "es": statistics.fmean(t["energy_score"] for t in ts),
            "vs": statistics.fmean(t["variogram_score"] for t in ts),
            "sk": statistics.fmean(t["energy_skill"] for t in ts),
            "pit": statistics.fmean(t["pit_frac_extreme"] for t in ts),
            "rt": statistics.fmean(t["roundtrip_l1"] for t in ts),
        }
        g = got[kind]
        print(f"    G_{kind:9s} energy {g['es']:8.4f} (skill {g['sk']:+.3f})  "
              f"variogram {g['vs']:9.4f}  PIT-extreme {g['pit']:.3f}  "
              f"RT {g['rt']:.4f}")

    c, k_, w, m = (got["collapse"], got["correct"], got["wide"], got["marginal"])

    # The headline property: the correct conditional wins on the proper score.
    check("ENERGY: G_correct beats G_collapse", k_["es"] < c["es"],
          f"{k_['es']:.4f} vs {c['es']:.4f}")
    check("ENERGY: G_correct beats G_marginal", k_["es"] < m["es"],
          f"{k_['es']:.4f} vs {m['es']:.4f}")
    check("ENERGY: G_correct beats G_wide", k_["es"] < w["es"],
          f"{k_['es']:.4f} vs {w['es']:.4f}")
    check("ENERGY skill: G_correct > 0 (better than ignoring pi)", k_["sk"] > 0.0,
          f"{k_['sk']:+.4f}")
    check("ENERGY skill: G_marginal ~ 0 (it IS the marginal)", m["sk"] < 0.25,
          f"{m['sk']:+.4f}")

    # The variogram is what catches over-dispersion, which energy barely sees.
    check("VARIOGRAM: separates G_wide from G_correct more sharply than energy "
          "does", (w["vs"] / k_["vs"]) > (w["es"] / k_["es"]),
          f"variogram {w['vs'] / k_['vs']:.2f}x vs energy {w['es'] / k_['es']:.2f}x")

    # Calibration: a point mass puts the truth outside the ensemble every time.
    check("PIT: G_collapse is maximally mis-calibrated", c["pit"] > 0.9,
          f"{c['pit']:.3f} of ranks in the outer 10%")
    check("PIT: G_correct is roughly calibrated", k_["pit"] < 0.45,
          f"{k_['pit']:.3f} (uniform would be 0.10)")

    # And the reason this test exists at all.
    check("RT is IMPROPER: the memorising point mass BEATS the correct "
          "conditional on it", c["rt"] < k_["rt"],
          f"G_collapse {c['rt']:.4f} < G_correct {k_['rt']:.4f}")
    check("ENERGY is not fooled the same way", c["es"] > k_["es"],
          f"G_collapse {c['es']:.4f} > G_correct {k_['es']:.4f}")


def test_crps() -> None:
    """CRPS: the closed form, the properness that RT lacks, and the skill score."""
    print("\n--- CRPS ---")
    rng = random.Random(7)

    def crps_brute(S, y):
        M = len(S)
        a = statistics.fmean(abs(v - y) for v in S)
        tot = sum(abs(S[i] - S[j]) for i in range(M) for j in range(M))
        return a - 0.5 * tot / (M * M)

    # 1. the O(M log M) closed form equals the O(M^2) definition
    worst = 0.0
    for _ in range(200):
        M = rng.randint(1, 40)
        S = [rng.gauss(0, rng.uniform(0.1, 3.0)) for _ in range(M)]
        y = rng.gauss(0, 2)
        worst = max(worst, abs(rc.crps(S, y) - crps_brute(S, y)))
    check("crps closed form == double sum", worst < 1e-12, f"max diff {worst:.2e}")

    # 2. it is the 1-D energy score, same V-statistic convention
    worst = 0.0
    for _ in range(100):
        M = rng.randint(2, 25)
        S = [rng.gauss(0, 1.5) for _ in range(M)]
        y = rng.gauss(0, 1)
        worst = max(worst, abs(rc.crps(S, y)
                               - rc.energy_score([[v] for v in S], [y], beta=1.0)))
    check("crps == energy_score on 1-D", worst < 1e-12, f"max diff {worst:.2e}")

    # 3. PROPERNESS -- the whole reason this replaces the round-trip statistic.
    #    A point mass at the conditional mean must LOSE on CRPS and WIN on the
    #    mean per-sample distance. Both halves are asserted: the second is what
    #    made the old headline unfalsifiable.
    mu, sd, M, N = 1.3, 0.8, 256, 3000
    c_correct = c_collapse = rt_correct = rt_collapse = 0.0
    for _ in range(N):
        y = rng.gauss(mu, sd)
        draws = [rng.gauss(mu, sd) for _ in range(M)]
        c_correct += rc.crps(draws, y)
        c_collapse += rc.crps([mu] * M, y)
        rt_correct += statistics.fmean(abs(v - y) for v in draws)
        rt_collapse += abs(mu - y)
    c_correct /= N; c_collapse /= N; rt_correct /= N; rt_collapse /= N
    check("CRPS is proper: point mass LOSES", c_correct < c_collapse,
          f"correct {c_correct:.4f} vs collapse {c_collapse:.4f}")
    check("RT is IMPROPER: point mass WINS", rt_collapse < rt_correct,
          f"correct {rt_correct:.4f} vs collapse {rt_collapse:.4f} "
          f"-- this is why RT is not the headline")

    # 4. an over-dispersed conditional also loses
    c_wide = statistics.fmean(
        rc.crps([rng.gauss(mu, 2 * sd) for _ in range(M)], rng.gauss(mu, sd))
        for _ in range(N))
    check("CRPS catches over-dispersion", c_correct < c_wide,
          f"correct {c_correct:.4f} vs 2x-wide {c_wide:.4f}")

    # 5. crps_vector averages per coordinate, and skill() is sane
    S = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
    want = statistics.fmean([rc.crps([1.0, 2.0, 3.0], 2.0),
                             rc.crps([10.0, 20.0, 30.0], 25.0)])
    close("crps_vector averages coordinates", rc.crps_vector(S, [2.0, 25.0]),
          want, 1e-12)
    close("crps_vector honours dims=", rc.crps_vector(S, [2.0, 25.0], dims=[0]),
          rc.crps([1.0, 2.0, 3.0], 2.0), 1e-12)
    close("skill(x, x) == 0", rc.skill(0.4, 0.4), 0.0, 1e-12)
    close("skill halves to 0.5", rc.skill(0.2, 0.4), 0.5, 1e-12)
    check("skill(_, 0) is nan not inf", rc.skill(0.2, 0.0) != rc.skill(0.2, 0.0))
    check("crps of an empty sample is nan",
          rc.crps([], 1.0) != rc.crps([], 1.0))


def main() -> int:
    print("=" * 78)
    print("report_conditioning.py — known answers and controls")
    print("=" * 78)
    test_linear_algebra()
    test_proj_simplex()
    test_layer0_known_answer()
    test_scrambled_join_is_caught()
    test_layer2_decoder()
    test_generator_controls()
    test_proper_scores()
    test_crps()
    print("\n" + "=" * 78)
    if FAILS:
        print(f"FAILED ({len(FAILS)}):")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("All conditioning checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

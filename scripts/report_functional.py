#!/usr/bin/env python
"""
report_functional.py -- does the generated model BEHAVE like one trained on pi?

Retrieval baseline and dose-response, built as one instrument. Pure stdlib, so it
runs on a laptop over copied-down JSON, exactly like report_retrieval.py.

WHAT IT MEASURES, AND WHY NOT THE OTHER THING
---------------------------------------------
The retrieval report (report_retrieval.py) scores a model by ONE number,

    L(m, pi) = sum_d pi_d * log ppl_d(m),

the mixture-weighted mean log perplexity. That collapse makes the evaluation
unfalsifiable. L up-weights exactly the domains the nearest trained neighbour already
agrees on, so on every interior held-out mixture retrieval matches or beats the model
actually trained there, and no generative claim is available there at all.

Score the FULL five-domain profile instead -- v(m) = (log ppl_d(m))_d, the model's
capability vector -- and the picture inverts. Leave-one-mixture-out, mean |delta log
ppl| in nats, measured on the real zoo with no holdout design at all:

                                    Mini      Small
    ceiling (replicate noise)      0.0101     0.0109
    quadratic surface on pi        0.0664     0.0749
    retrieval (1-NN)               0.0970     0.1086
    linear surface on pi           0.0978     0.1154
    global mean                    0.1183     0.1221

Retrieval sits at ~10x the noise floor at BOTH scales. There is a factor-of-ten
dynamic range everywhere, not one resolvable cell at a corner. That is the target
this script scores a generated arm against.

Two things follow, and both are reported here.

1. The bar is the QUADRATIC SURFACE, not the Gaussian null. A closed-form ridge
   regression on pi, fitted in seconds with no generative model at all, reaches
   0.066 nats. Any generative result must be reported against
   that, decoded to weights. `gauss_codes` is not a serious baseline here.

2. Retrieval cannot be continuous. It has exactly 85 answers, forever: ask a
   5000-point lattice and 1.7% of requests get a distinct model. Along barycentre ->
   multilingual vertex it is a 5-step staircase whose 4 discontinuities carry the
   whole 0.868-nat swing. A conditional generator is smooth there or it is not
   conditional -- the dose-response slope, which is a CONTROL test and stays immune to
   the collapse trap, because a collapsed generator draws flat lines however
   flattering its dPPL.

AVERAGING CONVENTION
--------------------
Errors are averaged PER MEMBER (an anchor group of four contributes four times),
because a generated model is one draw and the thing being predicted is one model.
The per-MIXTURE average is also reported; on Mini retrieval reads 0.0970 per member
against 0.0595 per mixture, and quoting one for the other is a 1.6x error.

  python scripts/report_functional.py --data_dir reports/data/n100 --arms 0.15
  python scripts/report_functional.py --data_dir reports/data/n100 --arms 0.15 \
      --gen flow=reports/data/gen/gen_tr92_k46.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from report_conditioning import (_ridge, _predict, l1,  # noqa: E402
                                 crps as _crps, skill)

DOMAINS = ("web", "code", "math", "books", "multilingual")
Matrix = List[List[float]]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _read(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_zoo(ds_path: str, meta_path: str, label: str) -> Optional[dict]:
    ds, meta = _read(ds_path), _read(meta_path)
    if ds is None or meta is None:
        print(f"  [skip] {label}: need both {ds_path} and {meta_path}")
        return None
    domains = tuple(meta.get("domains") or DOMAINS)
    members = {int(m["idx"]): m for m in meta["members"]}
    rows = []
    for r in ds["rows"]:
        i = int(r["idx"])
        if i not in members:
            continue
        m = members[i]
        rows.append({"idx": i, "pi": [float(x) for x in m["pi"]],
                     "kind": m["kind"], "mixture_id": m["mixture_id"],
                     "v": [math.log(float(r["ppl"][d])) for d in domains]})
    return {"label": label, "domains": domains, "rows": rows,
            "beta": meta.get("beta"), "holdout": meta.get("holdout") or {}}


# ---------------------------------------------------------------------------
# the profile metric
# ---------------------------------------------------------------------------

UNITS = ("pct", "nats")


def err(a: Sequence[float], b: Sequence[float], unit: str = "nats") -> float:
    """
    Distance between two capability profiles. Mean rather than sum so it reads
    per-domain and does not change meaning if a sixth domain is ever added.

    Both arguments are ALWAYS log ppl (nats). Only the reported unit changes.

    ORDER MATTERS in "pct" and not in "nats": exp(a_d - b_d) = ppl_a / ppl_b, so the
    SECOND argument is the reference, the denominator, the thing being predicted.
    Every call site already passes (prediction, truth) in that order -- ladder passes
    the fitted surface then the real member, ceiling passes the member then its group
    mean, score_generated passes the generated model then the real one at that pi.


      "nats"  mean |delta log ppl|, additive and symmetric. This is the unit the
              retrieval headroom algebra needs -- L(m, pi) = sum_d pi_d ln ppl_d is
              additive in nats by construction, so sigma, headroom and rho are only
              meaningful there.
      "pct"   mean |ppl_gen/ppl_real - 1| x 100, the relative perplexity error.
              exp() is applied PER DOMAIN and then averaged, because the mean of the
              absolute log deviations is not the log of anything -- converting an
              aggregated nats figure with a single exp() would be a different
              statistic that happens to look similar (0.0314 nats -> 3.19% either
              way at this size, and they diverge as the errors grow).

    Reported as percent because a reader knows what "the generated model is 3.2%
    worse in perplexity" means, and does not know what 0.0314 nats means.
    """
    if unit == "pct":
        return statistics.fmean(
            100.0 * abs(math.exp(x - y) - 1.0) for x, y in zip(a, b))
    return statistics.fmean(abs(x - y) for x, y in zip(a, b))


def crps_profile(S: Sequence[Sequence[float]], y: Sequence[float],
                 unit: str = "nats") -> float:
    """
    Mean per-domain CRPS of a generated COHORT against the real model at that pi.

    This is the proper companion to the round-trip error above, and the reason the
    cohort is sampled at all. `err` scores one generated model against the target;
    averaging it over a cohort gives E|X - y|, which by Jensen is minimised by a
    POINT MASS at the conditional median. A generator that has memorised the
    conditional mean therefore WINS on round-trip dPPL -- measured, on the code
    matrix: a collapsed generator scores 0.131 against the correct conditional's
    0.157. CRPS subtracts the -1/2 E|X - X'| diversity term that repairs exactly
    this, and it needs only ONE real observation per mixture, which is all this zoo
    has at 80 of its 85 mixtures.

    Quote the two together or neither.
    """
    if not S:
        return float("nan")
    n = len(y)
    if unit == "pct":
        return statistics.fmean(
            _crps([100.0 * (math.exp(row[j] - y[j]) - 1.0) for row in S], 0.0)
            for j in range(n))
    return statistics.fmean(
        _crps([row[j] for row in S], float(y[j])) for j in range(n))


def feats(P: Sequence[Sequence[float]], quad: bool = True) -> Matrix:
    """Design matrix. No intercept: pi sums to 1 and already spans the constant."""
    out = []
    for p in P:
        f = list(p)
        if quad:
            f += [p[a] * p[b] for a in range(len(p)) for b in range(a, len(p))]
        out.append(f)
    return out


def fit_surface(rows: Sequence[dict], quad: bool, lam: float = 1e-6) -> Matrix:
    return _ridge(feats([r["pi"] for r in rows], quad), [r["v"] for r in rows], lam)


def nearest(pi: Sequence[float], pool: Sequence[dict]) -> Optional[dict]:
    return min(pool, key=lambda m: l1(pi, m["pi"])) if pool else None


def ceiling(rows: Sequence[dict], unit: str = "nats") -> Tuple[float, int]:
    """
    The noise floor: how far two models trained on the SAME mixture differ.

    This is the only thing that says whether any difference below is a measurement.
    Measured from the anchor replicates, which are the only source of variation at
    fixed pi -- and they sit at the five one-hot VERTICES, so homoscedasticity in pi
    is an assumption this cannot test.
    """
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        groups.setdefault(r["mixture_id"], []).append(r)
    devs, n = [], 0
    for g in groups.values():
        if len(g) < 2:
            continue
        mu = [statistics.fmean(m["v"][j] for m in g) for j in range(len(g[0]["v"]))]
        devs += [err(m["v"], mu, unit) for m in g]
        n += 1
    return (statistics.fmean(devs) if devs else float("nan")), n


def ladder(rows: Sequence[dict], hold: Optional[Sequence[int]] = None,
           unit: str = "nats") -> dict:
    """
    Leave-one-MIXTURE-out (or a fixed region holdout) over every baseline.

    Leave-one-ROW-out would leak: dropping one anchor branch leaves three siblings at
    the identical mixture, so retrieval would "predict" it at distance 0.
    """
    by_mix: Dict[str, List[dict]] = {}
    for r in rows:
        by_mix.setdefault(r["mixture_id"], []).append(r)

    if hold is not None:
        held = {r["mixture_id"] for r in rows if r["idx"] in set(hold)}
        blocks = [[r for r in rows if r["mixture_id"] in held]]
    else:
        blocks = [v for v in by_mix.values()]

    acc: Dict[str, List[float]] = {k: [] for k in
                                   ("quadratic", "linear", "retrieval", "mean")}
    per_mix: Dict[str, List[float]] = {k: [] for k in acc}
    for te in blocks:
        if not te:
            continue
        hm = {r["mixture_id"] for r in te}
        tr = [r for r in rows if r["mixture_id"] not in hm]
        if len(tr) < 8:
            continue
        bq, bl = fit_surface(tr, True), fit_surface(tr, False)
        Pq = _predict(feats([r["pi"] for r in te], True), bq)
        Pl = _predict(feats([r["pi"] for r in te], False), bl)
        gm = [statistics.fmean(r["v"][j] for r in tr) for j in range(len(te[0]["v"]))]
        this: Dict[str, List[float]] = {k: [] for k in acc}
        for i, r in enumerate(te):
            nn = nearest(r["pi"], tr)
            this["quadratic"].append(err(Pq[i], r["v"], unit))
            this["linear"].append(err(Pl[i], r["v"], unit))
            this["retrieval"].append(err(nn["v"], r["v"], unit))
            this["mean"].append(err(gm, r["v"], unit))
        for k, v in this.items():
            acc[k] += v
            per_mix[k].append(statistics.fmean(v))

    ceil_, ngrp = ceiling(rows, unit)
    out = {"ceiling": ceil_, "unit": unit, "n_anchor_groups": ngrp,
           "n_scored": len(acc["retrieval"]), "n_blocks": len(blocks)}
    for k in acc:
        out[k] = statistics.fmean(acc[k]) if acc[k] else float("nan")
        out[f"{k}_per_mixture"] = (statistics.fmean(per_mix[k]) if per_mix[k]
                                   else float("nan"))
    return out


# ---------------------------------------------------------------------------
# retrieval is a step function
# ---------------------------------------------------------------------------

def staircase(rows: Sequence[dict], n_lattice: int = 2000,
              seed: int = 0) -> dict:
    """
    How many DISTINCT models can lookup return, and how discontinuous is it?

    A pure-stdlib Dirichlet(1) draw via the gamma-normalisation identity, so this
    needs no numpy and reproduces on the laptop.
    """
    rnd = __import__("random").Random(seed)
    P = len(rows[0]["pi"])
    pool = rows
    hits = set()
    dists = []
    for _ in range(n_lattice):
        g = [rnd.gammavariate(1.0, 1.0) for _ in range(P)]
        s = sum(g) or 1.0
        pi = [x / s for x in g]
        nn = nearest(pi, pool)
        hits.add(nn["idx"])
        dists.append(l1(pi, nn["pi"]))
    return {"n_lattice": n_lattice, "n_distinct_retrieved": len(hits),
            "frac_distinct": len(hits) / n_lattice,
            "mean_l1_to_nearest": statistics.fmean(dists)}


def path_staircase(rows: Sequence[dict], a: Sequence[float], b: Sequence[float],
                   n: int, domain_idx: int) -> dict:
    """Walk a line and count retrieval's discontinuities along it."""
    seq, vs = [], []
    for i in range(n):
        t = i / (n - 1)
        pi = [(1 - t) * x + t * y for x, y in zip(a, b)]
        nn = nearest(pi, rows)
        seq.append(nn["idx"])
        vs.append(nn["v"][domain_idx])
    jumps = sum(1 for i in range(1, n) if seq[i] != seq[i - 1])
    tv = sum(abs(vs[i] - vs[i - 1]) for i in range(1, n))
    return {"n_requests": n, "n_distinct": len(set(seq)), "jumps": jumps,
            "total_variation": tv, "end_to_end": vs[-1] - vs[0],
            # The raw series, so the staircase can be DRAWN rather than described.
            # It is the one figure a generative claim gets for free and retrieval
            # cannot have at any N: lookup has exactly one answer per trained
            # mixture, forever.
            "t": [i / (n - 1) for i in range(n)], "vs": vs}


def dose_response(rows: Sequence[dict], domains: Sequence[str],
                  pi_max: Optional[Sequence[float]] = None) -> List[dict]:
    """
    Dose-response: requested weight on domain d vs measured log ppl on d. OLS slope.

    A control test, not a quality test. A collapsed generator draws FLAT lines
    however good its dPPL looks, which is the direct answer to the collapse trap.
    """
    out = []
    for j, d in enumerate(domains):
        # RANGE MATCHING. The real zoo contains the five one-hot VERTICES at
        # pi_d = 1, and a Dirichlet(1) probe set reaches only ~0.78. Those vertices
        # are high-leverage points in an OLS fit, so comparing a generated slope
        # measured on [0, 0.78] against a real slope measured on [0, 1] is not a
        # like-for-like comparison. Passing pi_max truncates the reference to the
        # generated arm's own support.
        #
        # Measured on Mini b=0.15: it moves individual domains a lot (web 0.53 ->
        # 0.70 of the real slope, books 0.19 -> 0.11) and the MEAN across five
        # domains not at all (flow 0.54 -> 0.54, quadratic 0.63 -> 0.65). Reported
        # because it is the kind of thing that ought to be checked, not because it
        # changed the answer.
        pairs = [(r["pi"][j], r["v"][j]) for r in rows
                 if pi_max is None or r["pi"][j] <= pi_max[j] + 1e-12]
        if len(pairs) < 3:
            out.append({"domain": d, "slope": float("nan"), "r": float("nan"),
                        "n": len(pairs)})
            continue
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        n = len(xs)
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        syy = sum((y - my) ** 2 for y in ys)
        slope = sxy / sxx if sxx > 0 else float("nan")
        r = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")
        out.append({"domain": d, "slope": slope, "r": r, "n": n})
    return out


# ---------------------------------------------------------------------------
# the generated arm
# ---------------------------------------------------------------------------

def score_generated(zoo: dict, gen_path: str, label: str) -> Optional[dict]:
    """
    Score generated models against the real member at the same pi.

    Matching is by exact requested mixture (L1 < 1e-6), the same join
    report_retrieval.py uses. A generated row whose pi has no real counterpart is
    counted but cannot be scored -- it has no ceiling.
    """
    g = _read(gen_path)
    if g is None:
        print(f"  [skip] --gen {label}: no file at {gen_path}")
        return None
    domains = tuple(g.get("domains") or zoo["domains"])
    if domains != zoo["domains"]:
        print(f"  [skip] --gen {label}: domains {domains} != zoo {zoo['domains']}")
        return None
    real = zoo["rows"]
    # The pool retrieval is allowed to draw from.
    #
    # MATCHED BLINDING. If this arm's flow was trained with an exclusion ball
    # (leave-one-mixture-out is that ball at radius ~0; the extrapolation sweep
    # widens it), retrieval has to be blinded to exactly the same ball or the
    # comparison is rigged in retrieval's favour -- it would be allowed to look up
    # the very neighbours the generator was forbidden to learn from. The ball is
    # carried in the gen JSON by eval_generated.py because it cannot be
    # reconstructed from the rows.
    excl = g.get("flow_exclusion") or {}
    centre, radius = excl.get("center_pi"), excl.get("exclude_within_l1")
    if centre is not None and radius is not None:
        pool = [r for r in real if l1(r["pi"], centre) > float(radius)]
        blinded = len(real) - len(pool)
        print(f"  [{label}] matched blinding: retrieval pool excludes "
              f"{blinded} member(s) within L1 {float(radius):g} of the target "
              f"({len(pool)} remain)")
    else:
        # No ball: fall back to the zoo's own train/holdout split. Using the whole
        # zoo would let lookup return the held-out member itself.
        train_ids = set(zoo["holdout"].get("train") or [r["idx"] for r in real])
        pool = [r for r in real if r["idx"] in train_ids]
    by_arm: Dict[str, List[dict]] = {}
    for r in g["rows"]:
        by_arm.setdefault(str(r.get("arm", label)), []).append(r)

    out = {"label": label, "code_rms_ratio": g.get("code_rms_ratio"),
           "code_rms_ratio_at_fixed_pi": g.get("code_rms_ratio_at_fixed_pi"),
           "real_within_over_pooled": g.get("real_within_over_pooled"),
           "cond_mode": g.get("cond_mode"), "arms": {}}
    for aname, grows in by_arm.items():
        per, unmatched = [], 0
        gp = []
        # MATCHED baselines: retrieval and the replicate ceiling computed on exactly
        # the cells this arm was scored on. Comparing a generated arm measured at 5
        # sealed mixtures against retrieval averaged over all 85 is not a comparison,
        # and it was how the first version of this script read.
        cells: Dict[Tuple[float, ...], dict] = {}
        for r in grows:
            v = [math.log(float(r["ppl"][d])) for d in domains]
            gp.append({"pi": [float(x) for x in r["pi"]], "v": v,
                       "idx": r.get("idx", 0), "kind": r.get("kind", "")})
            match = [m for m in real if l1(m["pi"], r["pi"]) < 1e-6]
            if not match:
                unmatched += 1
                continue
            mu = [statistics.fmean(m["v"][j] for m in match)
                  for j in range(len(domains))]
            per.append(err(v, mu))
            key = tuple(round(x, 9) for x in r["pi"])
            c = cells.setdefault(key, {"pi": list(r["pi"]), "gen": [], "gen_v": [],
                                       "mu": mu, "n_real": len(match)})
            c["gen"].append(err(v, mu))
            c["gen_v"].append(v)

        for key, c in cells.items():
            # retrieval: nearest TRAIN member at a different mixture
            cand = [m for m in pool if l1(m["pi"], c["pi"]) > 1e-6]
            nn = nearest(c["pi"], cand) if cand else None
            c["nn_l1"] = l1(c["pi"], nn["pi"]) if nn else float("nan")
            # ceiling at this cell: replicate scatter, if this mixture has any
            reps = [m for m in real if l1(m["pi"], c["pi"]) < 1e-6]
            for u in UNITS:
                c[f"gen_err_{u}"] = statistics.fmean(
                    err(v, c["mu"], u) for v in c["gen_v"])
                c[f"retrieval_{u}"] = (err(nn["v"], c["mu"], u) if nn
                                       else float("nan"))
                c[f"ceiling_{u}"] = (
                    statistics.fmean(err(m["v"], c["mu"], u) for m in reps)
                    if len(reps) > 1 else float("nan"))
                # PROPER score, and its reference. The reference predictive is the
                # zoo's own marginal -- every training member's profile, used as one
                # big sample that ignores pi entirely. That is the climatological
                # forecast, and the skill score against it is the number that says
                # whether conditioning bought anything at this cell.
                c[f"crps_{u}"] = crps_profile(c["gen_v"], c["mu"], u)
                c[f"crps_marginal_{u}"] = crps_profile(
                    [m["v"] for m in pool], c["mu"], u)
                c[f"crps_skill_{u}"] = skill(c[f"crps_{u}"],
                                             c[f"crps_marginal_{u}"])
            c["gen_err"] = c["gen_err_nats"]
            c["retrieval"] = c["retrieval_nats"]
            c["ceiling"] = c["ceiling_nats"]
            c["n_samples"] = len(c["gen_v"])

        arm = {"n": len(grows), "n_scored": len(per), "n_unmatched": unmatched,
               "n_cells": len(cells),
               "n_samples_per_cell": (min(len(c["gen_v"]) for c in cells.values())
                                      if cells else 0),
               "cells": sorted(cells.values(), key=lambda c: -c["gen_err"]),
               "dose_response": dose_response(gp, domains)}
        # The reference slope on the SAME support as this arm actually covered.
        if gp:
            pmax = [max(r["pi"][j] for r in gp) for j in range(len(domains))]
            arm["pi_max"] = pmax
            arm["dose_response_real_full"] = dose_response(real, domains)
            arm["dose_response_real_matched"] = dose_response(real, domains, pmax)
        for u in UNITS:
            def _m(key):
                vals = [c[key] for c in cells.values() if c[key] == c[key]]
                return statistics.fmean(vals) if vals else float("nan")
            arm[f"profile_err_{u}"] = _m(f"gen_err_{u}")
            arm[f"matched_retrieval_{u}"] = _m(f"retrieval_{u}")
            arm[f"matched_ceiling_{u}"] = _m(f"ceiling_{u}")
            arm[f"crps_{u}"] = _m(f"crps_{u}")
            arm[f"crps_marginal_{u}"] = _m(f"crps_marginal_{u}")
            arm[f"crps_skill_{u}"] = skill(arm[f"crps_{u}"],
                                           arm[f"crps_marginal_{u}"])
        # Back-compat aliases: nats, per MEMBER (every sample counts once), which is
        # what the previously published numbers were averaged over.
        arm["profile_err"] = statistics.fmean(per) if per else float("nan")
        arm["matched_retrieval"] = arm["matched_retrieval_nats"]
        arm["matched_ceiling"] = arm["matched_ceiling_nats"]
        out["arms"][aname] = arm
    return out


# ---------------------------------------------------------------------------

def _f(x, w=8, d=4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return " " * (w - 3) + "n/a"
    return f"{x:{w}.{d}f}"


def print_arm(zoo: dict, gens: Sequence[dict], args) -> dict:
    rows, domains = zoo["rows"], zoo["domains"]
    print("\n" + "=" * 78)
    print(f"FUNCTIONAL LADDER — {zoo['label']}  (beta={zoo['beta']}, "
          f"N={len(rows)})")
    print("=" * 78)

    lad = ladder(rows, unit="pct")
    lad_nats = ladder(rows, unit="nats")
    print(f"\nprofile error over the {len(domains)} domains, "
          f"leave-one-mixture-out")
    print(f"  headline unit is %dPPL = mean_d |ppl_pred/ppl_real - 1| x 100; "
          f"nats = mean_d |dlog ppl|")
    print(f"  {'reference':<28}{'%dPPL':>10}{'per mix':>10}"
          f"{'x ceil':>9}{'nats':>10}")
    c, cn = lad["ceiling"], lad_nats["ceiling"]
    print(f"  {'ceiling (replicate noise)':<28}{c:>9.2f}%{c:>9.2f}%"
          f"{1.0:>8.1f}x{_f(cn, 10)}")
    for k, nm in (("quadratic", "quadratic surface on pi"),
                  ("linear", "linear surface on pi"),
                  ("retrieval", "RETRIEVAL (1-NN)"),
                  ("mean", "global mean")):
        print(f"  {nm:<28}{lad[k]:>9.2f}%{lad[k + '_per_mixture']:>9.2f}%"
              f"{lad[k] / c:>8.1f}x{_f(lad_nats[k], 10)}")
    print(f"  [{lad['n_scored']} members over {lad['n_blocks']} mixtures; "
          f"ceiling from {lad['n_anchor_groups']} replicate groups]")

    reg = {}
    if args.holdout_region:
        held = [r["idx"] for r in rows
                if any(r["pi"][j] >= args.holdout_region[1]
                       for j in [domains.index(d)
                                 for d in args.holdout_region[0].split(",")])]
        if held:
            reg = ladder(rows, hold=held)
            print(f"\nregion holdout: {args.holdout_region[0]} >= "
                  f"{args.holdout_region[1]}  (n={reg['n_scored']} members)")
            for k, nm in (("quadratic", "quadratic"), ("retrieval", "RETRIEVAL"),
                          ("mean", "global mean")):
                print(f"  {nm:<28}{_f(reg[k], 12)}{'':>13}{reg[k] / c:>10.1f}x")

    st = staircase(rows, args.n_lattice, args.seed)
    print(f"\nretrieval is a step function")
    print(f"  {st['n_lattice']} lattice requests -> "
          f"{st['n_distinct_retrieved']} distinct models "
          f"({100 * st['frac_distinct']:.1f}% of requests)")
    print(f"  mean L1 from request to the model actually returned: "
          f"{st['mean_l1_to_nearest']:.3f}")
    paths = {}
    P = len(domains)
    bary = [1.0 / P] * P
    for j, d in enumerate(domains):
        v = [1.0 if i == j else 0.0 for i in range(P)]
        ps = path_staircase(rows, bary, v, args.n_path, j)
        paths[d] = ps
        print(f"  barycentre -> {d:<14} {ps['n_distinct']} plateaus, "
              f"{ps['jumps']} jumps, carrying {ps['total_variation']:.3f} nats "
              f"(end-to-end {ps['end_to_end']:+.3f})")

    dr = dose_response(rows, domains)
    print(f"\ndose-response in the real zoo (reference): "
          f"d log ppl_d / d pi_d")
    for row in dr:
        print(f"  {row['domain']:<16}{row['slope']:+8.3f} nats/unit   "
              f"r={row['r']:+.3f}")

    scored = []
    for gspec in gens:
        gs = score_generated(zoo, gspec["path"], gspec["label"])
        if gs is None:
            continue
        scored.append(gs)
        # The verdict is at_fixed_pi, NEVER the pooled ratio: under
        # pi-conditioning a correct flow reads ~within/pooled pooled (0.33 at Mini),
        # which is numerically what a genuinely collapsed generator reads.
        ap = gs.get("code_rms_ratio_at_fixed_pi")
        exp = gs.get("real_within_over_pooled")
        print(f"\nGENERATED — {gs['label']}  (cond_mode={gs['cond_mode']}, "
              f"at_pi={_f(ap, 6, 3)}, pooled={_f(gs['code_rms_ratio'], 6, 3)}"
              + (f", correct pooled ~{exp:.2f}" if exp else "") + ")")
        if ap is not None and ap == ap and ap < 0.5:
            print("  !! CONDITIONAL COLLAPSE: spread at a fixed "
                  "mixture is far below real within-mixture variation. Such a "
                  "generator scores dPPL ~ 0 while generating nothing. "
                  "Nothing below is a result.")
        for aname, a in gs["arms"].items():
            e, mr = a["profile_err_pct"], a["matched_retrieval_pct"]
            mc = a["matched_ceiling_pct"]
            print(f"  arm {aname:<14} round-trip {e:>7.2f}% dPPL   "
                  f"({e / c:>5.1f}x the zoo-wide ceiling {c:.2f}%, "
                  f"{a['profile_err_nats']:.4f} nats)")
            # The comparison that counts: retrieval restricted to the SAME cells.
            print(f"     matched on these {a['n_cells']} cells: "
                  f"retrieval {mr:>6.2f}%"
                  + (f", ceiling {mc:.2f}%" if mc == mc else "")
                  + (f"   -> generated is {mr / e:.2f}x better"
                     if e > 0 and mr == mr else "")
                  + ("   [BEATS retrieval]" if mr == mr and e < mr
                     else "   [LOSES TO retrieval]" if mr == mr else ""))
            # The PROPER score. Round-trip dPPL above is a point statistic and is
            # minimised by a point mass, so it must never be quoted alone.
            cr, cm = a["crps_pct"], a["crps_marginal_pct"]
            sk = a["crps_skill_pct"]
            if cr == cr:
                print(f"     CRPS {cr:>7.3f}%  vs marginal {cm:>7.3f}%   "
                      f"skill {sk:>+6.3f}"
                      f"   [{a['n_samples_per_cell']} samples/cell]"
                      + ("" if a["n_samples_per_cell"] >= 16
                         else "  << too few samples for a stable CRPS"))
            if a["matched_ceiling"] == a["matched_ceiling"]:
                print(f"     matched replicate ceiling on these cells: "
                      f"{_f(a['matched_ceiling'], 8)}")
            print(f"     [{a['n_scored']}/{a['n']} scored, "
                  f"{a['n_unmatched']} had no real model at that pi]")
            print(f"     dose-response: " + "  ".join(
                f"{r['domain'][:4]}={r['slope']:+.2f}" for r in a["dose_response"])
                  + (f"   ({a['n_cells']} distinct pi -- too few for a slope)"
                     if a["n_cells"] < 12 else ""))
            # Fraction of the real slope recovered, against BOTH references: the
            # full zoo (which contains the one-hot vertices) and the zoo truncated
            # to this arm's own support. Quoting only the first would leave a
            # leverage artifact unexamined.
            ref_f = a.get("dose_response_real_full") or []
            ref_m = a.get("dose_response_real_matched") or []
            if ref_f and ref_m and a["n_cells"] >= 12:
                def _frac(ref):
                    vals = [g["slope"] / r["slope"]
                            for g, r in zip(a["dose_response"], ref)
                            if r["slope"] == r["slope"] and abs(r["slope"]) > 1e-9
                            and g["slope"] == g["slope"]]
                    return statistics.fmean(vals) if vals else float("nan")
                print(f"     fraction of the real slope: "
                      f"{_frac(ref_f):.2f} vs the full zoo, "
                      f"{_frac(ref_m):.2f} range-matched to pi_d <= "
                      f"{max(a['pi_max']):.2f}")
            if args.cells:
                for cc in a["cells"][:args.cells]:
                    print(f"       pi={[round(x, 3) for x in cc['pi']]}  "
                          f"gen {cc['gen_err']:.4f}  retr {cc['retrieval']:.4f}  "
                          f"nn_l1 {cc['nn_l1']:.3f}  "
                          f"ceil {_f(cc['ceiling'], 7)}")

    return {"label": zoo["label"], "beta": zoo["beta"], "n_members": len(rows),
            "domains": list(domains),
            # BOTH units are sealed. The headline is %dPPL and the nats column is
            # what every previously published number was quoted in, so a reader can
            # check continuity without re-running anything.
            "ladder": lad, "ladder_nats": lad_nats,
            "region_holdout": reg, "staircase": st,
            "paths": paths, "dose_response": dr, "generated": scored}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact_dir", default=None)
    p.add_argument("--arch", default="gpt2_zoo_mini")
    p.add_argument("--arms", nargs="+", default=["0.15"],
                   help="beta tags to report")
    p.add_argument("--data_dir", default=None,
                   help="Laptop mode: read reports/data/n100/{ds,meta}_b<tag>.json")
    p.add_argument("--zoo_dir", default=None)
    p.add_argument("--gen", action="append", default=[],
                   help="LABEL=PATH to an eval_generated.py report; repeatable")
    p.add_argument("--holdout_region", nargs=2, default=None,
                   metavar=("DOMAINS", "THRESH"),
                   help="e.g. --holdout_region multilingual,math 0.45")
    p.add_argument("--n_lattice", type=int, default=2000)
    p.add_argument("--n_path", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cells", type=int, default=0,
                   help="Print the N worst per-cell rows for each generated arm.")
    p.add_argument("--json_out", default=None)
    args = p.parse_args()
    if args.holdout_region:
        args.holdout_region = (args.holdout_region[0],
                               float(args.holdout_region[1]))

    gens = []
    for spec in args.gen:
        if "=" not in spec:
            print(f"  [skip] --gen {spec!r}: expected LABEL=PATH")
            continue
        lab, path = spec.split("=", 1)
        gens.append({"label": lab, "path": path})

    out = []
    for tag in args.arms:
        t = f"b{round(float(tag) * 100):03d}"
        if args.data_dir:
            ds = os.path.join(args.data_dir, f"ds_{t}.json")
            meta = os.path.join(args.data_dir, f"meta_{t}.json")
        else:
            art = args.artifact_dir or os.environ.get("ARTIFACT_DIR", "./artifacts")
            zd = args.zoo_dir or os.path.join(art, f"zoo_{t}_mini_n100", args.arch)
            ds = os.path.join(zd, "domain_separation.json")
            meta = os.path.join(zd, "zoo_meta.json")
        zoo = load_zoo(ds, meta, f"beta={tag}")
        if zoo is None:
            continue
        out.append(print_arm(zoo, gens, args))

    if args.json_out and out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[report_functional] -> {args.json_out}")
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())

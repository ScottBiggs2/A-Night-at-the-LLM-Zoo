#!/usr/bin/env python
"""
test_functional.py -- the functional ladder, its units, and its proper score.

PURE STDLIB, and a bare script like every other test here: `python
tests/test_functional.py`. report_functional.py imports from report_conditioning.py
and both are stdlib by design, so this runs on a laptop with no cluster and no GPU.

WHAT IT PINS DOWN
  1. the %dPPL <-> nats conversion, including the argument ORDER that only matters
     in percent (exp(a - b) = ppl_a / ppl_b, so `b` is the denominator);
  2. that CRPS is proper where the round-trip error is not, in FUNCTION space --
     the same demonstration tests/test_conditioning.py makes in code space;
  3. the golden nats numbers on the real Mini zoo, so switching the headline unit
     cannot quietly change what is being measured. Those three numbers were the
     published ladder before the unit changed, and they must survive it.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))
import report_functional as rf                                   # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def close(name, got, want, tol):
    check(name, abs(got - want) <= tol, f"got {got:.6f}, want {want:.6f} +/-{tol:g}")


def test_units():
    print("\n--- units ---")
    # ppl 100 -> 103 is +3%; ppl 50 -> 48 is -4%. Mean absolute = 3.5%.
    pred = [math.log(103.0), math.log(48.0)]
    truth = [math.log(100.0), math.log(50.0)]
    close("pct is the mean absolute relative ppl error",
          rf.err(pred, truth, "pct"), 3.5, 1e-9)
    close("nats is the mean absolute log difference", rf.err(pred, truth, "nats"),
          statistics.fmean([abs(math.log(1.03)), abs(math.log(0.96))]), 1e-12)
    check("nats is symmetric in its arguments",
          abs(rf.err(pred, truth, "nats") - rf.err(truth, pred, "nats")) < 1e-12)
    check("pct is NOT symmetric (the second argument is the denominator)",
          abs(rf.err(pred, truth, "pct") - rf.err(truth, pred, "pct")) > 1e-6,
          f"{rf.err(pred, truth, 'pct'):.4f} vs {rf.err(truth, pred, 'pct'):.4f}")
    close("identical profiles score 0 in pct", rf.err(truth, truth, "pct"), 0.0, 1e-12)
    close("identical profiles score 0 in nats", rf.err(truth, truth, "nats"),
          0.0, 1e-12)
    # the aggregation trap: exp() of the averaged nats is NOT the averaged percent
    naive = 100.0 * (math.exp(rf.err(pred, truth, "nats")) - 1.0)
    check("exp(mean nats) != mean percent, so the conversion must be per domain",
          abs(naive - rf.err(pred, truth, "pct")) > 1e-3,
          f"naive {naive:.4f}% vs correct {rf.err(pred, truth, 'pct'):.4f}%")


def test_crps_profile():
    print("\n--- functional CRPS ---")
    truth = [math.log(100.0)]
    # a cohort centred on the truth beats a point mass at the truth's own value
    wide = [[math.log(100.0 * (1.0 + s))] for s in
            (-0.04, -0.02, 0.0, 0.02, 0.04)]
    point = [[math.log(100.0)]] * 5
    c_wide = rf.crps_profile(wide, truth, "pct")
    c_point = rf.crps_profile(point, truth, "pct")
    check("a point mass ON the truth still wins when the truth is exact",
          c_point < c_wide, f"point {c_point:.4f} vs spread {c_wide:.4f}")

    # the real case: the truth is a DRAW, not the mean. Now spread must win.
    import random
    rng = random.Random(11)
    mu, sd, M, N = math.log(100.0), 0.05, 64, 2000
    cs = cp = rt_s = rt_p = 0.0
    for _ in range(N):
        y = [rng.gauss(mu, sd)]
        S = [[rng.gauss(mu, sd)] for _ in range(M)]
        cs += rf.crps_profile(S, y, "nats")
        cp += rf.crps_profile([[mu]] * M, y, "nats")
        rt_s += statistics.fmean(rf.err(v, y, "nats") for v in S)
        rt_p += rf.err([mu], y, "nats")
    cs, cp, rt_s, rt_p = cs / N, cp / N, rt_s / N, rt_p / N
    check("CRPS is proper in function space: the point mass LOSES", cs < cp,
          f"correct {cs:.5f} vs collapse {cp:.5f}")
    check("round-trip dPPL is IMPROPER: the point mass WINS", rt_p < rt_s,
          f"correct {rt_s:.5f} vs collapse {rt_p:.5f} "
          f"-- quote CRPS beside it or neither")
    check("crps_profile of an empty cohort is nan",
          rf.crps_profile([], truth, "pct") != rf.crps_profile([], truth, "pct"))


def test_dose_response_range_matching():
    print("\n--- dose-response range matching ---")
    # y = -1 * pi on [0, 1], plus one leverage point at pi = 1
    rows = [{"pi": [x, 1 - x], "v": [-x, 0.0]} for x in
            (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0)]
    full = rf.dose_response(rows, ("a", "b"))
    close("slope is recovered exactly on a clean line", full[0]["slope"], -1.0, 1e-9)
    trunc = rf.dose_response(rows, ("a", "b"), pi_max=[0.5, 1.0])
    close("truncation drops the leverage point and keeps the slope",
          trunc[0]["slope"], -1.0, 1e-9)
    check("truncation reduced n", trunc[0]["n"] == 6 and full[0]["n"] == 7,
          f"{trunc[0]['n']} vs {full[0]['n']}")
    thin = rf.dose_response(rows, ("a", "b"), pi_max=[0.05, 1.0])
    check("too few surviving points gives nan, not a fabricated slope",
          thin[0]["slope"] != thin[0]["slope"], f"n={thin[0]['n']}")


def test_golden_ladder():
    """The published nats numbers must survive the change of headline unit."""
    print("\n--- golden ladder (real Mini zoo, beta=0.15) ---")
    d = os.path.join(ROOT, "reports", "data", "n100")
    ds, meta = os.path.join(d, "ds_b015.json"), os.path.join(d, "meta_b015.json")
    if not (os.path.exists(ds) and os.path.exists(meta)):
        print("  [skip] reports/data/n100 not present")
        return
    zoo = rf.load_zoo(ds, meta, "beta=0.15")
    lad_n = rf.ladder(zoo["rows"], unit="nats")
    lad_p = rf.ladder(zoo["rows"], unit="pct")
    close("ceiling (nats)", lad_n["ceiling"], 0.0101, 5e-4)
    close("retrieval (nats, per member)", lad_n["retrieval"], 0.0970, 5e-4)
    close("quadratic (nats, per member)", lad_n["quadratic"], 0.1038, 5e-4)
    close("retrieval (nats, per mixture)", lad_n["retrieval_per_mixture"],
          0.0595, 5e-4)
    check("the percent ladder preserves the ORDER of the nats ladder",
          [k for k in ("retrieval", "quadratic", "linear", "mean")
           if True] and all(
              (lad_n[a] < lad_n[b]) == (lad_p[a] < lad_p[b])
              for a in ("retrieval", "quadratic", "linear", "mean")
              for b in ("retrieval", "quadratic", "linear", "mean")))
    check("retrieval is the strongest baseline under BOTH conventions",
          lad_n["retrieval"] < lad_n["quadratic"]
          and lad_n["retrieval_per_mixture"] < lad_n["quadratic_per_mixture"],
          f"per member {lad_n['retrieval']:.4f} < {lad_n['quadratic']:.4f}; "
          f"per mixture {lad_n['retrieval_per_mixture']:.4f} < "
          f"{lad_n['quadratic_per_mixture']:.4f}")
    print(f"  ceiling {lad_p['ceiling']:.2f}%  retrieval {lad_p['retrieval']:.2f}%  "
          f"quadratic {lad_p['quadratic']:.2f}%")


def main() -> int:
    print("=" * 74)
    print("report_functional: units, proper score, golden ladder")
    print("=" * 74)
    test_units()
    test_crps_profile()
    test_dose_response_range_matching()
    test_golden_ladder()
    print("\n" + "=" * 74)
    if FAILS:
        print(f"FAILED ({len(FAILS)}):")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("All functional checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

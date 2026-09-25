#!/usr/bin/env python3
"""
Downstream multiple-choice benchmarks: originals against generated models.

Reads the `bench` blocks that scripts/eval_generated.py seals on every row and
answers one question -- does a generated model score like the real model at the
same mixture? -- in two currencies:

  ACCURACY       the number a reviewer recognises, and the weak one. It is
                 quantised at 1/n_questions and these models sit near chance on a
                 4-way suite, so a delta smaller than a few questions is not a
                 measurement. Printed WITH its quantum and
                 its binomial standard error, so the non-resolution is visible
                 rather than implied.

  GOLD LOG SCORE the mean log-probability the model assigns to the CORRECT
                 continuation, in nats/token. A proper scoring rule, continuous,
                 and the same questions for every model -- so the comparison can be
                 PAIRED, which is where the power is.

VERTICES AND INTERIOR ARE NEVER POOLED. A one-hot mixture is the most extreme point
the design contains and vertex cells run about 10x the interior's functional error;
a subset that over-samples them reads high for a reason that has nothing to do with
the generator. Stratum is decided structurally here -- pi is
one-hot or it is not -- so the trap is avoided by construction.

The paired statistic follows scripts/select_config.py:136: report the mean, its
standard error, t = mean/se, and the win count, and call |t| >= 2 separated. No
p-value, because the cell count here is small and a p-value would dress that up.

  python3 scripts/report_bench.py reports/data/bench/*.json
  python3 scripts/report_bench.py --reference member --json out.json in1.json in2.json

Pure stdlib on purpose, like merge_gen.py: it reads JSON that a GPU job already
wrote, so it must run on the laptop from copied-down files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from typing import Dict, List, Optional, Tuple

PI_TOL = 1e-6
VERTEX_TOL = 1e-6


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def pi_key(pi) -> Tuple[float, ...]:
    return tuple(round(float(x), 9) for x in pi)


def stratum_of(pi) -> str:
    """A vertex is a one-hot mixture. Everything else is interior."""
    return "vertex" if max(float(x) for x in pi) > 1.0 - VERTEX_TOL else "interior"


def load(paths: List[str]) -> Tuple[Dict[str, list], Dict[str, dict], List[str]]:
    """-> (rows by arm, bench meta by source file, suite names)."""
    by_arm: Dict[str, list] = {}
    metas: Dict[str, dict] = {}
    suites: List[str] = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        meta = d.get("bench")
        if not meta:
            print(f"  [skip] {os.path.basename(p)}: no bench block -- it was scored "
                  f"without --bench")
            continue
        metas[os.path.basename(p)] = meta
        for b in meta["benchmarks"]:
            if b not in suites:
                suites.append(b)
        n_rows = 0
        for r in d.get("rows", []):
            if "bench" not in r:
                continue
            by_arm.setdefault(r["arm"], []).append(r)
            n_rows += 1
        print(f"  [read] {os.path.basename(p):<38} arms="
              f"{sorted({r['arm'] for r in d.get('rows', []) if 'bench' in r})} "
              f"rows={n_rows}")
    return by_arm, metas, suites


def check_fingerprints(metas: Dict[str, dict], suites: List[str]) -> None:
    """
    Every file must have asked the same questions, or nothing below is paired.

    This is the check the fingerprint exists for. Two files scored with different
    --bench_seed or --bench_n_questions would join happily on pi and produce a
    difference that is entirely a difference of question sets.
    """
    bad = False
    for b in suites:
        seen: Dict[str, List[str]] = {}
        for name, m in metas.items():
            fp = (m.get("fingerprint") or {}).get(b)
            if fp is not None:
                seen.setdefault(fp, []).append(name)
        if len(seen) > 1:
            bad = True
            print(f"\n  !! {b}: {len(seen)} DIFFERENT question sets across inputs:")
            for fp, names in seen.items():
                print(f"       {fp}  {names}")
    if bad:
        raise SystemExit(
            "\nRefusing to report: the inputs did not see the same questions, so no "
            "paired comparison below would mean anything. Re-run the odd arm with "
            "the same --bench_seed / --bench_n_questions / --bench_n_shots.")


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def cells(rows: list, suite: str, field: str) -> Dict[Tuple[float, ...], float]:
    """
    Collapse an arm's samples to ONE number per mixture.

    A flow contributes n_samples models at a cell and an anchor contributes its 4
    real members; both are replicates of "what this arm produces here", so the cell
    mean is the unit that pairs against another arm's cell mean.
    """
    acc: Dict[Tuple[float, ...], List[float]] = {}
    for r in rows:
        b = r["bench"].get(suite)
        if b is None or b.get(field) is None:
            continue
        v = float(b[field])
        if v != v:
            continue
        acc.setdefault(pi_key(r["pi"]), []).append(v)
    return {k: statistics.fmean(v) for k, v in acc.items()}


def paired(a: Dict[Tuple[float, ...], float], b: Dict[Tuple[float, ...], float],
           keys: List[Tuple[float, ...]]) -> Optional[dict]:
    """Mean / se / t of (a - b) over the cells both arms cover. Repo convention:
    |t| >= 2 is separated, and the win count is reported beside it."""
    d = [a[k] - b[k] for k in keys if k in a and k in b]
    n = len(d)
    if n < 2:
        return None
    m = statistics.fmean(d)
    sd = statistics.stdev(d)
    se = sd / math.sqrt(n)
    return {"mean": m, "sd": sd, "se": se,
            "t": (m / se if se > 0 else float("nan")),
            "n_cells": n, "n_positive": sum(1 for x in d if x > 0)}


def binomial_se(p: float, n: int) -> float:
    if n <= 0 or p != p:
        return float("nan")
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="eval_generated.py / merge_gen.py reports carrying bench "
                         "blocks.")
    ap.add_argument("--reference", default="member",
                    help="The arm every other arm is paired against. Default "
                         "'member' -- the real zoo models at the target mixture.")
    ap.add_argument("--json", default=None, help="Also write the tables as JSON.")
    args = ap.parse_args()

    print("=" * 78)
    print("Downstream benchmarks: generated models against the originals")
    print("=" * 78)
    by_arm, metas, suites = load(args.inputs)
    if not by_arm:
        raise SystemExit("no rows with a bench block in any input.")
    check_fingerprints(metas, suites)

    m0 = next(iter(metas.values()))
    n_q = int(m0.get("n_questions") or 0)
    print(f"\nsuites     : {suites}")
    print(f"questions  : {n_q} per suite, seed {m0.get('seed')}, "
          f"n_shots {m0.get('n_shots')}, max_length {m0.get('max_length')}")
    print(f"quantum    : {1.0 / n_q:.5f} accuracy per question -- a delta smaller "
          f"than a few questions is NOT a measurement")

    ref = args.reference
    if ref not in by_arm:
        print(f"\n  !! reference arm {ref!r} absent; arms present: "
              f"{sorted(by_arm)}. Levels are still reported; no pairing.")

    out: dict = {"suites": suites, "bench": m0, "reference": ref, "levels": [],
                 "paired": []}

    # ---- levels ----------------------------------------------------------
    for suite in suites:
        print("\n" + "-" * 78)
        print(f"{suite}  LEVELS   (chance is 1/n_choices; these are 51-355M base "
              f"models)")
        print("-" * 78)
        print(f"  {'arm':<26}{'stratum':<10}{'cells':>6}{'models':>7}"
              f"{'acc':>9}{'+-bSE':>8}{'acc_norm':>10}{'gold_lp':>10}")
        for arm in sorted(by_arm):
            for stratum in ("vertex", "interior"):
                rows = [r for r in by_arm[arm]
                        if stratum_of(r["pi"]) == stratum and suite in r["bench"]]
                if not rows:
                    continue
                accs = [r["bench"][suite]["acc"] for r in rows]
                norms = [r["bench"][suite]["acc_norm"] for r in rows]
                gold = [r["bench"][suite]["gold_mean_logprob_mean"] for r in rows
                        if r["bench"][suite].get("gold_mean_logprob_mean") is not None]
                a = statistics.fmean(accs)
                rec = {"suite": suite, "arm": arm, "stratum": stratum,
                       "n_cells": len({pi_key(r["pi"]) for r in rows}),
                       "n_models": len(rows), "acc": a,
                       "acc_binomial_se": binomial_se(a, n_q),
                       "acc_norm": statistics.fmean(norms),
                       "gold_mean_logprob": (statistics.fmean(gold) if gold
                                             else float("nan"))}
                out["levels"].append(rec)
                print(f"  {arm:<26}{stratum:<10}{rec['n_cells']:>6}"
                      f"{rec['n_models']:>7}{rec['acc']:>9.4f}"
                      f"{rec['acc_binomial_se']:>8.4f}{rec['acc_norm']:>10.4f}"
                      f"{rec['gold_mean_logprob']:>10.4f}")

    if ref not in by_arm:
        return _finish(out, args)

    # ---- paired against the originals ------------------------------------
    for suite in suites:
        print("\n" + "-" * 78)
        print(f"{suite}  PAIRED vs {ref}   (positive = the arm scores HIGHER than "
              f"the real models)")
        print("-" * 78)
        for field, label, unit in (("acc", "accuracy", ""),
                                   ("gold_mean_logprob_mean", "gold log score",
                                    " nats/tok")):
            ref_cells = cells(by_arm[ref], suite, field)
            print(f"\n  {label}{unit}")
            print(f"    {'arm':<26}{'stratum':<10}{'delta':>10}{'se':>9}"
                  f"{'t':>7}{'win':>9}   verdict")
            for arm in sorted(by_arm):
                if arm == ref:
                    continue
                arm_cells = cells(by_arm[arm], suite, field)
                for stratum in ("vertex", "interior"):
                    keys = sorted(k for k in arm_cells
                                  if k in ref_cells and stratum_of(k) == stratum)
                    st = paired(arm_cells, ref_cells, keys)
                    if st is None:
                        continue
                    # Accuracy has a second, harder floor than its own sampling
                    # error: the 1/n quantum. A mean delta below one question is
                    # not a difference no matter what t says.
                    if field == "acc":
                        resolves = (abs(st["mean"]) > 2.0 / n_q
                                    and abs(st["t"]) >= 2.0)
                        why = ("" if abs(st["mean"]) > 2.0 / n_q
                               else f" (< 2 questions = {2.0 / n_q:.4f})")
                    else:
                        resolves, why = abs(st["t"]) >= 2.0, ""
                    verdict = "separated" if resolves else "DOES NOT RESOLVE" + why
                    out["paired"].append({"suite": suite, "field": field,
                                          "arm": arm, "stratum": stratum,
                                          "resolves": resolves, **st})
                    print(f"    {arm:<26}{stratum:<10}{st['mean']:>+10.4f}"
                          f"{st['se']:>9.4f}{st['t']:>7.2f}"
                          f"{st['n_positive']:>5}/{st['n_cells']:<3}   {verdict}")

    return _finish(out, args)


def _finish(out: dict, args) -> int:
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print(f"\nwrote {args.json}")
    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

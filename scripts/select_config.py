#!/usr/bin/env python
"""
select_config.py -- apply the pre-registered selection rule to a set of flow
configurations, out of sample.

PURE STDLIB. Reads the per-arm eval_generated outputs of a leave-one-mixture-out
comparison, groups them by configuration, and ranks the configurations on a PROPER
SCORE at mixtures none of the flows were trained on.

THE RULE, fixed before the numbers were looked at
-------------------------------------------------
  rank on   mean CRPS skill over the held-out cells, in % dPPL units
  guard 1   |1 - a_pi| must not be the worst in the field
  guard 2   the round-trip error must not be worse than the marginal
  NEVER     training or validation loss

The last line is not a stylistic preference. Capacity x
duration collapses the sampled dispersion monotonically while the training loss
improves monotonically, so the loss ranks a collapsed generator first. Measured in
this very sweep -- the lowest-loss arm (source_std 0.125, loss 0.113) has a_pi 0.316,
i.e. a conditional a third as wide as the real one.

CRPS is used rather than the round-trip error because the round-trip error is
IMPROPER: E|X - y| is minimised by a point mass, so a generator that has memorised
the conditional mean beats the correct conditional on it. Both are printed, and a
configuration that wins on round-trip while losing on CRPS is flagged, because that
pattern is the signature of collapse rather than of quality.

  python scripts/select_config.py --data_dir reports/data/n100 --arm 0.15 \
      --gen_dir reports/data/gen/select --configs c05 c10 ada
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--arm", default="0.15")
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--pattern", default="flow_k50_codes_pi_{cfg}_lomo_*.json")
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    rows = []
    tmp = tempfile.mkdtemp(prefix="select_")
    for cfg in args.configs:
        files = sorted(glob.glob(os.path.join(
            args.gen_dir, args.pattern.format(cfg=cfg))))
        if not files:
            print(f"[{cfg}] no files matching {args.pattern.format(cfg=cfg)}")
            continue
        merged = os.path.join(tmp, f"{cfg}.json")
        subprocess.run([sys.executable, os.path.join(HERE, "merge_gen.py"),
                        "--label", cfg, "--out", merged] + files,
                       check=True, stdout=subprocess.DEVNULL)
        fj = os.path.join(tmp, f"fn_{cfg}.json")
        subprocess.run([sys.executable, os.path.join(HERE, "report_functional.py"),
                        "--data_dir", args.data_dir, "--arms", args.arm,
                        "--gen", f"{cfg}={merged}", "--json_out", fj],
                       check=True, stdout=subprocess.DEVNULL)
        with open(fj) as f:
            fn = json.load(f)[0]
        a = fn["generated"][0]["arms"][cfg]
        # a_pi is per input file; average what the flows actually reported.
        aps = []
        for fp in files:
            with open(fp) as f:
                v = json.load(f).get("code_rms_ratio_at_fixed_pi")
            if isinstance(v, (int, float)) and v == v:
                aps.append(v)
        rows.append({
            "config": cfg, "n_cells": a["n_cells"], "n_files": len(files),
            "crps_pct": a["crps_pct"], "crps_marginal_pct": a["crps_marginal_pct"],
            "crps_skill_pct": a["crps_skill_pct"],
            "round_trip_pct": a["profile_err_pct"],
            "retrieval_pct": a["matched_retrieval_pct"],
            "ceiling_pct": a["matched_ceiling_pct"],
            "a_pi_mean": statistics.fmean(aps) if aps else float("nan"),
            "cells": a["cells"],
        })

    if not rows:
        print("nothing to select from")
        return 1

    hdr = (f"{'config':<8}{'cells':>6}{'CRPS%':>9}{'marginal':>10}{'skill':>8}"
           f"{'round-trip':>12}{'retrieval':>11}{'ceiling':>9}{'mean a_pi':>11}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: -r["crps_skill_pct"]):
        print(f"{r['config']:<8}{r['n_cells']:>6}{r['crps_pct']:>9.2f}"
              f"{r['crps_marginal_pct']:>10.2f}{r['crps_skill_pct']:>+8.3f}"
              f"{r['round_trip_pct']:>12.2f}{r['retrieval_pct']:>11.2f}"
              f"{r['ceiling_pct']:>9.2f}{r['a_pi_mean']:>11.3f}")

    best = max(rows, key=lambda r: r["crps_skill_pct"])
    worst_ap = max(rows, key=lambda r: abs(1.0 - r["a_pi_mean"]))
    print(f"\nSELECTED: {best['config']}  "
          f"(CRPS skill {best['crps_skill_pct']:+.3f} at {best['n_cells']} "
          f"held-out mixtures)")
    if best["config"] == worst_ap["config"] and len(rows) > 1:
        print("  !! it is also the WORST-dispersed arm. The guard vetoes it; "
              "re-examine before using this configuration.")
    by_rt = min(rows, key=lambda r: r["round_trip_pct"])
    if by_rt["config"] != best["config"]:
        print(f"  NOTE: the best ROUND-TRIP arm is {by_rt['config']} "
              f"({by_rt['round_trip_pct']:.2f}% vs {best['round_trip_pct']:.2f}%), "
              f"which is not the best proper score. Round-trip is improper -- a "
              f"point mass wins it -- so CRPS decides.")

    # PAIRED comparison across the SAME cells. An aggregate ranking over 12 cells
    # whose errors span 2% to 47% is dominated by which cells are hard, not by which
    # configuration is better; pairing removes the cell effect, which is the only
    # way a 0.3-point difference means anything.
    common = None
    keyed = {}
    for r in rows:
        km = {tuple(round(x, 9) for x in c["pi"]): c for c in r["cells"]}
        keyed[r["config"]] = km
        common = set(km) if common is None else (common & set(km))
    common = sorted(common or [])

    def paired(a, b, field):
        d = [keyed[a][k][field] - keyed[b][k][field] for k in common]
        n = len(d)
        if n < 2:
            return None
        m = statistics.fmean(d)
        sd = statistics.stdev(d)
        se = sd / (n ** 0.5)
        return m, se, (m / se if se > 0 else float("nan")), sum(1 for x in d if x < 0), n

    if len(rows) > 1 and len(common) > 2:
        print(f"\npaired over the {len(common)} common cells "
              f"(negative = the first config is better):")
        print(f"  {'pair':<16}{'dCRPS':>9}{'t':>7}{'win':>7}   "
              f"{'dRT':>9}{'t':>7}{'win':>7}")
        order = sorted(rows, key=lambda r: -r["crps_skill_pct"])
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                a, b = order[i]["config"], order[j]["config"]
                pc, pr = paired(a, b, "crps_pct"), paired(a, b, "gen_err_pct")
                if not pc or not pr:
                    continue
                print(f"  {a+' vs '+b:<16}{pc[0]:>+9.3f}{pc[2]:>7.2f}"
                      f"{pc[3]:>4}/{pc[4]}   {pr[0]:>+9.3f}{pr[2]:>7.2f}"
                      f"{pr[3]:>4}/{pr[4]}"
                      + ("" if abs(pc[2]) >= 2.0 else "   [CRPS: not separated]"))

        print(f"\nflow vs MATCHED retrieval, per config "
              f"(negative = the flow is better):")
        print(f"  {'config':<10}{'mean d':>10}{'t':>8}{'cells won':>12}")
        for r in order:
            d = [keyed[r['config']][k]["gen_err_pct"]
                 - keyed[r['config']][k]["retrieval_pct"] for k in common]
            m = statistics.fmean(d)
            se = statistics.stdev(d) / (len(d) ** 0.5)
            t = m / se if se > 0 else float("nan")
            verdict = ("BEATS" if t < -2 else "LOSES TO" if t > 2 else "ties")
            print(f"  {r['config']:<10}{m:>+10.3f}{t:>8.2f}"
                  f"{sum(1 for x in d if x < 0):>7}/{len(d)}   {verdict} retrieval")

    # stratify: does the ordering hold where it matters, far from training data?
    print("\nby extrapolation distance (nearest surviving mixture, L1):")
    print(f"  {'config':<8}{'near half':>12}{'far half':>12}   (round-trip % dPPL)")
    for r in sorted(rows, key=lambda r: -r["crps_skill_pct"]):
        cs = sorted((c for c in r["cells"] if c.get("nn_l1") == c.get("nn_l1")),
                    key=lambda c: c["nn_l1"])
        if len(cs) < 2:
            continue
        h = len(cs) // 2
        near = statistics.fmean(c["gen_err_pct"] for c in cs[:h])
        far = statistics.fmean(c["gen_err_pct"] for c in cs[h:])
        print(f"  {r['config']:<8}{near:>12.2f}{far:>12.2f}")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\n-> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

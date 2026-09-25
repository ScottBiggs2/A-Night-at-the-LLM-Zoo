#!/usr/bin/env python
"""
report_flow_arms.py -- the selection table for a flow sweep.

PURE STDLIB. Reads flow_meta.json from every arm of a sweep and prints them side by
side, so a configuration is chosen by looking at a table rather than at whichever
run happened to finish last.

WHAT IT REFUSES TO RANK ON
--------------------------
`best_loss`. Capacity x duration collapses the sampled
dispersion MONOTONICALLY while the training loss improves MONOTONICALLY --

    width        params   epochs   train loss   sample rms
    64,128,64      51k      500      1.846       1.039 (correct)
    256,512,256   362k     8000      0.651       0.183
    512,1024,512  1.23M    2000      0.650       0.158 (worst)

so ranking on the loss selects the most collapsed checkpoint on the curve. The loss
is printed here only so that it can be seen NOT tracking the thing we care about.

WHAT IT RANKS ON INSTEAD
------------------------
`a_pi` = code_rms_ratio_at_fixed_pi: the sampled spread at ONE mixture over that
mixture's own within-group spread. 1.0 is correct; below 1 the conditional is too
tight, which is the failure this sweep exists to fix, and above 1 it is too loose.
`|1 - a_pi|` is therefore the ordering, and it is two-sided on purpose -- on a real zoo over-fitting shows up as a_pi falling BELOW 1 while the
pooled ratio still reads healthy, so a one-sided rule would miss it.

This is a GUARD, not the verdict. No dispersion statistic can separate "learned
p(code|pi)" from "memorised the members at that pi" -- a mixture of four memorised
deltas has exactly the right conditional spread. The verdict is the proper score
(CRPS/energy) on a held-out mixture, which needs report_conditioning.py.

  python scripts/report_flow_arms.py --run_dir $ARTIFACT_DIR/runs/zoo_b015_mini_k99
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def read(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def fmt(v, w=8, d=4) -> str:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return " " * (w - 2) + "--"
    if v != v:
        return " " * (w - 3) + "nan"
    return f"{v:>{w}.{d}f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--pattern", default="flow_k*_codes_pi*")
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    arms = []
    for d in sorted(glob.glob(os.path.join(args.run_dir, args.pattern))):
        meta = read(os.path.join(d, "flow_meta.json"))
        if meta is None:
            continue
        t = meta.get("train") or {}
        net = meta.get("net") or {}
        rt = t.get("round_trip") or {}
        name = os.path.basename(d)
        arms.append({
            "arm": name,
            "cond_inject": net.get("cond_inject", "concat"),
            "hidden_dims": net.get("hidden_dims"),
            "source_std": (meta.get("flow") or {}).get("source_std"),
            "path_noise": (meta.get("flow") or {}).get("path_noise"),
            "a_pi": t.get("code_rms_ratio_at_fixed_pi"),
            "pooled_rms": t.get("code_rms_ratio"),
            "pi_corr": t.get("pi_code_distance_corr"),
            "best_loss": t.get("best_loss"),
            "rt_rel_l2": rt.get("rel_l2_space"),
            "n_rows": t.get("n_rows"),
            "n_excluded": t.get("n_excluded_rows"),
            "epochs_run": t.get("epochs_run"),
        })

    if not arms:
        print(f"no flow arms under {args.run_dir} matching {args.pattern!r}")
        return 1

    def dev(a):
        v = a["a_pi"]
        return abs(1.0 - v) if isinstance(v, (int, float)) and v == v else float("inf")

    print(f"\n{len(arms)} arm(s) under {args.run_dir}\n")
    hdr = (f"{'arm':<34} {'inject':<7} {'src':>5} {'pnoise':>7} "
           f"{'a_pi':>8} {'|1-a_pi|':>9} {'pooled':>8} {'pi_corr':>8} "
           f"{'loss':>8} {'rt_relL2':>9} {'rows':>5}")
    print(hdr)
    print("-" * len(hdr))
    for a in sorted(arms, key=dev):
        print(f"{a['arm']:<34} {str(a['cond_inject']):<7} "
              f"{fmt(a['source_std'], 5, 2)} {fmt(a['path_noise'], 7, 3)} "
              f"{fmt(a['a_pi'])} {fmt(dev(a), 9)} {fmt(a['pooled_rms'])} "
              f"{fmt(a['pi_corr'])} {fmt(a['best_loss'])} "
              f"{fmt(a['rt_rel_l2'], 9, 5)} {str(a['n_rows']):>5}")

    best = min(arms, key=dev)
    print(f"\nclosest to a_pi = 1: {best['arm']}  (a_pi {fmt(best['a_pi'], 6)})")
    print("This is the dispersion GUARD only. Confirm with a proper score on a "
          "held-out mixture before calling it the selected configuration.")

    # Does the loss disagree with the dispersion? If it does, say so loudly --
    # that disagreement is the loss/dispersion divergence this script exists for.
    scored = [a for a in arms if isinstance(a["best_loss"], (int, float))
              and dev(a) != float("inf")]
    if len(scored) > 2:
        by_loss = min(scored, key=lambda a: a["best_loss"])
        if by_loss["arm"] != best["arm"]:
            print(f"\nNOTE: the LOWEST-LOSS arm is {by_loss['arm']} "
                  f"(loss {by_loss['best_loss']:.4f}, a_pi {fmt(by_loss['a_pi'], 6)}), "
                  f"which is NOT the best-dispersed arm. Selecting on loss here "
                  f"would have picked it.")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(arms, f, indent=2)
        print(f"\n-> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

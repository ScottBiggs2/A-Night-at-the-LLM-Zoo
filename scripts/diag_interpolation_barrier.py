#!/usr/bin/env python
"""
diag_interpolation_barrier.py -- Frankle-style instability analysis on the zoo.

    python scripts/diag_interpolation_barrier.py --arch gpt2_zoo_mini \
        --zoo_dir $ARTIFACT_DIR/zoo_b015_mini_n100/gpt2_zoo_mini \
        --out $ARTIFACT_DIR/population/mini_b015 --n_pairs 20

WHAT IT MEASURES
----------------
For a pair of members (i, j), score the linear interpolant
w(alpha) = (1-alpha) w_i + alpha w_j at several alpha and report the BARRIER:

    barrier = max_alpha  loss(w(alpha))  -  [(1-alpha) loss(w_i) + alpha loss(w_j)]

Frankle, Dziugaite, Roy & Carbin (2020) define instability exactly this way:
branch one training run at a shared checkpoint, vary only the data order, and ask
whether the two endpoints are linearly connected. That is literally this zoo's
design -- one trunk at beta=0.15, then 100 branches -- so the protocol transfers
without reinterpretation.

READ THE RESULT IN BOTH DIRECTIONS, AND SAY SO
----------------------------------------------
A LOW barrier means the members share a linearly connected basin. That is the
result that licenses skipping weight-space canonicalization (Neyshabur, Sedghi & Zhang 2020 use the same check for models from a
shared init) -- permutation alignment buys nothing between models that are already
mode-connected.

It is ALSO the result a sceptic reads as "so they are all the same model."

Both readings are available from one number, so this script is not evidence of
diversity and must not be quoted as if it were. The diversity evidence is the
branch-delta cosine matrix (scripts/diag_zoo_geometry.py --dump_gram) and the
next-token disagreement (scripts/diag_predictions.py). Report all three together.

The comparison that makes the barrier informative is WITHIN-anchor versus
BETWEEN-anchor: same pi with different data order, against different pi. If the
between-anchor barrier exceeds the within-anchor one, the mixture axis is a real
obstruction in loss space and not only in weight space.

COST
----
n_pairs x (n_alpha - 2) forward evaluations of five domains, plus the endpoints,
which are cached across pairs since members are reused. At Mini a five-domain pass
is ~10 s, so 20 pairs at 5 interior alphas is ~20 min on one GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

import numpy as np
import torch

from llmzoo.data.mixtures import DOMAINS
from llmzoo.eval.domains import load_evalsets, perplexity
from llmzoo.models.registry import build_zoo_model
from llmzoo.models.weight_extractor import build_stack_spec, write_stack_to_model


def mean_nll(model, evalsets, device, batch_size):
    """Mean per-domain log-perplexity = mean cross-entropy in nats.

    Averaged in LOG space. Perplexities across these five domains span more than
    an order of magnitude, so an arithmetic mean of ppl would make the barrier a
    statement about whichever domain happens to be hardest.
    """
    per = {d: perplexity(model, evalsets[d], device, batch_size) for d in DOMAINS}
    return float(np.mean([math.log(per[d]) for d in DOMAINS])), per


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="gpt2_zoo_mini")
    p.add_argument("--zoo_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_pairs", type=int, default=20,
                   help="pairs per pool (within-anchor and between-anchor)")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--n_docs", type=int, default=64)
    p.add_argument("--n_ctx", type=int, default=1024)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--pair_seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--evalset_cache", default=None)
    args = p.parse_args()

    art = os.environ.get("ARTIFACT_DIR", "./artifacts")
    with open(os.path.join(args.zoo_dir, "zoo_meta.json")) as f:
        zmeta = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evalsets = load_evalsets(args.evalset_cache or art,
                             args.seed, args.n_docs, args.n_ctx)

    members = {m["idx"]: m for m in zmeta["members"]}
    have = [i for i in sorted(members)
            if os.path.exists(os.path.join(args.zoo_dir, f"w_{i}.npy"))]
    kind = {i: members[i]["kind"] for i in have}
    group = {i: members[i]["mixture_id"] for i in have}

    rng = random.Random(args.pair_seed)
    within, between = [], []
    anchors = [i for i in have if kind[i] == "anchor"]
    for a in range(len(anchors)):
        for b in range(a + 1, len(anchors)):
            i, j = anchors[a], anchors[b]
            (within if group[i] == group[j] else between).append((i, j))
    rng.shuffle(within)
    rng.shuffle(between)
    pairs = ([("within_anchor", *q) for q in within[:args.n_pairs]]
             + [("between_anchor", *q) for q in between[:args.n_pairs]])
    if not pairs:
        print("No anchor pairs; this zoo has no repeated mixtures.")
        return 2
    print(f"{len(pairs)} pairs: {min(len(within), args.n_pairs)} within-anchor, "
          f"{min(len(between), args.n_pairs)} between-anchor")

    model = build_zoo_model(args.arch, seed=0).to(device)
    _flat0, spec = build_stack_spec(model, args.arch,
                                    exclude_1d=zmeta["exclude_1d"],
                                    include_extra=zmeta["include_extra"])
    if _flat0.size != int(zmeta["n_params"]):
        raise RuntimeError(f"spec D={_flat0.size:,} != zoo D={int(zmeta['n_params']):,}")

    W = {}          # member idx -> flat float32, loaded on demand
    endpoint = {}   # member idx -> (mean log-ppl, per-domain)

    def flat(i):
        if i not in W:
            W[i] = np.load(os.path.join(args.zoo_dir, f"w_{i}.npy"))
        return W[i]

    def score(w):
        write_stack_to_model(w, model, args.arch, spec)
        return mean_nll(model, evalsets, device, args.batch_size)

    rows = []
    for n, (pool, i, j) in enumerate(pairs):
        for e in (i, j):
            if e not in endpoint:
                endpoint[e] = score(flat(e))
        li, lj = endpoint[i][0], endpoint[j][0]
        curve = []
        for a in args.alphas:
            if a <= 0.0:
                val, per = li, endpoint[i][1]
            elif a >= 1.0:
                val, per = lj, endpoint[j][1]
            else:
                val, per = score((1.0 - a) * flat(i) + a * flat(j))
            curve.append({"alpha": float(a), "mean_log_ppl": val,
                          "chord": (1.0 - a) * li + a * lj,
                          "excess": val - ((1.0 - a) * li + a * lj),
                          "ppl": per})
        # Maximise over the INTERIOR only. excess is identically 0 at alpha 0 and
        # 1 by construction, so including the endpoints clamps the reported
        # barrier at >= 0 -- and when the whole path runs BELOW the chord (which
        # is what these models actually do) every pair reports exactly +0.0000
        # and the real finding is invisible. The signed interior extremum is the
        # quantity Frankle et al. define and it is allowed to be negative.
        interior = [c for c in curve if 0.0 < c["alpha"] < 1.0] or curve
        worst = max(interior, key=lambda c: c["excess"])
        best = min(interior, key=lambda c: c["excess"])
        mid = min(curve, key=lambda c: abs(c["alpha"] - 0.5))
        rows.append({"pool": pool, "i": i, "j": j,
                     "mixture_i": group[i], "mixture_j": group[j],
                     "log_ppl_i": li, "log_ppl_j": lj,
                     "barrier": worst["excess"],
                     "barrier_at_alpha": worst["alpha"],
                     "max_dip_below_chord": best["excess"],
                     "dip_at_alpha": best["alpha"],
                     # A negative barrier says the path is under the chord. This
                     # says something stronger and independent: the midpoint beats
                     # the BETTER of the two endpoints, i.e. a model soup effect.
                     "midpoint_minus_best_endpoint":
                         mid["mean_log_ppl"] - min(li, lj),
                     "curve": curve})
        print(f"  [{n + 1:>2}/{len(pairs)}] {pool:<15} {i:>3}-{j:<3} "
              f"endpoints {li:.4f}/{lj:.4f}  barrier {worst['excess']:+.4f}  "
              f"dip {best['excess']:+.4f}  mid-best "
              f"{mid['mean_log_ppl'] - min(li, lj):+.4f} nats", flush=True)
        # Two flat Mini members are 412 MB and Medium's are 2.8 GB; keep only the
        # endpoints still needed rather than the whole zoo.
        for e in list(W):
            if not any(e in (q[1], q[2]) for q in pairs[n + 1:]):
                del W[e]

    summary = {}
    for pool in ("within_anchor", "between_anchor"):
        sel = [r for r in rows if r["pool"] == pool]
        v = [r["barrier"] for r in sel]
        dip = [r["max_dip_below_chord"] for r in sel]
        mid = [r["midpoint_minus_best_endpoint"] for r in sel]
        summary[pool] = {
            "n": len(v),
            "mean": float(np.mean(v)) if v else None,
            "median": float(np.median(v)) if v else None,
            "max": float(np.max(v)) if v else None,
            "dip_mean": float(np.mean(dip)) if dip else None,
            "dip_min": float(np.min(dip)) if dip else None,
            "midpoint_minus_best_endpoint_mean": float(np.mean(mid)) if mid else None,
            "n_midpoint_beats_both": int(sum(1 for x in mid if x < 0)),
        }

    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, "interpolation_barrier.json")
    with open(dest, "w") as f:
        json.dump({"zoo_dir": args.zoo_dir, "arch": args.arch,
                   "beta": zmeta.get("beta"), "alphas": args.alphas,
                   "n_docs": args.n_docs, "n_ctx": args.n_ctx,
                   "evalset_seed": args.seed, "pair_seed": args.pair_seed,
                   "units": "nats of mean-over-domain log perplexity",
                   "summary": summary, "rows": rows}, f, indent=2)

    print("\n" + "=" * 70)
    print(f"linear-interpolation barrier — {args.arch}  beta={zmeta.get('beta')}")
    print("=" * 70)
    for pool, s in summary.items():
        if s["n"]:
            print(f"  {pool:<16}n={s['n']:<4}barrier mean {s['mean']:+.4f}  "
                  f"max {s['max']:+.4f}   dip mean {s['dip_mean']:+.4f}")
            print(f"  {'':<16}midpoint - best endpoint "
                  f"{s['midpoint_minus_best_endpoint_mean']:+.4f} nats; "
                  f"midpoint beats BOTH endpoints in "
                  f"{s['n_midpoint_beats_both']}/{s['n']} pairs")
    print("  A LOW barrier licenses skipping canonicalization; it is NOT")
    print("  evidence of diversity. Quote it beside the cosine matrix and the")
    print("  next-token disagreement, never instead of them.")
    print("  A NEGATIVE barrier means the path runs below the chord; if")
    print("  midpoint-minus-best-endpoint is also negative that is a model-soup")
    print("  effect, which is a separate and stronger statement.")
    print("=" * 70)
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
diag_predictions.py -- function-space diversity: do the members DISAGREE?

    python scripts/diag_predictions.py --arch gpt2_zoo_mini \
        --zoo_dir $ARTIFACT_DIR/zoo_b015_mini_n100/gpt2_zoo_mini \
        --out $ARTIFACT_DIR/population/mini_b015

WHAT THIS IS FOR
----------------
Every member of this zoo descends from one trunk, so the obvious reviewer question
is whether 100 members are 100 models or one model with 100 roundings. Weight-space
distance alone cannot answer it: two networks can be far apart in weights and
identical as functions.

Prediction disagreement can. Fort, Hu & Lakshminarayanan (2019) use exactly this to
separate a genuine deep ensemble from one built along a single trajectory, and the
shared-trunk design here is that comparison's hard case. So: for every member,
record the argmax token at every position of the SAME held-out slices the gate
scored, then report the pairwise fraction of positions where two members disagree.

WHY IT IS A SEPARATE SCRIPT AND NOT A FLAG ON eval_domains.py
-------------------------------------------------------------
`eval_domains.py` is the domain-separation gate and it is chained `afterok` behind
live branch arrays. Editing a script while a dependency chain is queued is unsafe:
a queued job runs whatever is deployed at exec
time. This script imports the gate's own `load_evalsets` and its own write-back
path instead, so the tokens scored here are byte-identical to the tokens the gate
scored, with no risk of perturbing a queued gate.

`load_evalsets` REFUSES to build a slice. That is deliberate and it means this
script cannot run before the gate has. Good: the gate is the precondition anyway.

COST
----
Top-1 ids only, as uint16 -- the zoo vocab is GPT-2 BPE at 50257, which fits. At
n_docs=64, n_ctx=1024 that is 5 domains x 64 x 1023 positions = 327k per member,
so 100 members is 65 MB. The N x N disagreement matrix afterwards is pure NumPy
and runs on a laptop. Storing logits instead would be 50257x larger; storing only
the pairwise matrix would throw away every per-domain and per-position breakdown.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
import torch

from llmzoo.data.mixtures import DOMAINS
from llmzoo.eval.domains import load_evalsets, top1
from llmzoo.models.registry import build_zoo_model
from llmzoo.models.weight_extractor import build_stack_spec, write_stack_to_model


# `top1` lives in llmzoo.eval.domains so that generated models
# are scored through the SAME argmax path as these originals; the two sets of
# tokens are compared directly, so a second implementation would be a second
# reduction order.


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="gpt2_zoo_mini")
    p.add_argument("--zoo_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_docs", type=int, default=64)
    p.add_argument("--n_ctx", type=int, default=1024)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--evalset_cache", default=None,
                   help="defaults to $ARTIFACT_DIR, where eval_domains.py cached it")
    p.add_argument("--members", type=int, nargs="*", default=None)
    p.add_argument("--save_predictions", action="store_true",
                   help="also write the raw uint16 top-1 ids (65 MB at Mini "
                        "defaults). The disagreement matrix is derived from them "
                        "either way; keep them to re-slice per domain offline.")
    args = p.parse_args()

    art = os.environ.get("ARTIFACT_DIR", "./artifacts")
    with open(os.path.join(args.zoo_dir, "zoo_meta.json")) as f:
        zmeta = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evalsets = load_evalsets(args.evalset_cache or art, args.seed,
                             args.n_docs, args.n_ctx)
    # Positions per domain, so an offline consumer can slice the flat prediction
    # vector back into domains without re-deriving the block shapes.
    spans, off = {}, 0
    for d in DOMAINS:
        n = evalsets[d].shape[0] * (evalsets[d].shape[1] - 1)
        spans[d] = [off, off + n]
        off += n
    total = off

    model = build_zoo_model(args.arch, seed=0).to(device)
    _flat0, spec = build_stack_spec(model, args.arch,
                                    exclude_1d=zmeta["exclude_1d"],
                                    include_extra=zmeta["include_extra"])
    if _flat0.size != int(zmeta["n_params"]):
        raise RuntimeError(
            f"spec gives D={_flat0.size:,} but the zoo was flattened at "
            f"D={int(zmeta['n_params']):,}.")

    idxs = args.members if args.members is not None else list(
        range(int(zmeta["n_members"])))
    members = {m["idx"]: m for m in zmeta["members"]}
    kept, preds = [], []
    for i in idxs:
        wp = os.path.join(args.zoo_dir, f"w_{i}.npy")
        if not os.path.exists(wp):
            print(f"  member {i}: MISSING {wp}, skipping")
            continue
        write_stack_to_model(np.load(wp), model, args.arch, spec)
        preds.append(np.concatenate([top1(model, evalsets[d], device,
                                          args.batch_size) for d in DOMAINS]))
        kept.append(i)
        print(f"  member {i:>3} {members[i]['mixture_id']:<20} "
              f"{len(preds[-1]):,} positions", flush=True)

    P = np.stack(preds)                      # (N, total) uint16
    N = len(kept)
    if P.shape[1] != total:
        raise RuntimeError(f"position count {P.shape[1]} != expected {total}")

    # Pairwise disagreement, overall and per domain. 100 x 327k uint16 is 65 MB,
    # so the naive double loop is 4950 comparisons over 327k bytes -- seconds.
    dis = np.zeros((N, N), dtype=np.float64)
    per_domain = {d: np.zeros((N, N), dtype=np.float64) for d in DOMAINS}
    for a, b in itertools.combinations(range(N), 2):
        ne = P[a] != P[b]
        dis[a, b] = dis[b, a] = ne.mean()
        for d, (s, e) in spans.items():
            v = ne[s:e].mean()
            per_domain[d][a, b] = per_domain[d][b, a] = v

    kind = {i: members[i]["kind"] for i in kept}
    group = {i: members[i]["mixture_id"] for i in kept}
    pools = {"within_anchor": [], "between_anchor": [],
             "singleton_singleton": [], "anchor_singleton": []}
    for a, b in itertools.combinations(range(N), 2):
        ia, ib = kept[a], kept[b]
        ka, kb = kind[ia], kind[ib]
        if ka == "anchor" and kb == "anchor":
            key = "within_anchor" if group[ia] == group[ib] else "between_anchor"
        elif ka == "singleton" and kb == "singleton":
            key = "singleton_singleton"
        else:
            key = "anchor_singleton"
        pools[key].append(dis[a, b])

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "disagreement.npy"), dis)
    if args.save_predictions:
        np.save(os.path.join(args.out, "top1_predictions.npy"), P)

    summary = {
        "zoo_dir": args.zoo_dir, "arch": args.arch, "beta": zmeta.get("beta"),
        "n_members": N, "member_idx": [int(i) for i in kept],
        "n_positions": int(total), "domain_spans": spans,
        "n_docs": args.n_docs, "n_ctx": args.n_ctx, "evalset_seed": args.seed,
        "disagreement_pools": {k: (float(np.mean(v)) if v else None)
                               for k, v in pools.items()},
        "n_pairs": {k: len(v) for k, v in pools.items()},
        "disagreement_mean": float(dis[np.triu_indices(N, 1)].mean()),
        "disagreement_min": float(dis[np.triu_indices(N, 1)].min()),
        "disagreement_max": float(dis[np.triu_indices(N, 1)].max()),
        "per_domain_mean": {d: float(m[np.triu_indices(N, 1)].mean())
                            for d, m in per_domain.items()},
        # A member vs itself is 0 by construction; the useful floor is how much
        # two members of the SAME anchor mixture disagree -- that is the
        # data-order-only difference, and every other pool is read against it.
        "note": "within_anchor is the same-pi, different-data-order floor.",
    }
    with open(os.path.join(args.out, "predictions.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"next-token disagreement — {args.arch}  N={N}  "
          f"{total:,} positions over {len(DOMAINS)} domains")
    print("=" * 70)
    for k, v in summary["disagreement_pools"].items():
        if v is not None:
            print(f"  {k:<22}{v:>8.4f}   ({summary['n_pairs'][k]} pairs)")
    print(f"  {'all pairs':<22}{summary['disagreement_mean']:>8.4f}   "
          f"(min {summary['disagreement_min']:.4f} "
          f"max {summary['disagreement_max']:.4f})")
    print("=" * 70)
    print(f"-> {os.path.join(args.out, 'predictions.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

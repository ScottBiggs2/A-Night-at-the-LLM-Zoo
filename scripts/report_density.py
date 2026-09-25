#!/usr/bin/env python
"""
report_density.py -- p(code | pi) and the divergences that follow from it.

This is the arm that needs the flow's own LIKELIHOOD rather than its samples. Three
questions, all of which reduce to a KL:

1. HELD-OUT CONDITIONAL NLL.  -E_real[log p_theta(x|pi)] is a cross-entropy, so

       -E[log p_theta(x|pi)]  =  H(p_real(.|pi))  +  KL(p_real || p_theta).

   H is unknown but is a property of the ZOO, not of any generator, so it is the
   same constant for every arm. Differences in held-out NLL between arms are
   therefore differences in KL, exactly. That is what makes this the one divergence
   we can estimate honestly with one real observation per mixture.

2. THE CONDITIONING GAP = a lower bound on I(x; pi).  Running the identical
   architecture with the discrete (family) encoder makes the flow unconditional on a
   single-architecture zoo, so

       E[log p_pi(x|pi)] - E[log p_0(x)]  <=  I(x; pi),

   and it is an equality when both flows are exact. This is the ablation the whole
   paper rests on, and it is exact at fixed codes, fixed capacity, fixed budget.

3. IDENTIFICATION.  The matrix log p(x_i | pi_j) over all candidate mixtures. The
   RANK of the true mixture is calibration-free -- it does not care that a flow
   trained for 20000 epochs is over-confident -- and the InfoNCE bound on the same
   matrix is valid because a flow likelihood is normalised, which a Gaussian
   surrogate's is not (fitted on this zoo the surrogate reports -15,335 nats).

Plus two sample-based divergences that need no density and so check the above:
k-NN KL (Perez-Cruz) between real and generated codes, and a classifier two-sample
test on the JOINT (x, pi) -- the joint matches if and only if the conditionals do,
and unlike a per-mixture divergence it is estimable when 80 of 85 mixtures are seen
exactly once.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

from llmzoo.artifacts.bundle import load_run
from llmzoo.artifacts.io import atomic_write_json
from llmzoo.gen.density import assert_consistent, identification, log_prob, logp_matrix
from llmzoo.gen.flow import load_flow


def knn_kl(P: np.ndarray, Q: np.ndarray, k: int = 1) -> float:
    """
    Perez-Cruz k-NN estimator of KL(P || Q) from samples.

    KL = d/n * sum log(s_k(i)/r_k(i)) + log(m/(n-1)), with r the k-th NN distance
    within P (self excluded) and s the k-th NN distance into Q. Consistent; badly
    biased in high dimension, which is why it is a cross-check on the flow
    likelihood and never the headline.
    """
    n, d = P.shape
    m = Q.shape[0]
    dpp = np.linalg.norm(P[:, None] - P[None], axis=2)
    np.fill_diagonal(dpp, np.inf)
    dpq = np.linalg.norm(P[:, None] - Q[None], axis=2)
    r = np.sort(dpp, axis=1)[:, k - 1]
    s = np.sort(dpq, axis=1)[:, k - 1]
    ok = (r > 0) & (s > 0)
    return float(d / ok.sum() * np.sum(np.log(s[ok] / r[ok])) + math.log(m / (n - 1)))


def c2st(A: np.ndarray, B: np.ndarray, seed: int = 0, folds: int = 5) -> float:
    """
    Classifier two-sample test accuracy, logistic regression by plain gradient
    descent so this needs no sklearn. 0.5 = indistinguishable.

    Run on the JOINT (x, pi): if the conditionals match, so does the joint.
    """
    rng = np.random.default_rng(seed)
    X = np.vstack([A, B])
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    accs = []
    for f in range(folds):
        te = np.zeros(len(X), bool)
        te[f::folds] = True
        Xtr, ytr, Xte, yte = X[~te], y[~te], X[te], y[te]
        w = np.zeros(Xtr.shape[1])
        b = 0.0
        for _ in range(400):
            p = 1.0 / (1.0 + np.exp(-(Xtr @ w + b)))
            g = Xtr.T @ (p - ytr) / len(Xtr)
            w -= 0.5 * (g + 1e-3 * w)
            b -= 0.5 * float((p - ytr).mean())
        pred = (Xte @ w + b) > 0
        accs.append(float((pred == (yte > 0.5)).mean()))
    return float(np.mean(accs))


def three_sample_test(gen: np.ndarray, train: np.ndarray,
                      test: np.ndarray) -> dict:
    """
    Meehan-Chaudhuri-Dasgupta data-copying test.

    The intuition the test formalises: a generator that MEMORISES sits closer to the
    training set than genuinely fresh data of the same distribution does. So compare
    two distance distributions --

        L(gen, train)   nearest-neighbour distance from each generated point to the
                        training set,
        L(test, train)  the same for HELD-OUT REAL members,

    -- with a two-sample Mann-Whitney U, standardised to Z_U.

        Z_U << 0  generated points are abnormally CLOSE to training data: copying.
        Z_U ~  0  generated points sit at the same distance real held-out data does.
        Z_U >> 0  generated points are abnormally FAR: underfitting.

    This is the test that needs a third sample, and it is why holding members out
    mattered before any generator existed. Note it is run in code space: the decode
    is linear, so a copy in weight space is a copy in code space, and the leading
    block is where a copy would have to live (c0-c4 carries 84% of the trace).

    Reported two-sided; the copying alarm is the LEFT tail alone.
    """
    def nn(A, B):
        d = np.linalg.norm(A[:, None] - B[None], axis=2)
        return d.min(axis=1)

    lg, lt = nn(gen, train), nn(test, train)
    n, m = len(lg), len(lt)
    allv = np.concatenate([lg, lt])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks over ties
    i = 0
    sv = allv[order]
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    R = ranks[:n].sum()
    U = R - n * (n + 1) / 2.0
    mu = n * m / 2.0
    sd = math.sqrt(n * m * (n + m + 1) / 12.0)
    z = (U - mu) / sd if sd > 0 else float("nan")
    return {"n_gen": n, "n_test": m,
            "mean_nn_gen_to_train": float(lg.mean()),
            "mean_nn_test_to_train": float(lt.mean()),
            "ratio": float(lg.mean() / max(lt.mean(), 1e-12)),
            "Z_U": z,
            "verdict": ("DATA COPYING" if z < -3 else
                        "underfitting" if z > 3 else "no copying detected")}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact_dir", default=None)
    p.add_argument("--run_name", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--flow_dir", required=True, help="the pi-conditioned flow")
    p.add_argument("--null_flow_dir", default=None,
                   help="the family-mode flow: the exact ablation at fixed codes, "
                        "capacity and budget.")
    p.add_argument("--zoo_dir", required=True)
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--n_gen", type=int, default=4000,
                   help="generated codes for the sample-based divergences")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--held_idx", type=int, nargs="*", default=None,
                   help="Zoo indices of HELD-OUT real members -- the third sample "
                        "the data-copying test needs. Their weights are read from "
                        "--zoo_dir and projected into this basis. Default: the "
                        "zoo_meta holdout (interior + vertex).")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    art = args.artifact_dir or os.environ.get("ARTIFACT_DIR", "./artifacts")
    run_root = os.path.join(art, "runs", args.run_name)
    with open(os.path.join(args.zoo_dir, "zoo_meta.json")) as f:
        zmeta = json.load(f)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The dataset is needed only to project held-out members into this basis:
    # DualGramPCA stores no explicit basis, so a projection is a pass over the
    # ensemble, exactly like a decode.
    bundle = load_run(run_root, args.k, want=("dataset", "pca", "codes"),
                      device=str(dev))
    arch = bundle.arch_list[0]
    pca = bundle.pcas[arch]
    fidx_val = bundle.family_idx(arch)
    codes = np.asarray(bundle.codes)[:, :args.k]

    sel = bundle.ens_meta.get("member_idxs")
    basis_rows = list(range(len(codes))) if sel is None else [int(i) for i in sel]
    zmem = {int(m["idx"]): m for m in zmeta["members"]}
    PI = np.array([zmem[i]["pi"] for i in basis_rows], dtype=np.float64)

    fm = load_flow(os.path.join(run_root, args.flow_dir),
                   code_stats=bundle.code_stats, device=str(dev))
    X = fm.space.to_flow(torch.tensor(codes, dtype=torch.float32, device=dev),
                         torch.full((len(codes),), fidx_val, dtype=torch.long,
                                    device=dev))
    fam = torch.full((len(codes),), fidx_val, dtype=torch.long, device=dev)
    pit = torch.tensor(PI, dtype=torch.float32, device=dev)

    out = {"run_name": args.run_name, "k": args.k, "n_members": len(codes),
           "n_steps": args.n_steps, "flow_dir": args.flow_dir}

    # --- 0. is the density trustworthy at this step count? ---
    cons = assert_consistent(fm.flow, X[:8], fam[:8], pi=pit[:8],
                             n_steps=args.n_steps)
    out["consistency"] = cons
    print(f"[density] trace-vs-map consistency: relative {cons['relative']:.5f} "
          f"({'OK' if cons['ok'] else 'TOO FEW STEPS'})")

    # --- 1. conditional NLL under the pi flow ---
    lp, _ = log_prob(fm.flow, X, fam, n_steps=args.n_steps, pi=pit)
    out["nll_pi_per_dim"] = float(-lp.mean() / args.k)
    print(f"[density] conditional NLL  {out['nll_pi_per_dim']:+.4f} nats/dim")

    # --- 2. the ablation: the SAME architecture with a constant encoder ---
    if args.null_flow_dir:
        fn = load_flow(os.path.join(run_root, args.null_flow_dir),
                       code_stats=bundle.code_stats, device=str(dev))
        lp0, _ = log_prob(fn.flow, X, fam, n_steps=args.n_steps)
        out["nll_uncond_per_dim"] = float(-lp0.mean() / args.k)
        gap = float((lp - lp0).mean())
        out["mi_lower_bound_nats"] = gap
        out["mi_lower_bound_per_dim"] = gap / args.k
        out["null_cond_mode"] = fn.cond_mode
        print(f"[density] unconditional NLL {out['nll_uncond_per_dim']:+.4f} "
              f"nats/dim  (cond_mode={fn.cond_mode})")
        print(f"[density] I(x;pi) lower bound = {gap:+.2f} nats "
              f"({gap / args.k:+.4f}/dim)")

    # --- 3. identification over every candidate mixture ---
    uniq, first = [], []
    for r, q in enumerate(PI):
        j = next((t for t, u in enumerate(uniq)
                  if float(np.abs(u - q).sum()) < 1e-9), None)
        if j is None:
            uniq.append(q)
            j = len(uniq) - 1
        first.append(j)
    U = torch.tensor(np.array(uniq), dtype=torch.float32, device=dev)
    M = logp_matrix(fm.flow, X, fam, U, n_steps=args.n_steps)
    ident = identification(M, first)
    out["identification"] = ident
    print(f"[density] identification over {ident['n_pi']} candidate mixtures: "
          f"top-1 {100 * ident['top1']:.1f}% (chance {100 * ident['chance_top1']:.1f}%)"
          f"  median rank {ident['median_rank']:.1f}/{ident['n_pi']} "
          f"(chance {ident['chance_median_rank']:.1f})")
    print(f"[density] InfoNCE I(x;pi) >= {ident['infonce_nats']:.3f} nats "
          f"of a possible {ident['max_infonce']:.3f}")

    # --- 4. sample-based divergences, as a cross-check on the above ---
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    reps = max(1, args.n_gen // len(codes))
    Xg, PIg = [], []
    with torch.no_grad():
        for _ in range(reps):
            x0 = torch.randn(len(codes), args.k, generator=g).to(dev) * fm.flow.source_std
            Xg.append(fm.flow.sample(len(codes), fam, n_steps=args.n_steps,
                                     x0=x0, pi=pit).cpu().numpy())
            PIg.append(PI)
    Xg = np.concatenate(Xg)
    PIg = np.concatenate(PIg)
    Xr = X.cpu().numpy()

    lead = min(8, args.k)
    out["knn_kl_real_to_gen_lead8"] = knn_kl(Xr[:, :lead], Xg[:, :lead])
    out["c2st_joint_lead8"] = c2st(np.c_[Xr[:, :lead], PI], np.c_[Xg[:, :lead], PIg],
                                   seed=args.seed)
    out["c2st_codes_only_lead8"] = c2st(Xr[:, :lead], Xg[:, :lead], seed=args.seed)
    out["n_gen"] = len(Xg)
    print(f"[density] k-NN KL(real || gen) on c0-c7 = "
          f"{out['knn_kl_real_to_gen_lead8']:+.3f} nats")
    print(f"[density] C2ST joint (x,pi) = {out['c2st_joint_lead8']:.3f}   "
          f"codes only = {out['c2st_codes_only_lead8']:.3f}   (0.5 = indistinguishable)")

    # --- 5. memorisation: the three-sample test ---
    # The third sample is real members the FLOW never trained on. There are two ways
    # to obtain one, and they are not equally good.
    #
    # PREFERRED -- the flow's own exclusion ball (train.excluded_rows). Those members
    # are still BASIS rows, so their codes are exact and the test measures only what
    # it claims to.
    #
    # LEGACY -- members held out of the BASIS as well, whose codes must be projected
    # in from a fit that does not span them. That projection discards 40-64% of a
    # held-out member's squared displacement, which inflates its distance to
    # everything and therefore biases Z_U TOWARD declaring copying. The verdict
    # survived it, but the raw gen->train vs test->train ratio did not and must not
    # be quoted from such a run. This path is kept only so old runs still report.
    excl = sorted(int(i) for i in (fm.meta.get("train", {}) or {})
                  .get("excluded_rows", []) or [])
    excl = [i for i in excl if 0 <= i < len(Xr)]
    if excl:
        keep = [i for i in range(len(Xr)) if i not in set(excl)]
        ht = Xr[excl]
        Xr_train = Xr[keep]
        print(f"[density] third sample: {len(excl)} member(s) the flow was blinded "
              f"to, IN BASIS (no projection loss); train sample is the remaining "
              f"{len(keep)}")
        lead = min(5, args.k)
        out["three_sample_source"] = "flow_exclusion"
        out["three_sample_n_excluded"] = len(excl)
        out["three_sample_lead5"] = three_sample_test(
            Xg[:, :lead], Xr_train[:, :lead], ht[:, :lead])
        out["three_sample_full"] = three_sample_test(Xg, Xr_train, ht)
        t5 = out["three_sample_lead5"]
        print(f"[density] three-sample (c0-c4): Z_U = {t5['Z_U']:+.2f}  "
              f"gen->train {t5['mean_nn_gen_to_train']:.4f} vs "
              f"heldout->train {t5['mean_nn_test_to_train']:.4f}  "
              f"-> {t5['verdict']}")
        held_idx = []
    else:
        ho = zmeta.get("holdout") or {}
        held_idx = (args.held_idx if args.held_idx is not None else
                    sorted(set(ho.get("interior", [])) | set(ho.get("vertex", []))))
        held_idx = [i for i in held_idx if i not in set(basis_rows)]

    if excl:
        pass
    elif held_idx:
        rows = []
        for i in held_idx:
            wp = os.path.join(args.zoo_dir, f"w_{i}.npy")
            if not os.path.exists(wp):
                print(f"  [held] missing {wp}, skipping")
                continue
            rows.append(pca.transform_vector(np.load(wp), bundle.dataset, arch,
                                             k=args.k))
        held = np.asarray(rows, dtype=np.float64)[:, :args.k]
        print(f"[density] projected {len(held)} held-out members into the "
              f"{len(basis_rows)}-member basis for the third sample")
        ht = fm.space.to_flow(
            torch.tensor(held, dtype=torch.float32, device=dev),
            torch.full((len(held),), fidx_val, dtype=torch.long,
                       device=dev)).cpu().numpy()
        lead = min(5, args.k)
        out["three_sample_source"] = "basis_holdout_projected"
        out["three_sample_lead5"] = three_sample_test(
            Xg[:, :lead], Xr[:, :lead], ht[:, :lead])
        out["three_sample_full"] = three_sample_test(Xg, Xr, ht)
        t5 = out["three_sample_lead5"]
        print(f"[density] three-sample (c0-c4): Z_U = {t5['Z_U']:+.2f}  "
              f"gen->train {t5['mean_nn_gen_to_train']:.4f} vs "
              f"heldout->train {t5['mean_nn_test_to_train']:.4f}  "
              f"-> {t5['verdict']}")
        print("  NOTE: the third sample was projected into a basis that does not "
              "span it. The verdict is conservative (the bias runs toward crying "
              "copying) but the raw distance RATIO above is not quotable.")
    else:
        # Without a third sample the honest fallback is the distance ratio between
        # generated-to-train and train-to-train, which cannot distinguish a correct
        # generator from a copier as cleanly.
        dtt = np.linalg.norm(Xr[:, None] - Xr[None], axis=2)
        np.fill_diagonal(dtt, np.inf)
        dgt = np.linalg.norm(Xg[:, None] - Xr[None], axis=2)
        out["nn_gen_to_train"] = float(dgt.min(axis=1).mean())
        out["nn_train_to_train"] = float(dtt.min(axis=1).mean())
        out["nn_ratio"] = out["nn_gen_to_train"] / max(out["nn_train_to_train"], 1e-12)
        out["nn_distinct_targets"] = int(len(set(dgt.argmin(axis=1).tolist())))
        print(f"[density] NN gen->train {out['nn_gen_to_train']:.4f} vs "
              f"train->train {out['nn_train_to_train']:.4f} "
              f"(ratio {out['nn_ratio']:.3f}); "
              f"{out['nn_distinct_targets']}/{len(Xr)} training members are somebody's "
              f"nearest neighbour")

    atomic_write_json(args.out, out)
    print(f"\n[density] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

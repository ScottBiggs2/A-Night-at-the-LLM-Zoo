#!/usr/bin/env python
"""
make_flow_manifest.py -- enumerate the arms of a flow sweep as one TSV line each.

PURE STDLIB, like report_stack.py: it reads the zoo's own meta JSON and writes a
manifest that slurm/flow_arm.sbatch indexes with $SLURM_ARRAY_TASK_ID. Nothing here
needs a GPU, a cluster, or torch.

WHY A MANIFEST RATHER THAN A NESTED sbatch LOOP
-----------------------------------------------
Three designs share one runner, and all three are just "train the same flow with a
different set of training rows or a different noise setting":

  sweep   the noise/capacity arms          (the conditional is
          measured too NARROW -- a_pi ~ 0.46 at sealed mixtures against a correct
          value of 1 -- and path_noise is the knob that widens it, currently 0.0)
  lomo    leave-one-mixture-out, 85 arms   (one flow per held-out mixture)
  radius  the extrapolation-distance sweep (push the nearest mixture the flow ever
          saw AWAY from the target on purpose)

The manifest also carries the x-axis each design is plotted against -- for lomo and
radius that is `nn_l1`, the L1 distance from the target to the nearest mixture that
SURVIVES the exclusion. train_flow.py recomputes and prints it; having it here means
the design can be read, and rejected, before any of it is submitted.

THE EXCLUSION IS ON THE FLOW'S TRAINING ROWS ONLY. The PCA basis is untouched, so k
is fixed across every arm and the codes are literally the same numbers in every one.
That is what makes these arms comparable, and it is why this replaces adding new zoo
members: the holdout is a property of the design, not of the data.

  python scripts/make_flow_manifest.py --design lomo \
      --meta reports/data/n100/meta_b015.json --out reports/data/manifests/lomo_b015.tsv
"""
from __future__ import annotations

import argparse
import json
import os


def l1(a, b) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def load_mixtures(meta_path: str):
    """[{mixture_id, pi, idxs, kind}] -- one entry per DISTINCT mixture."""
    with open(meta_path) as f:
        meta = json.load(f)
    groups = {}
    for m in meta["members"]:
        g = groups.setdefault(str(m["mixture_id"]),
                              {"mixture_id": str(m["mixture_id"]),
                               "pi": [float(x) for x in m["pi"]],
                               "kind": str(m["kind"]), "idxs": []})
        g["idxs"].append(int(m["idx"]))
    out = sorted(groups.values(), key=lambda g: g["mixture_id"])
    return meta, out


def nn_surviving(target, mixtures, radius: float):
    """(L1 to the nearest mixture outside the exclusion ball, n_excluded_mixtures).

    Mirrors train_flow.py's `keep = d1 > radius` exactly, including the strict >,
    so the manifest cannot disagree with what the job actually trains on.
    """
    kept = [m for m in mixtures if l1(m["pi"], target) > radius]
    n_excl = len(mixtures) - len(kept)
    best = min((l1(m["pi"], target) for m in kept), default=float("nan"))
    return best, n_excl, sum(len(m["idxs"]) for m in kept)


def fmt_pi(pi) -> str:
    return " ".join(f"{x:.9g}" for x in pi)


def design_sweep(_mixtures):
    """Round 1. Noise arms; no exclusion, since these tune the model not the holdout."""
    arms = []
    for pn in (0.0, 0.01, 0.05, 0.1):
        arms.append((f"pn{pn:g}".replace(".", "p"),
                     f"--path_noise {pn}", float("nan"), 0, -1, ""))
    for ss in (0.5, 2.0):
        arms.append((f"ss{ss:g}".replace(".", "p"),
                     f"--source_std {ss}", float("nan"), 0, -1, ""))
    return arms


def design_null(_mixtures):
    """
    The MI ablation: the SAME net, the SAME budget, conditioning removed.

    On a one-architecture zoo `--cond_mode family` is an nn.Embedding lookup with a
    single row, i.e. a constant, i.e. genuinely unconditional. The held-out
    log-likelihood gap between this flow and the pi-conditioned one is then a lower
    bound on I(x; pi) -- but ONLY if everything else is identical. The family flows
    already on disk were trained for 500 epochs against the pi flows' 20000, so
    using them would measure a training budget and report it as mutual information.

    One arm per configuration under test, so the ablation is matched to whichever
    configuration the sweep selects. Submit these with COND_MODE=family; the mode is
    an environment variable on the sbatch rather than an extra_arg, because two
    --cond_mode flags on one command line resolve by position.
    """
    arms = []
    for ss in (0.5, 1.0):
        arms.append((f"null_ss{ss:g}".replace(".", "p"),
                     f"--source_std {ss}", float("nan"), 0, -1, ""))
    return arms


def design_sweep2(_mixtures):
    """
    Round 2, sited by round 1's MEASURED result rather than by prediction.

    Round 1 on Mini b=0.15, k=50, all-100 basis (a_pi: 1.0 is correct, below 1 is
    an over-tight conditional, which is the failure being chased):

        source_std 0.5    a_pi 0.877   <- best, and the direction was NOT obvious
        source_std 1.0    a_pi 0.707      (the committed default)
        source_std 2.0    a_pi 0.712
        path_noise 0.01   a_pi 0.633   <- WORSE, and worse than the 0.0 baseline
        path_noise 0.05   a_pi 0.648
        path_noise 0.1    a_pi 0.683

    path_noise is dropped. It was the a-priori favourite -- it blurs the
    interpolant target, so it "should" widen the conditional -- and it does the
    opposite: it raises the POOLED spread above 1 (1.07-1.08) while tightening the
    conditional. That is the pooled/conditional inversion of the dispersion ratio,
    appearing in a knob rather than in a statistic, and it is reported as a
    negative result rather than dropped quietly.

    So: walk source_std DOWN, and add the adaLN-Zero arms at the two source_std
    values that bracket round 1's best.
    """
    arms = []
    for ss in (0.125, 0.25, 0.35, 0.75):
        arms.append((f"ss{ss:g}".replace(".", "p"),
                     f"--source_std {ss}", float("nan"), 0, -1, ""))
    for ss in (0.5, 1.0):
        tag = f"adaln_ss{ss:g}".replace(".", "p")
        arms.append((tag, f"--cond_inject adaln --source_std {ss}",
                     float("nan"), 0, -1, ""))
    # adaLN on a WIDER residual stream: the point of per-block modulation is that
    # the stream can carry more than the input concat could, so a width arm is the
    # honest test of whether the mechanism or the capacity is doing the work.
    arms.append(("adaln_w128_ss0p5",
                 "--cond_inject adaln --source_std 0.5 --hidden_dims 128 256 128",
                 float("nan"), 0, -1, ""))
    return arms


def stratified_subset(mixtures, n: int):
    """
    `n` mixtures spanning the nearest-neighbour-distance range, anchors first.

    Difficulty here IS nn_l1 -- how far the nearest mixture the flow still sees is
    from the target -- so a subset that does not span it cannot say whether the
    flow degrades more slowly than retrieval. Every anchor is taken first because
    the five one-hot vertices are the only EXTRAPOLATION cells the zoo contains,
    and they are exactly where retrieval has headroom.
    """
    scored = [(nn_surviving(m["pi"], mixtures, 1e-6)[0], m) for m in mixtures]
    anchors = [m for d, m in scored if m["kind"] == "anchor"]
    singles = sorted((x for x in scored if x[1]["kind"] != "anchor"),
                     key=lambda x: x[0])
    out = list(anchors[:n])
    room = n - len(out)
    if room > 0 and singles:
        # even strides through the sorted-by-difficulty list
        step = max(1, len(singles) // room)
        out += [m for _, m in singles[::step]][:room]
    return out


def design_lomo(mixtures, subset=None):
    """One arm per distinct mixture. radius 1e-6 drops that mixture's replicates
    and nothing else."""
    targets = mixtures if not subset else stratified_subset(mixtures, subset)
    arms = []
    for m in targets:
        nn, n_excl, n_rows = nn_surviving(m["pi"], mixtures, 1e-6)
        arms.append((f"lomo_{m['mixture_id']}",
                     f"--center_pi {fmt_pi(m['pi'])} --exclude_within_l1 1e-6",
                     nn, n_excl, n_rows, fmt_pi(m["pi"])))
    return arms


def design_radius(mixtures, targets, radii):
    arms = []
    for t in targets:
        m = next((x for x in mixtures if x["mixture_id"] == t), None)
        if m is None:
            raise SystemExit(f"--targets: no mixture named {t!r}")
        for r in radii:
            nn, n_excl, n_rows = nn_surviving(m["pi"], mixtures, r)
            if n_rows < 3:
                print(f"  [skip] {t} r={r}: only {n_rows} training rows survive")
                continue
            arms.append((f"rad_{t}_r{r:g}".replace(".", "p"),
                         f"--center_pi {fmt_pi(m['pi'])} --exclude_within_l1 {r}",
                         nn, n_excl, n_rows, fmt_pi(m["pi"])))
    return arms


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--design", required=True,
                   choices=["sweep", "sweep2", "null", "lomo", "radius"])
    p.add_argument("--meta", required=True, help="zoo_meta.json for the tier")
    p.add_argument("--out", required=True)
    p.add_argument("--targets", nargs="+",
                   default=["anchor_math", "dirichlet_050", "dirichlet_063",
                            "dirichlet_068", "dirichlet_070"],
                   help="radius design only: mixture_ids to centre the balls on")
    p.add_argument("--radii", nargs="+", type=float,
                   default=[0.0, 0.1, 0.2, 0.3, 0.4])
    p.add_argument("--subset", type=int, default=None,
                   help="lomo only: use this many mixtures instead of all of "
                        "them, stratified over nearest-neighbour distance with "
                        "every anchor taken first. For comparing CONFIGURATIONS "
                        "out of sample before committing to the full 85.")
    p.add_argument("--config_args", default="",
                   help="Appended verbatim to every arm, e.g. "
                        "'--source_std 0.5'. Lets one design be run under several "
                        "model configurations without editing the design.")
    p.add_argument("--tag_prefix", default="",
                   help="Prepended to every tag, so the same design under two "
                        "configurations writes two disjoint sets of directories.")
    args = p.parse_args()

    meta, mixtures = load_mixtures(args.meta)
    print(f"{args.meta}: {meta.get('n_members')} members, "
          f"{len(mixtures)} distinct mixtures, beta={meta.get('beta')}")

    if args.design == "sweep":
        arms = design_sweep(mixtures)
    elif args.design == "sweep2":
        arms = design_sweep2(mixtures)
    elif args.design == "null":
        arms = design_null(mixtures)
    elif args.design == "lomo":
        arms = design_lomo(mixtures, args.subset)
    else:
        arms = design_radius(mixtures, args.targets, args.radii)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# tag\tnn_l1\tn_excluded_mixtures\tn_train_rows"
                "\textra_args\ttarget_pi\n")
        for tag, extra, nn, n_excl, n_rows, pi in arms:
            tag = f"{args.tag_prefix}{tag}"
            if args.config_args:
                extra = f"{extra} {args.config_args}".strip()
            f.write(f"{tag}\t{nn:.6f}\t{n_excl}\t{n_rows}\t{extra}\t{pi}\n")

    print(f"-> {args.out}: {len(arms)} arm(s)")
    if args.design in ("lomo", "radius"):
        vals = sorted(a[2] for a in arms if a[2] == a[2])
        if vals:
            print(f"   nearest-surviving-mixture L1: min {vals[0]:.3f}  "
                  f"median {vals[len(vals) // 2]:.3f}  max {vals[-1]:.3f}")
    print(f"   submit with: sbatch --array=0-{len(arms) - 1} slurm/flow_arm.sbatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

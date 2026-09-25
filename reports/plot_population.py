#!/usr/bin/env python
"""
plot_population.py -- the population-side figures, from reports/population_payload.json.

The generator-side figures are reports/plot_conditioning.py. The population side needs
no cluster job: the Schuerholt-convention PCA/t-SNE coordinates are not shipped, only the
centred Grams -- which is enough: a 100x100 Gram eigendecomposition IS the PCA of the population, and it is a
one-liner here rather than a cluster job.

Conventions are plot_conditioning.py's, deliberately: same rcParams, same semantic
palette, PDF and PNG side by side, skip() rather than crash on a missing input, and its
hard rule -- if a figure disagrees with a table, the figure is wrong. Every number drawn
here is read from the payload, never recomputed, with one exception that is flagged in
its own docstring.

  python3 reports/plot_population.py \\
      --payload reports/population_payload.json \\
      --gram "Mini=reports/data/population/mini_b015/weight_embeddings" \\
      --out_dir outputs/figs

No torch, no GPU, laptop-local.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# plot_conditioning.py's palette, extended with one entry per domain. The domain
# colours are chosen to be distinguishable in greyscale order as well as in hue,
# because the weight-space panel is read as five clusters and not as five values.
C = {"real": "#111111", "ceiling": "#8c8c8c", "flow": "#1f77b4",
     "quadratic": "#ff7f0e", "retrieval": "#2ca02c", "linear": "#9467bd",
     "mean": "#7f7f7f", "gauss": "#d62728"}
DOMAIN_C = {"web": "#1f77b4", "code": "#ff7f0e", "math": "#2ca02c",
            "books": "#9467bd", "multilingual": "#d62728"}
SCALE_C = [C["flow"], C["quadratic"], C["linear"]]
POOLS = ["within_anchor", "between_anchor", "singleton_singleton",
         "anchor_singleton"]
POOL_LABEL = {"within_anchor": "within-anchor\n(same $\\pi$)",
              "between_anchor": "between-anchor\n(diff. $\\pi$)",
              "singleton_singleton": "interior--\ninterior",
              "anchor_singleton": "anchor--\ninterior"}

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8,
})

MADE, SKIPPED = [], []


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    MADE.append(name)
    print(f"  [fig] {name}.pdf / .png")


def skip(name, why):
    SKIPPED.append((name, why))
    print(f"  [skip] {name}: {why}")


def read(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def gram_pca(gram_dir, ncomp=2):
    """The population's principal coordinates, from the centred Gram.

    This is the one quantity on this page that is computed rather than read. A centred
    Gram's eigendecomposition gives the same coordinates as a PCA of the weights --
    U sqrt(Lambda) -- without ever materialising a D x k basis, which is the same dual
    trick the paper's representation uses. Nothing downstream quotes a number from it;
    it positions points.
    """
    g = os.path.join(gram_dir, "centered_gram.npy")
    lab = os.path.join(gram_dir, "labels.json")
    if not (os.path.exists(g) and os.path.exists(lab)):
        return None, None
    G = np.load(g)
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1][:ncomp]
    coords = V[:, order] * np.sqrt(np.clip(w[order], 0, None))
    with open(lab) as f:
        return coords, json.load(f)


def domain_of(gram_dir, labels, payload):
    """The dominant domain of each member's mixture, for colouring.

    Anchors carry their domain in labels.json; a singleton is named only by its draw
    index, so the mixture itself is read from the zoo plan that sits beside the Gram.
    Falls back to the label when no plan is found, which leaves the interior grey.
    """
    meta = read(os.path.join(os.path.dirname(gram_dir.rstrip("/")),
                             "zoo_meta.json"))
    doms = (meta or {}).get("domains") or (payload or {}).get("domains") or []
    by_idx = {int(m["idx"]): m for m in ((meta or {}).get("members") or [])}
    out = []
    for L in labels:
        m = by_idx.get(int(L.get("member_idx", L.get("row", -1))))
        if m and doms:
            pi = m["pi"]
            out.append(doms[max(range(len(pi)), key=lambda j: pi[j])])
        else:
            out.append(L.get("domain"))
    return out


def fig_zoo_design(payload, grams, out_dir):
    """What the population is, and that the mixture organises it.

    Four panels because the claim has four parts, and three of them are otherwise only
    readable off a table: the design is corners plus a close-packed interior; the
    mixture separates the population in weight space; and the separation is visible in
    both weight space and function space, narrowing with scale in both.
    """
    rungs = payload.get("rungs") or []
    if not rungs:
        return skip("zoo_design", "payload carries no rungs")

    fig, ax = plt.subplots(2, 2, figsize=(9.2, 6.6))

    # (a) the design, in the two coordinates that matter for it: how concentrated a
    # mixture is, and how much entropy it has. The anchors are one point, at (1, 0).
    d = rungs[0]["displacement"]
    kinds = np.array(d["kind"])
    mx, ent = np.array(d["max_pi"]), np.array(d["pi_entropy"])
    a0 = ax[0][0]
    m = kinds == "singleton"
    a0.scatter(mx[m], ent[m], s=22, color=C["flow"], alpha=0.75, linewidths=0,
               label=f"singletons ($n={int(m.sum())}$, {int(m.sum())} distinct $\\pi$)")
    a0.scatter(mx[~m], ent[~m], s=70, facecolors="none", edgecolors=C["real"],
               linewidths=1.3,
               label=f"anchors ($n={int((~m).sum())}$, $5$ distinct $\\pi$)")
    pg = rungs[0]["pi_regression"]["pi_geometry"]
    a0.axhline(pg["entropy_uniform"], color=C["ceiling"], ls=":", lw=1.2)
    a0.text(0.985, pg["entropy_uniform"], " barycentre", ha="right", va="bottom",
            fontsize=7.5, color=C["ceiling"])
    a0.set_xlabel("$\\max_d \\pi_d$ (mixture concentration)")
    a0.set_ylabel("$H(\\pi)$ (nats)")
    a0.set_title("(a) the mixture design on $\\Delta^4$")
    a0.legend(loc="upper right")

    # (b) the Schuerholt-convention panel: the population in its own principal
    # coordinates, coloured by which domain dominates the mixture.
    a1 = ax[0][1]
    lab_used = None
    for nm, gd in (grams or {}).items():
        coords, labels = gram_pca(gd)
        if coords is None:
            continue
        lab_used = nm
        # An interior member is coloured by which domain DOMINATES its mixture, not
        # left grey. labels.json names a singleton "dirichlet_037", so the mixture
        # comes from the zoo plan beside the Gram; without it the interior cloud
        # carries no information and the panel only shows five clusters one already
        # knows are there.
        doms = domain_of(gd, labels, payload)
        for i, L in enumerate(labels):
            anchor = L.get("kind") == "anchor"
            a1.scatter(coords[i, 0], coords[i, 1],
                       s=64 if anchor else 20,
                       color=DOMAIN_C.get(doms[i], C["mean"]),
                       alpha=0.95 if anchor else 0.50,
                       marker="D" if anchor else "o", linewidths=0)
        break
    if lab_used is None:
        a1.text(0.5, 0.5, "no centred Gram supplied", ha="center", va="center",
                transform=a1.transAxes, color=C["ceiling"])
    else:
        for dom, col in DOMAIN_C.items():
            a1.scatter([], [], s=28, color=col, linewidths=0, label=dom)
        a1.legend(loc="best", ncol=2, fontsize=7)
    a1.set_xlabel("principal coordinate 1")
    a1.set_ylabel("principal coordinate 2")
    a1.set_title(f"(b) weight space, {lab_used or '--'}"
                 "  ($\\diamond$ anchors)")

    # (c) and (d): the same four pools in weight space and in function space. Two
    # independent measurements whose orderings agree at the top and disagree at the
    # bottom, which is a result and not a discrepancy.
    x = np.arange(len(POOLS))
    w = 0.26
    for j, r in enumerate(rungs):
        cos = [r["cos"]["drift_removed"][p] for p in POOLS]
        dis = [r["predictions"]["disagreement_pools"][p] for p in POOLS]
        for a, vals in ((ax[1][0], cos), (ax[1][1], dis)):
            a.bar(x + (j - 1) * w, vals, width=w, color=SCALE_C[j % len(SCALE_C)],
                  label=r["label"])
    for a in (ax[1][0], ax[1][1]):
        a.set_xticks(x)
        a.set_xticklabels([POOL_LABEL[p] for p in POOLS], fontsize=7.5)
        a.axhline(0.0, color=C["real"], lw=0.9)
    ax[1][0].set_ylabel("branch-delta cosine (drift removed)")
    ax[1][0].set_title("(c) weight-space alignment by pool")
    ax[1][0].legend(loc="upper right")
    ax[1][1].set_ylabel("next-token disagreement")
    ax[1][1].set_title("(d) functional difference (same pools)")

    fig.suptitle("Mixture Identity Organizes Weight Space and Behavior",
                 y=1.01, fontsize=11)
    fig.tight_layout()
    save(fig, out_dir, "zoo_design")

    # A population whose variation has a known cause: "
    #              "mixture identity organises both weight space and behaviou


def fig_zoo_barrier(payload, out_dir):
    """The trunk-branch shortcut, audited.

    Every interpolant between two members lies BELOW the chord at every scale and in
    both pools, so there is no barrier to align away and canonicalisation has nothing
    to do. Drawn rather than tabulated because the claim is about a whole curve: a
    table row can only report its maximum, and the maximum of a curve that never rises
    is the least informative point on it.
    """
    rungs = payload.get("rungs") or []
    if not rungs:
        return skip("zoo_barrier", "payload carries no rungs")
    if not any(r.get("barrier", {}).get("rows") for r in rungs):
        return skip("zoo_barrier", "payload carries no per-pair barrier curves")

    fig, axes = plt.subplots(1, len(rungs), figsize=(3.4 * len(rungs), 2.9),
                             squeeze=False, sharey=True)
    style = {"within_anchor": (C["flow"], "within-anchor (same $\\pi$)"),
             "between_anchor": (C["quadratic"], "between-anchor (diff. $\\pi$)")}
    for j, (a, r) in enumerate(zip(axes[0], rungs)):
        rows = r["barrier"]["rows"]
        seen = set()
        for row in rows:
            col, nm = style.get(row["pool"], (C["mean"], row["pool"]))
            cur = row["curve"]
            a.plot([c["alpha"] for c in cur], [c["excess"] for c in cur],
                   color=col, lw=0.9, alpha=0.45,
                   label=(nm if row["pool"] not in seen else None))
            seen.add(row["pool"])
        a.axhline(0.0, color=C["real"], ls="--", lw=1.2)
        a.set_xlabel("$\\alpha$ along $(1-\\alpha)\\theta_i + \\alpha\\theta_j$")
        a.set_title(f"{r['label']} ({len(rows)} pairs)")
        if j == 0:
            a.set_ylabel("Excess over the chord (nats)")
            a.legend(loc="lower center")
    fig.suptitle("No Positive Interpolation Barrier", y=1.03, fontsize=11)
    save(fig, out_dir, "zoo_barrier")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--payload", default="reports/population_payload.json")
    p.add_argument("--gram", action="append", default=[], metavar="LABEL=DIR",
                   help="a weight_embeddings/ directory holding centered_gram.npy "
                        "and labels.json. Only the first is drawn.")
    p.add_argument("--out_dir", default="outputs/figs")
    a = p.parse_args()

    payload = read(a.payload)
    if not payload:
        print(f"no payload at {a.payload}")
        return 1
    grams = {}
    for spec in a.gram:
        if "=" not in spec:
            print(f"  [skip] --gram {spec!r}: expected LABEL=DIR")
            continue
        lab, d = spec.split("=", 1)
        grams[lab] = d

    print(f"rendering into {a.out_dir}/")
    fig_zoo_design(payload, grams, a.out_dir)
    fig_zoo_barrier(payload, a.out_dir)
    print(f"\n{len(MADE)} figure(s) written, {len(SKIPPED)} skipped.")
    for nm, why in SKIPPED:
        print(f"  - {nm}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
plot_conditioning.py -- the figures for the conditional-generation appendix.

THE ONLY FILE IN THIS PROJECT THAT IMPORTS MATPLOTLIB. reports/build_*_figures.py
and scripts/report_*.py are pure stdlib by design (tests/test_report.py's
test_stdlib_purity enforces it for five of them), so plotting lives here, behind the
JSON those scripts already emit. Nothing here recomputes a number: if a figure
disagrees with a table, the figure is wrong.

INPUTS, all optional -- every figure is skipped with a reason rather than crashing:
  --functional   report_functional.py --json_out      (ladder, staircase, dose-response)
  --density      report_density.py --out              (identification, memorisation)
  --gen LABEL=PATH   eval_generated.py output, repeatable  (dose-response scatter)
  --zoo          domain_separation.json               (the real zoo's scatter)
  --meta         zoo_meta.json                        (mixtures for the zoo scatter)
  --arms         report_flow_arms.py --json_out       (the tuning sweep)

  python reports/plot_conditioning.py --functional reports/data/fn_b015.json \
      --out_dir reports/figs
"""
from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker   # noqa: E402

DOMAINS = ("web", "code", "math", "books", "multilingual")

# One palette, used the same way everywhere: the real zoo is black, the flow is the
# one saturated colour, every baseline is grey-to-cool, and the null is red.
C = {"real": "#111111", "ceiling": "#8c8c8c", "flow": "#1f77b4",
     "pca_only": "#17becf", "quadratic": "#ff7f0e", "retrieval": "#2ca02c",
     "linear": "#9467bd", "mean": "#7f7f7f", "gauss": "#d62728"}

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8,
})

MADE: list = []
SKIPPED: list = []


def save(fig, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    MADE.append(name)
    print(f"  [fig] {name}.pdf / .png")


def skip(name: str, why: str) -> None:
    SKIPPED.append((name, why))
    print(f"  [skip] {name}: {why}")


def read(path):
    """None for anything unreadable. A figure script must degrade to a skipped
    figure, never to a traceback: it is run last, over whatever artifacts a long
    pipeline actually produced, and half the inputs are legitimately absent."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        print(f"  [warn] {path}: {e}")
        return None


# ---------------------------------------------------------------------------

def _arm_of(t):
    """The single generated arm of a functional report, with its parent record."""
    t = t[0] if isinstance(t, list) else t
    g = (t.get("generated") or [None])[0]
    if not g:
        return t, None, None
    arms = g.get("arms") or {}
    return t, g, (list(arms.values())[0] if arms else None)


def fig_ladder(tiers, fn, out_dir):
    """Where every arm sits between the noise floor and the best baseline.

    Drawn one panel per scale when --tier is given, since the ladder's shape
    holding across a 6.9x parameter range is most of what it has to say. Falls
    back to a single panel from --functional.
    """
    panels = (list(tiers.items()) if tiers
              else [(t.get("label", ""), t)
                    for t in (fn if isinstance(fn, list) else [fn] if fn else [])])
    if not panels:
        return skip("ladder", "no --tier LABEL=PATH and no --functional JSON")
    # Shared x and one set of y labels: the rungs are identical in every panel, so
    # repeating the labels cost horizontal room and let the longest of them collide
    # with the neighbouring panel's annotations, and unshared axes made bar lengths
    # incomparable across the very scales the figure exists to compare.
    fig, axes = plt.subplots(1, len(panels), figsize=(3.9 * len(panels) + 1.2, 3.1),
                             squeeze=False, sharex=True)
    for j, (ax, (lab, t)) in enumerate(zip(axes[0], panels)):
        t = t[0] if isinstance(t, list) else t
        lad = t["ladder"]
        rungs = [("ceiling (replicate noise)", lad["ceiling"], C["ceiling"])]
        for arm in t.get("generated") or []:
            for an, a in (arm.get("arms") or {}).items():
                v = a.get("profile_err_pct")
                if v == v:
                    rungs.append(("flow (ours)", v, C.get(an, C["flow"])))
        # The "_per_mixture" variants, NOT the bare keys. A generated arm is scored
        # once per held-out MIXTURE, so the bare keys -- which average once per
        # MEMBER over all 100 -- are a different denominator: retrieval reads 8.92
        # there against 5.75 here. The first version of this figure drew 4.05
        # against 8.92 and flattered the flow by an entire baseline. The tables in
        # (build_appendix_tables.py) use the per-mixture column, and a figure that disagrees with
        # a table is the figure that is wrong.
        for k, nm in (("quadratic_per_mixture", "quadratic on pi"),
                      ("linear_per_mixture", "linear on pi"),
                      ("mean_per_mixture", "global mean")):
            rungs.append((nm, lad[k], C[k.replace("_per_mixture", "")]))
        # Retrieval comes from the arm's MATCHED value, not the ladder's own
        # leave-one-mixture-out column. The two are separate estimators of the
        # same thing and disagree by ~0.01 pp; the tables print the matched one,
        # because it is computed on the cells the flow was scored on.
        for arm in t.get("generated") or []:
            for a in (arm.get("arms") or {}).values():
                mr = a.get("matched_retrieval_pct")
                if mr == mr and mr is not None:
                    rungs.append(("retrieval (1-NN)", mr, C["retrieval"]))
                    break
        rungs.sort(key=lambda r: r[1])
        y = range(len(rungs))
        ax.barh(list(y), [r[1] for r in rungs],
                color=[r[2] for r in rungs], height=0.62)
        ax.set_yticks(list(y))
        ax.set_yticklabels([r[0] for r in rungs] if j == 0 else [])
        ax.invert_yaxis()
        c = lad["ceiling"]
        for i, r in enumerate(rungs):
            ax.text(r[1], i, f"  {r[1]:.2f}%  ({r[1] / c:.1f}x)",
                    va="center", fontsize=7.5)
        ax.set_xlim(0, 21.0)
        ax.set_xlabel("% $\\Delta$PPL" if j else
                      "profile error  (% $\\Delta$PPL, mean over 5 domains)")
        ax.set_title(f"{lab or t.get('label', '')}  (N={t['n_members']})")
    fig.suptitle("Functional Gap vs Noise Expectation",
                 y=1.03, fontsize=11)
    save(fig, out_dir, "ladder")


def _dose_points(t, arm, j):
    """Per-domain (requested pi_d, measured log ppl) for the real zoo and the
    generated cohort, taken from the functional report's own cells.

    Reading them here rather than from a separate --zoo/--meta/--gen triple is what
    makes the figure agree with tab_dose: the generated slope refitted on `gen_v`
    reproduces the table's to four decimals on all five domains. The real points are
    per-MIXTURE means (`mu`, 85 of them), whereas the table's real slope is fitted
    per member (100), so the real LINE is drawn from the table's slope rather than
    refitted on the visible points -- an anchor's four replicates carry leverage
    that their mean does not.
    """
    real = [(c["pi"][j], c["mu"][j]) for c in arm["cells"]]
    gen = [(c["pi"][j], v[j]) for c in arm["cells"] for v in c["gen_v"]]
    return real, gen


def fig_dose_response(fn, gens, zoo, meta, out_dir):
    """Ask for more of domain d; does perplexity on d actually fall?"""
    if not fn:
        return skip("dose_response", "no --functional JSON")
    t = (fn if isinstance(fn, list) else [fn])[0]
    _, g0, arm = _arm_of(t)
    domains = t.get("domains") or list(DOMAINS)
    if not arm or not arm.get("cells"):
        return skip("dose_response", "functional JSON carries no per-cell records")
    real_slopes = {r["domain"]: r["slope"] for r in t["dose_response"]}
    gen_slopes = {r["domain"]: r["slope"] for r in arm["dose_response"]}

    fig, axes = plt.subplots(1, len(domains), figsize=(2.5 * len(domains), 2.9),
                             sharex=True)
    for j, (ax, d) in enumerate(zip(axes, domains)):
        real, gen = _dose_points(t, arm, j)
        for pts, col, sl, nm, s in ((real, C["real"], real_slopes[d], "real zoo", 9),
                                    (gen, C["flow"], gen_slopes[d], "flow", 4)):
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=s, c=col, alpha=0.30, linewidths=0)
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            gx = [min(xs), max(xs)]
            ax.plot(gx, [my + sl * (x - mx) for x in gx], color=col, lw=1.9,
                    ls="--" if nm == "real zoo" else "-",
                    label=(f"{nm} ({sl:+.2f})" if j == 0 else f"{sl:+.2f}"))
        ax.set_title(d)
        ax.set_xlabel("requested $\\pi_d$")
        ax.legend(loc="upper right", fontsize=6.5)
    axes[0].set_ylabel("log ppl on $d$  (nats)")
    fig.suptitle("Dose-Response Capture", y=1.05, fontsize=11)
    save(fig, out_dir, "dose_response")


def _traverse(g, domains):
    """A barycentre-to-vertex traverse: (domain, positions, mean log ppl).

    The samples at one request are averaged before plotting, the same way
    build_appendix_tables.tab_continuity does it, so the curve and the table agree.
    Without that averaging the flow's sampling noise reads as wobble, and wobble is
    exactly the quantity the total-variation row reports.
    """
    rows = [r for r in (g.get("rows") or []) if r.get("kind") == "path"]
    if not rows:
        return None
    dom = domains[max(range(len(domains)), key=lambda j: rows[-1]["pi"][j])]
    by = {}
    for r in rows:
        by.setdefault(r["label"], []).append(math.log(float(r["ppl"][dom])))
    keys = sorted(by)
    vs = [sum(by[k]) / len(by[k]) for k in keys]
    ts = [i / (len(keys) - 1) for i in range(len(keys))] if len(keys) > 1 else [0.0]
    return dom, ts, vs


def fig_continuity(fn, gens, out_dir):
    """Retrieval has 85 answers forever; a conditional generator has as many as
    you ask for."""
    if not fn:
        return skip("continuity", "no --functional JSON")
    t = (fn if isinstance(fn, list) else [fn])[0]
    domains = t.get("domains") or list(DOMAINS)
    paths = t.get("paths") or {}
    have = {d: p for d, p in paths.items() if p.get("vs")}
    if not have:
        return skip("continuity", "path_staircase carries no raw series "
                                  "(re-run report_functional.py)")

    # The flow's own traverse, when one was supplied. Overlaying it is the whole
    # point of the figure, and it was never implemented: the caption promised "the
    # flow's curve against retrieval's staircase" and the figure drew the staircase
    # alone on all five domains.
    trav = {}
    for lab, g in (gens or {}).items():
        tr = _traverse(g, domains)
        if tr:
            trav[tr[0]] = (lab, tr[1], tr[2])

    # With a traverse in hand the figure narrows to that domain and shows the
    # comparison full width; without one it falls back to retrieval on all five.
    show = [d for d in have if d in trav] or list(have)
    fig, axes = plt.subplots(1, len(show),
                             figsize=((5.6 if len(show) == 1 else 2.6 * len(show)),
                                      3.1), squeeze=False)
    for ax, d in zip(axes[0], show):
        p = have[d]
        ax.step(p["t"], p["vs"], where="post", color=C["retrieval"], lw=1.7,
                label=f"retrieval: {p['n_distinct']} answers, {p['jumps']} jumps")
        if d in trav:
            lab, ts, vs = trav[d]
            ax.plot(ts, vs, color=C["flow"], lw=1.7, marker="o", ms=2.6,
                    label=f"{lab}: {len(vs)} answers, 0 jumps")
        ax.set_title(f"barycentre $\\rightarrow$ {d}")
        ax.set_xlabel("position along the path")
        ax.legend(loc="best", fontsize=7)
    axes[0][0].set_ylabel("log ppl on the target domain (nats)")
    fig.suptitle("Lookup vs Flow Continuity in $\\pi$ ",
                 y=1.03, fontsize=11)
    save(fig, out_dir, "continuity")


def fig_arms(arms, out_dir):
    """The tuning sweep, and the reason it is not ranked on the loss."""
    if not arms:
        return skip("tuning_sweep", "no --arms JSON")
    good = [a for a in arms if isinstance(a.get("a_pi"), (int, float))
            and a["a_pi"] == a["a_pi"]]
    if not good:
        return skip("tuning_sweep", "no arm reports a_pi")
    good.sort(key=lambda a: abs(1.0 - a["a_pi"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    names = [a["arm"].replace("flow_k50_codes_pi", "").lstrip("_") or "(base)"
             for a in good]
    cols = [C["flow"] if a.get("cond_inject") == "concat" else C["quadratic"]
            for a in good]
    ax1.barh(range(len(good)), [a["a_pi"] for a in good], color=cols, height=0.62)
    ax1.axvline(1.0, color=C["real"], lw=1.4, ls="--")
    ax1.text(1.0, -0.9, " correct", fontsize=7.5, color=C["real"])
    ax1.set_yticks(range(len(good)))
    ax1.set_yticklabels(names, fontsize=7)
    ax1.invert_yaxis()
    ax1.set_xlabel("$a_\\pi$ = sampled spread / real within-mixture spread")
    ax1.set_title("Dispersion guard (1.0 is correct)")

    ax2.scatter([a["best_loss"] for a in good], [a["a_pi"] for a in good],
                c=cols, s=34, zorder=3)
    ax2.axhline(1.0, color=C["real"], lw=1.2, ls="--")
    best = min(good, key=lambda a: a["best_loss"])
    ax2.annotate("lowest loss,\nworst dispersion",
                 (best["best_loss"], best["a_pi"]),
                 textcoords="offset points", xytext=(14, 6), fontsize=7.5,
                 color=C["gauss"],
                 arrowprops=dict(arrowstyle="->", color=C["gauss"], lw=1.0))
    ax2.set_xlabel("validation loss (lower is 'better')")
    ax2.set_ylabel("$a_\\pi$")
    ax2.set_title("Why the loss cannot select the model")
    fig.suptitle("Flow Tuning", y=1.04, fontsize=11)
    save(fig, out_dir, "tuning_sweep")


def _tier_label(d):
    """Name a density run by its SCALE, not its rank. Every run in this appendix is
    k=50, so labelling by k drew three identical legend entries."""
    rn = d.get("run_name") or ""
    for tag, lab in (("mini", "Mini (51M)"), ("small", "Small (124M)"),
                     ("medium", "Medium (355M)")):
        if tag in rn:
            return lab
    return f"k={d.get('k')}"


def fig_identification(dens, out_dir):
    """Can the flow's own density pick the mixture a model was trained on?"""
    if not dens:
        return skip("identification", "no --density JSON")
    ds = dens if isinstance(dens, list) else [dens]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    drew = False
    for d in ds:
        ident = d.get("identification") or {}
        ranks = ident.get("ranks")
        if not ranks:
            continue
        n = ident.get("n_candidates") or max(ranks)
        xs = sorted(ranks)
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        ax.step(xs, ys, where="post", lw=1.8,
                label=f"{_tier_label(d)}  top-1 "
                      f"{ident.get('top1', float('nan')):.0%}")
        drew = True
    if not drew:
        return skip("identification", "density JSON carries no per-item ranks")
    ax.plot([1, n], [1 / n, 1.0], color=C["real"], ls="--", lw=1.2, label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("rank of the true mixture among candidates")
    ax.set_ylabel("fraction of held-out models at or below")
    ax.set_title("Identification from $\\log p_\\theta(x \\mid \\pi)$")
    ax.legend(loc="lower right")
    save(fig, out_dir, "identification")


def _excl_label(d):
    """Name a density run by the neighbourhood its flow was blinded to. Falls back
    through flow_dir to k, so a run with no exclusion still gets a label."""
    fd = d.get("flow_dir") or ""
    for tag in ("_rad_", "_lomo_"):
        if tag in fd:
            return fd.split(tag, 1)[1].replace("dirichlet_", "$\\pi_{") \
                     .replace("_r0p", "}$, r=0.").replace("_r0", "}$, r=0")
    return f"k={d.get('k')}"


def fig_memorisation(dens, out_dir):
    """Meehan-Chaudhuri-Dasgupta: are generated codes closer to training members
    than genuinely held-out members are?"""
    if not dens:
        return skip("memorisation", "no --density JSON")
    ds = dens if isinstance(dens, list) else [dens]
    have = [d for d in ds if d.get("three_sample_lead5")]
    if not have:
        return skip("memorisation", "density JSON has no three-sample test")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    labels, gen, test, zs, ns = [], [], [], [], []
    for d in have:
        t = d["three_sample_lead5"]
        # Every one of these runs is the same k on the same zoo -- what separates
        # them is WHICH neighbourhood the flow was blinded to, so that is what the
        # axis has to say. Labelling them all "k=50" (the first version of this
        # figure) made four independent tests look like one repeated four times.
        labels.append(_excl_label(d))
        gen.append(t["mean_nn_gen_to_train"])
        test.append(t["mean_nn_test_to_train"])
        zs.append(t["Z_U"])
        ns.append(t.get("n_test") or d.get("three_sample_n_excluded"))
    x = range(len(labels))
    top = max(max(gen), max(test))
    ax.bar([i - 0.19 for i in x], gen, width=0.36, color=C["flow"],
           label="generated $\\rightarrow$ nearest training member")
    ax.bar([i + 0.19 for i in x], test, width=0.36, color=C["ceiling"],
           label="held-out real $\\rightarrow$ nearest training member")
    for i, z in enumerate(zs):
        v = "no copying" if -3 <= z <= 3 else ("DATA COPYING" if z < -3
                                               else "underfitting")
        ax.text(i, max(gen[i], test[i]) + 0.03 * top,
                f"$Z_U$ = {z:+.2f}\n{v}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{l}\n$n_{{test}}$={n}" for l, n in zip(labels, ns)],
                       fontsize=8)
    ax.set_ylabel("Mean NN distance (code space, $c_0$-$c_4$)")
    # Headroom for the annotation, and then more for the legend above it, so the
    # two cannot overlap however the bars come out.
    ax.set_ylim(0, top * 1.65)
    ax.set_title("Data-Copying Test")
    ax.legend(loc="upper left", fontsize=7.5)
    save(fig, out_dir, "memorisation")


def collect_cells(fn, cells):
    """Per-cell records, preferring an explicit --cells file and otherwise digging
    them out of the functional JSON, where score_generated already stores them."""
    if cells:
        return cells if isinstance(cells, list) else cells.get("cells") or []
    out = []
    for t in (fn if isinstance(fn, list) else [fn]) if fn else []:
        for g in t.get("generated") or []:
            for an, a in (g.get("arms") or {}).items():
                for c in a.get("cells") or []:
                    c = dict(c)
                    c["arm"] = f"{g['label']}/{an}"
                    out.append(c)
    return out


def _binned(pts, nbins):
    """Equal-count bins over x, returning (centre, mean_y, stderr_y, n)."""
    pts = sorted(pts)
    n = len(pts)
    if n < nbins * 2:
        nbins = max(1, n // 4)
    out, per = [], n / float(nbins)
    for b in range(nbins):
        lo, hi = int(round(b * per)), int(round((b + 1) * per))
        chunk = pts[lo:hi]
        if len(chunk) < 2:
            continue
        xs = [x for x, _ in chunk]
        ys = [y for _, y in chunk]
        m = sum(ys) / len(ys)
        var = sum((y - m) ** 2 for y in ys) / (len(ys) - 1)
        out.append((sum(xs) / len(xs), m, math.sqrt(var / len(ys)), len(ys)))
    return out


def fig_generalization(tiers, out_dir):
    """The headline, stated with its own limit.

    Left: the paired advantage over retrieval against how far the flow had to reach.
    Positive means the flow beat the nearest real model it was allowed to keep. Right:
    CRPS skill on the same axis, the PROPER score and the one that decides. Per tier,
    because a result that reproduces at two scales is the claim and a result at one is an
    anecdote.

    TWO THINGS THIS DREW WRONGLY BEFORE, both of which put a claim in the caption that
    the data does not support.

    1. It binned all 85 cells together, so the last bin was mostly the five one-hot
       VERTICES -- whose absolute error is ~10x the interior's -- and the resulting
       "advantage grows then falls away" shape was entirely those five points. Binning
       is now over the 80 INTERIOR cells only and the vertices are drawn as open squares
       at their own distance, so the interior trend and the vertex caveat are separate
       marks rather than one line.
    2. It scattered all 255 per-cell points, whose spread is ~10x the binned signal
       (advantage runs -7.0 to +12.6 while the bin means run +0.0 to +3.0), so the axis
       was set by outliers and the trend the figure exists to show was a flat squiggle
       near zero. Spread is now drawn as a +/-1 SD band around each tier's bin means:
       nothing falls outside the axis, and the means are legible.
    """
    if not tiers:
        return skip("generalization", "no --tier LABEL=PATH given")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    colours = [C["flow"], C["quadratic"], C["retrieval"], C["linear"]]
    drew = False
    for i, (lab, fn) in enumerate(tiers.items()):
        cells = collect_cells(fn, None)
        if not cells:
            continue
        col = colours[i % len(colours)]
        vert = [c for c in cells if max(c["pi"]) > 0.999]
        inter = [c for c in cells if max(c["pi"]) <= 0.999]
        pan = ((axes[0], lambda c: c["retrieval_pct"] - c["gen_err_pct"]),
               (axes[1], lambda c: c.get("crps_skill_pct")))
        for ax, val in pan:
            keep = lambda S: [(c["nn_l1"], val(c)) for c in S
                              if c.get("nn_l1") is not None and val(c) is not None]
            bins = _binned(keep(inter), 5)
            if bins:
                x = [b[0] for b in bins]
                m = [b[1] for b in bins]
                # b[2] is the standard error; recover the SD it came from, which is the
                # honest width of the cell population in that bin.
                sd = [b[2] * math.sqrt(b[3]) for b in bins]
                ax.fill_between(x, [a - c for a, c in zip(m, sd)],
                                [a + c for a, c in zip(m, sd)],
                                color=col, alpha=0.10, lw=0, zorder=1)
                ax.errorbar(x, m, yerr=[b[2] for b in bins], color=col, lw=2.0,
                            marker="o", ms=5.0, capsize=3.0, zorder=4, label=lab)
            vpts = keep(vert)
            if vpts:
                ax.scatter(*zip(*vpts), s=38, facecolors="none", edgecolors=col,
                           linewidths=1.3, zorder=5)
        drew = True
    if not drew:
        return skip("generalization", "tier JSONs carry no per-cell records")

    for ax in axes:
        ax.axhline(0.0, color=C["real"], ls="--", lw=1.1, zorder=2)
        ax.set_xlabel("$L_1$ to the nearest mixture the flow still saw")
        # Log x, because 80 of the 85 cells sit in 0.17-0.47 and the five vertices
        # sprawl out to 0.77: on a linear axis the interior -- the part the claim is
        # about -- is squeezed into the left 30% of the panel. Log spreads it over ~two
        # thirds of the width and still plots every vertex at its true distance.
        ax.set_xscale("log")
        ax.set_xticks([0.2, 0.25, 0.3, 0.4, 0.5, 0.7])
        ax.set_xticklabels(["0.2", "0.25", "0.3", "0.4", "0.5", "0.7"])
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    axes[0].set_ylabel("retrieval $-$ flow   (pp of % $\\Delta$PPL)")
    axes[0].set_title("(a) advantage over the nearest real model")
    axes[1].set_ylabel("CRPS skill vs. the zoo marginal")
    axes[1].set_title("(b) proper score, same axis")
    # A proxy handle for the open squares, added once rather than per tier.
    axes[0].scatter([], [], s=38, facecolors="none", edgecolors=C["real"],
                    linewidths=1.3, label="one-hot vertex (not binned)")
    # One legend, in (a). The two panels share tiers and markers, and in (b) every
    # corner that could hold a legend holds a vertex point.
    axes[0].legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    save(fig, out_dir, "generalization")


def fig_extrapolation(cells, out_dir):
    """Error against how far the nearest mixture the flow still saw actually is."""
    if not cells:
        return skip("extrapolation", "no per-cell records (leave-one-mixture-out "
                                     "or radius sweep not scored yet)")
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for key, lab, col in (("gen_err_pct", "flow", C["flow"]),
                          ("retrieval_pct", "retrieval (1-NN)", C["retrieval"]),
                          ("ceiling_pct", "replicate ceiling", C["ceiling"])):
        pts = [(c["nn_l1"], c[key]) for c in cells
               if c.get("nn_l1") == c.get("nn_l1") and c.get(key) == c.get(key)]
        if not pts:
            continue
        pts.sort()
        ax.scatter(*zip(*pts), s=22, color=col, alpha=0.75, label=lab, zorder=3)
        if len(pts) > 3:
            xs, ys = zip(*pts)
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            if sxx > 0:
                b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
                gx = [min(xs), max(xs)]
                ax.plot(gx, [my + b * (x - mx) for x in gx], color=col, lw=1.5,
                        alpha=0.9)
    ax.set_xlabel("L1 to the nearest included mixture")
    ax.set_ylabel("profile error  (% $\\Delta$PPL)")
    ax.set_title("Retrieval vs Flow Extrapolation")
    ax.legend(loc="upper left")
    save(fig, out_dir, "extrapolation")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--functional", default=None)
    p.add_argument("--density", default=None, action="append")
    p.add_argument("--mcd", default=None, action="append",
                   help="report_density.py runs carrying three_sample_lead5, one "
                        "per blinded neighbourhood. Feeds the copying figure; "
                        "kept separate from --density so the identification "
                        "figure is not fed eight curves of the same three.")
    p.add_argument("--arms", default=None)
    p.add_argument("--cells", default=None,
                   help="JSON list of per-cell dicts with nn_l1 / gen_err_pct / "
                        "retrieval_pct, as score_generated emits.")
    p.add_argument("--gen", action="append", default=[], metavar="LABEL=PATH")
    p.add_argument("--zoo", default=None)
    p.add_argument("--meta", default=None)
    p.add_argument("--tier", action="append", default=[],
                   metavar="LABEL=PATH",
                   help="report_functional --json_out, one per scale. "
                        "Feeds the two-panel generalisation figure, which "
                        "is the only one that compares tiers.")
    p.add_argument("--out_dir", default="outputs/figs")
    args = p.parse_args()

    fn = read(args.functional)
    tiers = {}
    for spec in args.tier:
        if "=" not in spec:
            print(f"  [skip] --tier {spec!r}: expected LABEL=PATH")
            continue
        lab, path = spec.split("=", 1)
        t = read(path)
        if t:
            tiers[lab] = t
    dens = [d for d in (read(x) for x in (args.density or [])) if d]
    mcds = [d for d in (read(x) for x in (args.mcd or [])) if d]
    arms = read(args.arms)
    cells = read(args.cells)
    gens = {}
    for spec in args.gen:
        if "=" not in spec:
            print(f"  [skip] --gen {spec!r}: expected LABEL=PATH")
            continue
        lab, path = spec.split("=", 1)
        g = read(path)
        if g:
            gens[lab] = g

    print(f"rendering into {args.out_dir}/")
    fig_generalization(tiers, args.out_dir)
    fig_ladder(tiers, fn, args.out_dir)
    fig_dose_response(fn, gens, read(args.zoo), read(args.meta), args.out_dir)
    fig_continuity(fn, gens, args.out_dir)
    fig_arms(arms, args.out_dir)
    fig_identification(dens, args.out_dir)
    fig_memorisation(mcds or dens, args.out_dir)
    fig_extrapolation(collect_cells(fn, cells), args.out_dir)

    print(f"\n{len(MADE)} figure(s) written, {len(SKIPPED)} skipped.")
    for name, why in SKIPPED:
        print(f"  - {name}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

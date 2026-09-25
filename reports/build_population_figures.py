#!/usr/bin/env python3
"""
Assemble reports/population_payload.json from the sealed population artifacts.

PURE STDLIB, like scripts/report_stack.py: it runs on
a laptop against the copied-down JSON under reports/data/population/, so nothing
here ever needs a GPU or a cluster round-trip.

Inputs, per rung (mini_b015, small_b015, medium_b015):
    zoo_geometry.json          displacement + branch-delta cosines (diag_zoo_geometry)
    anova/anova.json           between/within, per tensor and per layer
    geometry/geometry.json     spectrum of the centred domain means
    containment/containment.json   anchor span vs mixture span, debiased
    composition/composition.json   is w_j a pi-weighted blend of the domain experts?
    predictions.json           next-token disagreement          (optional, GPU)
    interpolation_barrier.json linear-interpolation barrier      (optional, GPU)

REPORTING RULES THIS FILE ENFORCES
    - absolute effective_rank is always carried, and k is always stated;
    - block-only sits beside whole-stack, never instead of it;
    - the four cosine pools stay DISJOINT and separately labelled, because
      between_group_mean and pairwise_mean are different populations and combining
      them produces arithmetically impossible numbers.
"""
from __future__ import annotations

import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "population")
# The build() loop `continue`s on a rung with no zoo_geometry.json and no
# anova.json, so a rung may be listed here before its jobs land.
RUNGS = (("mini_b015", "Mini", 51_475_968),
         ("small_b015", "Small", 124_439_808),
         ("medium_b015", "Medium", 354_823_168))
DOMAINS = ("web", "code", "math", "books", "multilingual")
POOLS = ("within_anchor", "between_anchor", "singleton_singleton",
         "anchor_singleton", "all_pairs")


def rd(*parts):
    p = os.path.join(D, *parts)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def named(report, level, name):
    for r in (report or {}).get("rows", []):
        if r.get("level") == level and r.get("name") == name:
            return r
    return None


def region_rows(report, level):
    return [r for r in (report or {}).get("rows", []) if r.get("level") == level]


def summarize(values):
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"n": len(v), "mean": statistics.fmean(v), "median": statistics.median(v),
            "min": min(v), "max": max(v)}


def build():
    payload = {"domains": list(DOMAINS), "simplex_dim": len(DOMAINS) - 1,
               "pools": list(POOLS), "rungs": []}

    for tag, label, d_params in RUNGS:
        geom = rd(tag, "zoo_geometry.json")
        anova = rd(tag, "anova", "anova.json")
        dgeom = rd(tag, "geometry", "geometry.json")
        cont = rd(tag, "containment", "containment.json")
        comp = rd(tag, "composition", "composition.json")
        preds = rd(tag, "predictions.json")
        barrier = rd(tag, "interpolation_barrier.json")
        pireg = rd(tag, "pi_regression.json")
        if geom is None and anova is None:
            continue

        # --- weight-space geometry, and the diversity evidence -------------
        cos = None
        if geom and "cos_raw" in geom:
            cos = {
                "null": geom["cos_null_random_directions"],
                "reference": geom.get("cos_reference"),
                "is_branch_delta": geom.get("cos_is_branch_delta"),
                "n_pairs": geom["cos_raw"]["n_pairs"],
                "raw": {k: geom["cos_raw"][k] for k in POOLS},
                "drift_removed": {k: geom["cos_shared_drift_removed"][k]
                                  for k in POOLS},
                "leading_share_raw": geom["cos_raw"]["leading_eigenvalue_share"],
                "leading_share_drift_removed":
                    geom["cos_shared_drift_removed"]["leading_eigenvalue_share"],
            }

        # --- ANOVA: extra (embeddings) beside blocks, never one alone ------
        a_whole = named(anova, "model", "whole_model")
        a_extra = named(anova, "layer", "extra")
        a_layers = [r for r in region_rows(anova, "layer")
                    if r["name"].startswith("layer_")]
        a_block = None
        if a_layers:
            ssb = sum(r["ss_between"] for r in a_layers)
            ssw = sum(r["ss_within"] for r in a_layers)
            # A Gram is a sum over coordinates, so the region sums of squares are
            # exactly additive: SS_whole = SS_extra + sum(SS_layer). Reconstructing
            # the block region this way needs no second pass.
            # anova.py emits domain_variance_corrected / domain_fraction_corrected on
            # the rows it computes directly, but the blocks region is reconstructed
            # here by summing layer rows, so the correction has to be redone. It is a
            # function of this region's own SS_B and SS_W and is therefore well
            # defined -- but it is NOT additive across regions the way the sums of
            # squares are, so it cannot be summed from the layer rows.
            d_groups, s_reps = 5, 4
            msw = ssw / (d_groups * (s_reps - 1))
            var_c = (ssb - (d_groups - 1) * msw) / (d_groups * s_reps)
            a_block = {"name": "blocks_only", "n_params": sum(r["n_params"] for r in a_layers),
                       "ss_between": ssb, "ss_within": ssw,
                       "ss_total": ssb + ssw,
                       "ms_within": msw,
                       "domain_variance_corrected": var_c,
                       "domain_fraction_corrected":
                           var_c / (var_c + msw) if (var_c + msw) else None,
                       "between_fraction": ssb / (ssb + ssw) if ssb + ssw else None}

        # --- spectrum of the centred DOMAIN MEANS (not the member spectrum) -
        g_whole = named(dgeom, "model", "whole_model")
        g_extra = named(dgeom, "layer", "extra")
        # The prose quotes the token-embedding matrix specifically, which is a
        # TENSOR-level row; `extra` is the layer-level region (wte+wpe+ln_f) and
        # the two differ. Carry both rather than let one stand in for the other.
        g_wte = named(dgeom, "tensor", "transformer.wte.weight")

        # --- per-member mixture entropy -------------------------------------
        # The displacement block already carries max(pi); entropy is the other
        # concentration measure the prose quotes, and corr(||d||, entropy) is
        # NEGATIVE where corr(||d||, max pi) is positive. Sealing it here keeps
        # both derivable from the payload with no zoo access.
        meta = rd(tag, "zoo_meta.json")
        ent = {}
        for mm in (meta or {}).get("members", []):
            h = 0.0
            for q in mm.get("pi", []):
                if q > 0:
                    h -= q * math.log(q)
            ent[mm["idx"]] = h

        # --- containment ---------------------------------------------------
        cont_rows = []
        for r in (cont or {}).get("rows", []):
            ev = r.get("debiased_mixture_eigenvalues") or []
            cont_rows.append({
                "region": r.get("region"), "level": r.get("level"),
                "n_params": r.get("n_params"),
                "simplex_dimension": r.get("simplex_dimension"),
                "debiased_dim_anchor_span": r.get("debiased_dim_anchor_span"),
                "debiased_dim_mixture_span": r.get("debiased_dim_mixture_span"),
                # containment.py prefixes the whole spectrum_above_noise block
                # with "debiased_"; the unprefixed names never appear on a row.
                "mixture_dimension_above_noise":
                    r.get("debiased_mixture_dimension_above_noise"),
                "noise_floor_per_direction":
                    r.get("debiased_noise_floor_per_direction"),
                "nonlinear_beyond_simplex": r.get("nonlinear_beyond_simplex"),
                "debiased_principal_cosines": r.get("debiased_principal_cosines"),
                "debiased_anchor_energy_in_mixture_span":
                    r.get("debiased_anchor_energy_in_mixture_span"),
                "debiased_mixture_energy_in_anchor_span":
                    r.get("debiased_mixture_energy_in_anchor_span"),
                # The RAW pair matters specifically because containment.py:271
                # clips projection energy to [0,1]: at Mini the model-level
                # mixture-in-anchor energy debiases from 0.3186 to exactly
                # 1.0000, so the debiased field cannot show how far the
                # correction overshot. Without the raw value beside it the
                # caveat is unverifiable from copied-down JSON.
                "raw_anchor_energy_in_mixture_span":
                    r.get("raw_anchor_energy_in_mixture_span"),
                "raw_mixture_energy_in_anchor_span":
                    r.get("raw_mixture_energy_in_anchor_span"),
                # Carry the head of the FULL spectrum plus how many entries there
                # were, so a reader can see the tail go negative -- that is what
                # says the noise debiasing bit, rather than that the tail is
                # hiding above the floor.
                "eigenvalues_head": ev[:12],
                "n_eigenvalues": len(ev),
                "n_positive_eigenvalues": sum(1 for v in ev if v > 0),
            })

        # --- composition: model-level rows over the 80 singletons -----------
        c_rows = [r for r in (comp or {}).get("rows", []) if r.get("level") == "model"]
        composition = {f: summarize([r.get(f) for r in c_rows]) for f in
                       ("relative_residual", "residual_to_noise",
                        "perpendicular_fraction", "projection_rank")}

        # --- interior predictive R^2, the analogue the ANOVA cannot give -----
        # Flattened region x design x eval-set. Only r2_lomo and r2_in_sample
        # are carried: the sums of squares are scale-dependent on the region and
        # nothing in the paper quotes them.
        pi_regression = None
        if pireg:
            pi_regression = {
                "n_folds": pireg.get("n_folds"),
                "n_anchor_folds": pireg.get("n_anchor_folds"),
                "n_interior_folds": pireg.get("n_interior_folds"),
                "fold_rule": pireg.get("fold_rule"),
                "pi_geometry": pireg.get("pi_geometry"),
                "disagreement_vs_l1": pireg.get("disagreement_vs_l1"),
                # Carried so the overshoot is checkable offline. Its
                # implied_ceiling_NOT_A_CEILING field lands BELOW the achieved
                # interior R^2, because replicates at a one-hot VERTEX overstate
                # the seed noise of an interior mixture. Do not quote it as a
                # ceiling; the field is named to make that hard to do by accident.
                "within_anchor_reference": pireg.get("within_anchor_reference"),
                "regions": {},
            }
            for rn, rv in (pireg.get("regions") or {}).items():
                cell = {}
                for dn, dv in (rv.get("designs") or {}).items():
                    cell[dn] = {"n_cols": dv.get("n_cols"), "rank": dv.get("rank"),
                                "shuffled_r2_lomo_interior":
                                    dv.get("shuffled_r2_lomo_interior")}
                    for es in ("all", "interior", "anchor", "interior_fit"):
                        if es in dv:
                            cell[dn][es] = {k: dv[es][k] for k in
                                            ("r2_lomo", "r2_in_sample", "n_eval")}
                pi_regression["regions"][rn] = cell

        # --- per-member displacement vs how concentrated its mixture is -----
        # The single number that explains the composition residual: a one-hot
        # anchor travels much further off the trunk than an interior mixture, so
        # a pi-weighted blend of the vertices overshoots every interior member.
        disp = None
        members = (geom or {}).get("gram_member_idx")
        meta = rd(tag, "zoo_meta.json")
        if members and meta:
            pi = {m["idx"]: m["pi"] for m in meta["members"]}
            kind = {m["idx"]: m["kind"] for m in meta["members"]}
            disp = {"member_idx": members,
                    "kind": [kind[i] for i in members],
                    "max_pi": [max(pi[i]) for i in members],
                    "pi_entropy": [ent.get(i) for i in members],
                    # sealed per member by diag_zoo_geometry.py, in this same
                    # order; without it the scatter would need the Gram .npy and
                    # therefore NumPy, breaking this file's stdlib-only contract
                    "to_trunk": (geom or {}).get("to_trunk"),
                    "to_mean": (geom or {}).get("to_mean")}

        payload["rungs"].append({
            "tag": tag, "label": label, "d_params": d_params,
            "beta": (geom or {}).get("beta") or (anova or {}).get("beta"),
            "n_members": (geom or {}).get("n_members"),
            "geom": {k: (geom or {}).get(k) for k in
                     ("disp_rel", "spread_over_disp", "mean_frac",
                      "to_trunk_mean", "trunk_norm", "pairwise_mean",
                      "within_group_mean", "between_group_mean",
                      "between_over_within", "n_within_pairs",
                      "n_between_pairs")},
            "cos": cos,
            "anova": {"whole": a_whole, "extra": a_extra, "blocks": a_block,
                      "layers": a_layers,
                      "df_between": (anova or {}).get("df_between"),
                      "df_within": (anova or {}).get("df_within")},
            "domain_mean_spectrum": {"whole": g_whole, "extra": g_extra,
                                     "wte": g_wte,
                                     "layers": [r for r in region_rows(dgeom, "layer")
                                                if r["name"].startswith("layer_")]},
            "containment": cont_rows,
            "composition": composition,
            # The INTERIOR analogue of the ANOVA: leave-one-mixture-out
            # predictive R^2 of the branch delta on pi. The eta^2 table is
            # measured on the twenty anchors, which sit at the one-hot VERTICES
            # of Delta^4, and is therefore an upper bound on what an interior
            # mixture gives; this measures the interior directly. Computed by
            # scripts/diag_pi_regression.py, which emits JSON precisely so this
            # file never has to open a .npy and can stay stdlib-only.
            "pi_regression": pi_regression,
            "displacement": disp,
            "predictions": preds,
            "barrier": barrier,
        })

    # The scaling axis, from the sealed member spectra rather than from anything
    # computed here. k is 99 and is stated, because effective_rank_ratio is NOT
    # k-invariant: it divides by k while the effective rank barely moves.
    payload["member_spectrum_k99"] = {}
    for tag, label, _d in RUNGS:
        # ladder/blockspec_<slug>_b015.json is where diag_block_spectrum.py sealed
        # these; Small's never landed in its own zoo dir.
        bs = rd("ladder", f"blockspec_{tag.replace('_b015', '')}_b015.json")
        if bs:
            payload["member_spectrum_k99"][tag] = {
                name: {"effective_rank": r.get("spectrum_effective_rank"),
                       "effective_rank_ratio": r.get("spectrum_effective_rank_ratio"),
                       "variance_share": r.get("variance_share_of_whole"),
                       # Carry the WHOLE k-sweep, not just k=99. The effective
                       # rank moves with k while the ratio moves as ~1/k, so a
                       # cross-rung claim is only meaningful read at matched k --
                       # and the "embedding rank is pinned at 4" reading that two
                       # rungs supported is checked by comparing k=24 and k=49
                       # against k=99, which needs all three here.
                       "k_sweep": {int(kk): {
                           "effective_rank": vv.get("effective_rank"),
                           "effective_rank_ratio": vv.get("effective_rank_ratio"),
                           "ev0_over_median": vv.get("ev0_over_median")}
                           for kk, vv in (r.get("k_sweep") or {}).items()}}
                for name, r in (bs.get("regions") or {}).items()}
            # Depth, so effective rank per block is derivable offline: block-only
            # rank rises along the ladder but sublinearly in depth, and that is a
            # statement about rank/n_layers, not about rank.
            payload["member_spectrum_k99"][tag]["n_layers"] = bs.get("n_layers")
            payload["member_spectrum_k99"][tag]["embedding_fraction_of_D"] = \
                bs.get("embedding_fraction_of_D")
    payload["k"] = 99
    return payload


if __name__ == "__main__":
    p = build()
    dest = os.path.join(HERE, "population_payload.json")
    with open(dest, "w") as f:
        json.dump(p, f, indent=1)
    print(f"wrote {dest}  ({os.path.getsize(dest) / 1024:.0f} KB)")
    for r in p["rungs"]:
        c = r["cos"]
        print(f"  {r['label']:<6} D={r['d_params']:>12,}  "
              f"disp {r['geom']['disp_rel']:.5f}  "
              f"b/w {r['geom']['between_over_within']:.3f}")
        if c:
            print(f"         cos within_anchor {c['raw']['within_anchor']:+.4f} -> "
                  f"{c['drift_removed']['within_anchor']:+.4f}   "
                  f"between_anchor {c['raw']['between_anchor']:+.4f} -> "
                  f"{c['drift_removed']['between_anchor']:+.4f}   "
                  f"(null {c['null']:.2e})")

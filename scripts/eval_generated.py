#!/usr/bin/env python
"""
eval_generated.py -- measure what a GENERATED model can actually do.

This is the arm `scripts/report_retrieval.py --gen` has been waiting for, and the
one that decides whether any of this is a result:

    "If the generated model is no better than 'use the model trained on the closest
     mixture we already have,' the flow did not generalize, it memorized and
     retrieved."

Everything upstream of here is a code-space measurement. A code-space statistic can
say the requested mixture controls the generated code; it cannot say the decoded
network is a good language model, and the two come apart (relL2 2e-3
cost +0.378% ppl for a learned reconstruction and ~0.03% for isotropic noise of the same
size). So: sample -> decode -> write into a real GPT-2 -> measure five perplexities
on the same held-out text the zoo was scored on.

WHY THIS IS AFFORDABLE, AND WHY THAT IS THE POINT
-------------------------------------------------
The flow is 51k parameters in a 200 KB checkpoint. Sampling a code is free. The
expensive step is the decode, because DualGramPCA stores no explicit basis -- at
Mini, reconstructing ONE model streams 100 x 206 MB = 20.6 GB of ensemble reads.
Done naively, a 200-model sweep moves 4 TB and looks hung rather than failed.

`inverse_transform_many` already solves this: a whole cohort costs ONE pass, because
`a` promotes from (N,) to (n, N) and the per-chunk contraction becomes a matmul.
Scoring is then ~1.4 s/model (measured: 100 members x 5 domains in 2m22s). So a few hundred generated models is minutes, not days -- which is the
entire reason a generative model beats a lookup table, and the reason this script
sweeps a lattice rather than evaluating four holdout points.

  # the sealed flow, scored at the mixtures it was never trained on
  python scripts/eval_generated.py --run_name zoo_b015_mini_tr92_k46 --k 46 \
      --flow_dir flow_k46_codes_pi --zoo_dir $ARTIFACT_DIR/zoo_b015_mini_n100/gpt2_zoo_mini \
      --targets holdout --n_samples 8 --out $ARTIFACT_DIR/runs/.../gen_ds.json

  # then the verdict
  python scripts/report_retrieval.py --gen flow=.../gen_ds.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from llmzoo.artifacts.bundle import load_run
from llmzoo.artifacts.io import atomic_write_json
from llmzoo.data.mixtures import DOMAINS
from llmzoo.eval.benchmarks import (
    ALL_BENCHMARKS, examples_fingerprint, load_bench_examples,
    load_zoo_tokenizer, score_bench,
)
from llmzoo.eval.core import get_max_context_length
from llmzoo.eval.domains import domain_profile, load_evalsets
from llmzoo.gen.flow import load_flow
from llmzoo.models.registry import build_zoo_model
from llmzoo.models.weight_extractor import build_stack_spec, write_stack_to_model
import llmzoo.wandb_utils as wb


# ---------------------------------------------------------------------------
# Target mixtures
# ---------------------------------------------------------------------------

def dirichlet_lattice(n: int, n_domains: int, seed: int) -> np.ndarray:
    """Dirichlet(1) probes, drawn the same way the zoo's singletons were."""
    return np.random.default_rng(seed).dirichlet(np.ones(n_domains), n)


def path_targets(a: Sequence[float], b: Sequence[float], n: int) -> np.ndarray:
    """
    A straight line across the simplex, for the dose-response figure.

    The point of sampling a path densely is that retrieval CANNOT follow one. Along
    barycentre -> multilingual vertex the nearest trained member changes 4 times in
    100 requests, so lookup is a 5-step staircase and delivers the whole 0.868-nat
    swing in log ppl_multilingual at 4 discontinuities. A conditional generator is
    continuous there or it is not conditional.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    return (1.0 - t) * a + t * b


def resolve_targets(args, zmeta: dict) -> List[dict]:
    """-> [{label, kind, pi}]. `pi` is emitted verbatim: report_retrieval matches
    a generated row to a holdout cell by L1 < 1e-6 against the REQUESTED mixture."""
    P = len(zmeta.get("domains") or DOMAINS)
    members = {int(m["idx"]): m for m in zmeta["members"]}
    out: List[dict] = []

    if args.targets in ("holdout", "all"):
        ho = zmeta.get("holdout") or {}
        for kind in ("interior", "vertex"):
            seen = {}
            for i in ho.get(kind, []):
                if i in members:
                    seen.setdefault(tuple(members[i]["pi"]), []).append(i)
            for pi_t, idxs in seen.items():
                out.append({"label": f"{members[idxs[0]]['mixture_id']}",
                            "kind": f"holdout_{kind}", "pi": list(pi_t)})

    if args.targets in ("train", "all"):
        ho = zmeta.get("holdout") or {}
        seen = {}
        for i in ho.get("train", []):
            if i in members:
                seen.setdefault(tuple(members[i]["pi"]), []).append(i)
        for pi_t, idxs in seen.items():
            out.append({"label": members[idxs[0]]["mixture_id"],
                        "kind": "train", "pi": list(pi_t)})

    if args.targets in ("probe", "all") and args.n_probe > 0:
        for j, pi in enumerate(dirichlet_lattice(args.n_probe, P, args.seed)):
            out.append({"label": f"probe_{j:03d}", "kind": "probe",
                        "pi": [float(v) for v in pi]})

    if args.path is not None:
        if len(args.path) != 2 * P:
            raise SystemExit(
                f"--path needs {2 * P} numbers (start then end on the {P}-simplex), "
                f"got {len(args.path)}")
        a, b = args.path[:P], args.path[P:2 * P]
        for j, pi in enumerate(path_targets(a, b, args.n_path)):
            out.append({"label": f"path_{j:03d}", "kind": "path",
                        "pi": [float(v) for v in pi]})

    if args.pi:
        for j, flat in enumerate(args.pi):
            if len(flat) != P:
                raise SystemExit(f"--pi needs {P} numbers, got {len(flat)}")
            s = float(sum(flat))
            if abs(s - 1.0) > 1e-6:
                raise SystemExit(f"--pi {flat} sums to {s}, not 1")
            out.append({"label": f"pi_{j:03d}", "kind": "explicit",
                        "pi": [float(v) for v in flat]})

    if not out:
        raise SystemExit(
            "no targets. Pass --targets {holdout,train,probe,all}, --path, or --pi.")
    return out


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact_dir", default=None)
    p.add_argument("--run_name", required=True)
    p.add_argument("--k", type=int, required=True,
                   help="Rank. SCALAR, like train_flow.py: a codes-space flow's "
                        "input width IS k, so one flow cannot span ranks.")
    p.add_argument("--source", default="flow",
                   choices=["flow", "pca_only", "quadratic", "gauss", "soup",
                            "trunk", "member"],
                   help="What produces the codes.\n"
                        "  flow      -- sample the conditional flow (the arm)\n"
                        "  member    -- the REAL zoo members at the target mixture, "
                        "loaded straight from w_<i>.npy with no PCA round trip. Not a "
                        "generator at all: it is the ORIGINAL, the thing a generated "
                        "model is supposed to reproduce. Distinct from pca_only, which "
                        "is the same member after a rank-k truncation. --n_samples is "
                        "ignored: a mixture has as many originals as it has members "
                        "(4 at an anchor, 1 at a singleton), and inventing more would "
                        "be resampling the same file.\n"
                        "  pca_only  -- a real member's OWN code, decoded at this "
                        "rank. The RECONSTRUCTION FLOOR: no generative model is "
                        "involved, so whatever perplexity this loses is the cost of "
                        "the truncation alone. Every other arm must be read against "
                        "it, not against the real member.\n"
                        "  quadratic -- closed-form ridge pi -> code fitted on the "
                        "basis members, decoded. The FREE BASELINE any generative "
                        "result has to beat.\n"
                        "  gauss     -- N(mean, diag var) of the training codes, "
                        "ignoring pi. The null.\n"
                        "  soup      -- TASK ARITHMETIC. sum_d pi_d * (mean code of "
                        "the four members at vertex d). No fitting of any kind: the "
                        "coefficients ARE the requested mixture. This is the "
                        "reviewer's first question -- 'is the delta linear in pi?' -- "
                        "and it reads only the 5 anchor groups, so it is the matched "
                        "competitor for an anchors-only flow.\n"
                        "  trunk     -- the shared pretraining trunk itself, scored "
                        "as a model. The 'no branching at all' floor. Emits TWO arms, "
                        "trunk (raw weights) and trunk_proj (projected to rank k and "
                        "decoded); needs --trunk_flat.")
    p.add_argument("--trunk_flat", default=None,
                   help="Flat (D,) .npy of the trunk, for --source trunk. Written by "
                        "scripts/diag_zoo_geometry.py --dump_trunk_flat; already "
                        "present for Mini and Small under "
                        "$ARTIFACT_DIR/population/<scale>_b015/trunk_flat.npy.")
    p.add_argument("--flow_dir", default=None,
                   help="Flow directory RELATIVE to the run root, e.g. "
                        "flow_k46_codes_pi. Taken explicitly because load_run globs "
                        "flow_k<k>_<space> and therefore cannot see a _pi flow. "
                        "Required for --source flow.")
    p.add_argument("--zoo_dir", required=True,
                   help="The zoo whose zoo_meta.json defines the mixtures, the "
                        "holdout and the flattening layout.")
    p.add_argument("--arch", default=None,
                   help="Default: the single arch in the ensemble.")
    p.add_argument("--targets", default="holdout",
                   choices=["holdout", "train", "probe", "all", "none"])
    p.add_argument("--n_probe", type=int, default=0,
                   help="Dirichlet(1) probe mixtures, drawn as the singletons were.")
    p.add_argument("--path", type=float, nargs="+", default=None,
                   help="2P numbers: start simplex point then end point, for the "
                        "dose-response sweep.")
    p.add_argument("--n_path", type=int, default=21)
    p.add_argument("--pi", type=float, nargs="+", action="append", default=None,
                   help="An explicit mixture; repeatable.")
    p.add_argument("--n_samples", type=int, default=8,
                   help="Generated models PER target. Sampling is free; decoding "
                        "and scoring are not, so this is the real cost knob.")
    p.add_argument("--cohort", type=int, default=16,
                   help="Codes decoded per streaming pass over the ensemble. "
                        "Memory is cohort x D x 4 bytes: at Mini D=51.5M so 16 rows "
                        "is 3.3 GB. Raise it on a big-memory node -- it divides the "
                        "number of ensemble passes.")
    p.add_argument("--n_steps", type=int, default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--evalset_seed", type=int, default=7)
    p.add_argument("--evalset_n_docs", type=int, default=64)
    p.add_argument("--evalset_n_ctx", type=int, default=1024)
    p.add_argument("--evalset_cache", default=None,
                   help="Where the held-out slices live. Default $ARTIFACT_DIR.")
    p.add_argument("--arm", default=None,
                   help="Arm name recorded on every row; report_retrieval groups by "
                        "it, so a gauss/quadratic control can share one file.")
    # ---- multiple-choice benchmarks -------------------------------------------
    #
    # Scored on the SAME in-memory model as the five-domain perplexity, immediately
    # after domain_profile, so one PCA decode pass yields both. Off by default: the
    # decode is the expensive part of every existing arm and nothing that already
    # ran should get slower or change shape.
    p.add_argument("--bench", nargs="*", default=None,
                   choices=list(ALL_BENCHMARKS) + ["synthetic"],
                   help="Multiple-choice suites to score beside the perplexity "
                        "profile. Bare --bench means all three. EXPECT NEAR CHANCE: "
                        "at 51-355M and ~20 tokens per parameter these models do not "
                        "move a 4-way suite, and at n questions the accuracy quantum "
                        "is 1/n. The gold-continuation log score sealed "
                        "beside acc is the statistic with the power.")
    p.add_argument("--bench_n_questions", type=int, default=200)
    p.add_argument("--bench_seed", type=int, default=0,
                   help="Fixes WHICH questions. Every model in a comparison must "
                        "share it or the paired log score is not paired; the sealed "
                        "bench_fingerprint is the check.")
    p.add_argument("--bench_n_shots", type=int, default=None,
                   help="Override the few-shot count for mmlu/gpqa (hellaswag is "
                        "0-shot by construction). Worth setting to 0 here: GPT-2's "
                        "n_positions is 1024 and score_choices left-truncates, so a "
                        "nominal 5-shot MMLU prompt is silently 1-2 shot on these "
                        "models.")
    p.add_argument("--bench_max_length", type=int, default=None,
                   help="Default: the model's own max position embeddings.")
    p.add_argument("--out", required=True)
    p.add_argument("--no_wandb", action="store_true")
    wb.add_wandb_args(p)
    args = p.parse_args()

    if args.arm is None:
        args.arm = args.source
    art = args.artifact_dir or os.environ.get("ARTIFACT_DIR", "./artifacts")
    run_root = os.path.join(art, "runs", args.run_name)
    with open(os.path.join(args.zoo_dir, "zoo_meta.json")) as f:
        zmeta = json.load(f)
    domains = list(zmeta.get("domains") or DOMAINS)

    targets = resolve_targets(args, zmeta)

    # --- the trunk is pi-INDEPENDENT, so it is one model, not one per target ----
    #
    # Every target would decode the identical weight vector. At Mini one decode
    # streams 100 x 206 MB = 20.6 GB of ensemble reads, so fanning 85 targets
    # through the cohort loop would move 1.75 TB to compute one number 85 times.
    # Instead: score once, then replicate the profile across every requested
    # mixture at row-assembly time, so the file still joins per-cell downstream.
    report_targets = targets
    if args.source == "trunk":
        if not args.trunk_flat:
            raise SystemExit("--source trunk needs --trunk_flat")
        if args.n_samples != 1:
            print(f"[eval_generated] --source trunk is deterministic; forcing "
                  f"--n_samples 1 (was {args.n_samples})")
            args.n_samples = 1
        bary = [1.0 / len(domains)] * len(domains)
        targets = [{"label": "trunk", "kind": "trunk", "pi": bary}]

    # --- the originals: one row per REAL member at each target mixture ----------
    #
    # Resolved against the whole zoo, not against the PCA basis rows. pca_only
    # deliberately refuses a target the basis never saw, because projecting a sealed
    # member back in would defeat the sealing; but a raw w_<i>.npy is just the file
    # on disk and is defined for every member, held out or not. That is the point of
    # this arm -- it is the ground truth a LOMO cell is scored against.
    member_plan: Optional[List[int]] = None
    if args.source == "member":
        all_members = {int(m["idx"]): m for m in zmeta["members"]}
        member_plan, kept = [], []
        for t in targets:
            hit = [i for i, m in sorted(all_members.items())
                   if float(np.abs(np.asarray(m["pi"], dtype=float)
                                   - np.asarray(t["pi"], dtype=float)).sum()) < 1e-6]
            if not hit:
                raise SystemExit(
                    f"--source member: no zoo member sits at target {t['label']} "
                    f"(pi={t['pi']}). This arm scores the real models at a mixture, "
                    f"so the mixture has to be one the zoo actually contains.")
            for j, i in enumerate(hit):
                member_plan.append(i)
                kept.append({"target": t, "sample": j, "member_idx": i})
        if args.n_samples != 1:
            print(f"[eval_generated] --source member ignores --n_samples "
                  f"({args.n_samples}): a mixture has as many originals as it has "
                  f"members.")
        print(f"[eval_generated] {len(targets)} target mixtures -> "
              f"{len(member_plan)} real members "
              f"({len(member_plan) / max(len(targets), 1):.2f} per mixture)")
        member_rows = kept

    n_models = (len(member_plan) if member_plan is not None
                else len(targets) * args.n_samples)
    print(f"[eval_generated] {len(targets)} target mixtures x {args.n_samples} "
          f"samples = {n_models} models to decode and score")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The ensemble and the PCA are both required: DualGramPCA keeps no basis, so a
    # decode is a weighted sum over the members themselves.
    bundle = load_run(run_root, args.k, want=("dataset", "pca", "codes"),
                      device=str(device))
    arch = args.arch or bundle.arch_list[0]
    pca = bundle.pcas[arch]
    fidx_val = bundle.family_idx(arch)

    fm, fdir = None, None
    if args.source == "flow":
        if not args.flow_dir:
            raise SystemExit("--source flow needs --flow_dir")
        fdir = args.flow_dir if os.path.isabs(args.flow_dir) else os.path.join(
            run_root, args.flow_dir)
        fm = load_flow(fdir, code_stats=bundle.code_stats, device=str(device))
        print(f"[eval_generated] flow {fdir}  cond_mode={fm.cond_mode}  "
              f"dim={fm.space.dim}  trust={fm.trust}")
        if fm.cond_mode != "pi":
            print(f"  !! cond_mode={fm.cond_mode!r}: this flow is UNCONDITIONAL on "
                  f"a single-arch zoo. Every target will draw from the same "
                  f"marginal. That is the control arm, not a result.")
        if fm.space.dim != args.k:
            raise SystemExit(f"flow dim {fm.space.dim} != --k {args.k}")
    else:
        print(f"[eval_generated] source={args.source} (no flow; this is a baseline "
              f"arm sharing the decode and scoring path exactly)")

    evalsets = load_evalsets(args.evalset_cache or art, args.evalset_seed,
                             args.evalset_n_docs, args.evalset_n_ctx, domains)

    model = build_zoo_model(arch, seed=0).to(device)
    _flat0, spec = build_stack_spec(model, arch,
                                    exclude_1d=zmeta["exclude_1d"],
                                    include_extra=zmeta["include_extra"])
    if _flat0.size != int(zmeta["n_params"]):
        raise SystemExit(
            f"spec gives D={_flat0.size:,} but the zoo was flattened at "
            f"D={int(zmeta['n_params']):,}.")

    # ---- multiple-choice questions, loaded ONCE ------------------------------
    #
    # Bare --bench means all three; omitting it means off. Loaded here, before the
    # decode loop, because the question set does not depend on the model and every
    # model in the comparison must see the identical list in the identical order for
    # the paired gold log score to be paired at all.
    bench_list = list(ALL_BENCHMARKS) if args.bench == [] else (args.bench or [])
    bench_examples: Dict[str, list] = {}
    bench_tok = None
    bench_max_length = args.bench_max_length or get_max_context_length(model)
    bench_meta: Optional[dict] = None
    if bench_list:
        if "gpqa" in bench_list and not os.environ.get("HF_TOKEN"):
            raise SystemExit(
                "--bench gpqa needs HF_TOKEN: Idavidrein/gpqa is a gated dataset. "
                "Export it before sbatch, and print ${#HF_TOKEN} rather than the "
                "value if you log it.")
        bench_examples = load_bench_examples(
            bench_list, args.bench_n_questions,
            os.environ.get("HF_HOME") or os.environ.get("HF_DATASETS_CACHE"),
            args.bench_seed, n_shots=args.bench_n_shots)
        bench_tok = load_zoo_tokenizer()
        bench_meta = {
            "benchmarks": bench_list,
            "n_questions": args.bench_n_questions,
            "seed": args.bench_seed,
            "n_shots": args.bench_n_shots,
            "max_length": bench_max_length,
            # A fingerprint per suite, so a report that pairs two files can PROVE
            # they saw the same questions instead of assuming the loaders were
            # deterministic. Same discipline as train_flow.py's pi_fingerprint.
            "fingerprint": {b: examples_fingerprint(ex)
                            for b, ex in bench_examples.items()},
        }
        print(f"[eval_generated] benchmarks {bench_list} at "
              f"n_questions={args.bench_n_questions} seed={args.bench_seed} "
              f"max_length={bench_max_length}; accuracy quantum is "
              f"{1.0 / max(args.bench_n_questions, 1):.4f} -- a delta smaller than a "
              f"few questions is not a measurement.")

    wb.init_run(job_type="eval_generated",
                config={**vars(args), "arch": arch, "n_models": n_models},
                tags=[arch, args.run_name, f"k{args.k}", args.arm],
                enabled=not args.no_wandb, artifact_dir=run_root,
                name_suffix=f"{args.run_name}_k{args.k}_gen",
                group=args.wandb_group, project=args.wandb_project)

    # ---- sample every code up front. This is the free part. ----
    #
    # The source noise is drawn on CPU and moved, rather than passing a generator
    # into flow.sample: torch.randn refuses a CPU generator with a CUDA device, and
    # seeding on the device would make the draw depend on where the job landed.
    # Same pattern as diag_conditioning.py, so the two agree sample for sample.
    n_steps = (args.n_steps or fm.default_steps) if fm is not None else 0
    gs = ((fm.default_guidance if args.guidance_scale is None
           else args.guidance_scale) if fm is not None else 1.0)
    # --source member's plan was built from the zoo, not sampled: its rows are
    # already fixed and there are no codes behind them.
    plan: List[dict] = [] if member_plan is None else member_rows
    codes = np.empty((n_models, args.k), dtype=np.float64)
    real_all = np.asarray(bundle.codes)[:, :args.k]

    # Row -> zoo member index, so a target can be matched back to the member whose
    # code it is (pca_only) and so the quadratic surface can be fitted on pi.
    sel = bundle.ens_meta.get("member_idxs")
    basis_rows = (list(range(real_all.shape[0])) if sel is None
                  else [int(i) for i in sel])
    zmembers = {int(m["idx"]): m for m in zmeta["members"]}
    basis_pi = [zmembers[i]["pi"] for i in basis_rows]

    def _qfeats(P):
        out = []
        for q in P:
            f = list(q) + [q[a] * q[b] for a in range(len(q))
                           for b in range(a, len(q))]
            out.append(f)
        return np.asarray(out, dtype=np.float64)

    if args.source == "quadratic":
        # Closed-form ridge pi -> code on the basis members. No intercept: pi sums
        # to 1 and already spans the constant, so the design is rank-deficient by
        # one and needs the same tiny ridge floor report_conditioning uses.
        A = _qfeats(basis_pi)
        G = A.T @ A + 1e-9 * float(np.trace(A.T @ A)) / A.shape[1] * np.eye(A.shape[1])
        Bq = np.linalg.solve(G, A.T @ real_all)
    if args.source == "gauss":
        gmu, gsd = real_all.mean(axis=0), real_all.std(axis=0, ddof=1)

    # --- the soup: task arithmetic over the five vertex means ---------------
    #
    # w_hat(pi) = w_trunk + sum_d pi_d (w_bar_d - w_trunk) = sum_d pi_d w_bar_d,
    # because pi sums to 1 -- THE TRUNK CANCELS. So the prediction needs only the
    # five anchor group means and no trunk at all.
    #
    # Done in CODE space on purpose. The decode is linear, so mixing codes and then
    # decoding equals mixing the rank-k reconstructions; the code-space soup is the
    # weight-space soup minus the truncation residual. That keeps the soup on the
    # SAME truncation footing as the flow, which is the only fair comparison. The
    # raw weight-space version is already measured, at all three scales, by
    # src/llmzoo/analysis/mixture_composition/composition.py.
    #
    # Free correctness check: at a vertex the soup returns that vertex's group mean,
    # so its dPPL there must land near the replicate ceiling (~1.01% at Mini). If it
    # does not, this arm is wrong and nothing else in the file is worth reading.
    # --- the trunk, raw and projected ---------------------------------------
    #
    # The trunk is NOT an ensemble member and is not in the span of the centred
    # zoo, so it has no code. Two readings, and the gap between them is itself the
    # number worth having -- it says how much of the trunk this basis can represent
    # at all, which is the transductivity limitation measured rather than asserted:
    #   trunk       raw (D,) written straight into the model, no PCA involved
    #   trunk_proj  projected to rank k and decoded, i.e. the trunk as the basis
    #               sees it, on the same truncation footing as every other arm
    trunk_flat = None
    if args.source == "trunk":
        trunk_flat = np.load(args.trunk_flat).astype(np.float32, copy=False)
        if trunk_flat.size != int(zmeta["n_params"]):
            raise SystemExit(
                f"--trunk_flat has D={trunk_flat.size:,} but the zoo was flattened "
                f"at D={int(zmeta['n_params']):,}. Wrong scale's trunk?")
        print(f"[eval_generated] trunk {args.trunk_flat}  D={trunk_flat.size:,}")

    anchor_means, soup_exclusion = None, {}
    if args.source == "soup":
        rows_by_mix: Dict[str, List[int]] = {}
        for r, midx in enumerate(basis_rows):
            m = zmembers.get(midx)
            if m is not None:
                rows_by_mix.setdefault(str(m["mixture_id"]), []).append(r)
        anchor_means = np.empty((len(domains), args.k), dtype=np.float64)
        missing = []
        for d, dom in enumerate(domains):
            rws = rows_by_mix.get(f"anchor_{dom}", [])
            if not rws:
                missing.append(dom)
                continue
            anchor_means[d] = real_all[rws].mean(axis=0)
        if missing:
            raise SystemExit(
                f"--source soup needs all {len(domains)} anchor groups in the "
                f"basis; missing anchor_{{{','.join(missing)}}}. This basis has "
                f"mixtures {sorted(rows_by_mix)[:8]}...")
        print(f"[eval_generated] soup: {len(domains)} vertex means over "
              + ", ".join(f"anchor_{d}={len(rows_by_mix[f'anchor_{d}'])} rows"
                          for d in domains))

        # MATCHED BLINDING, and this is the part that decides whether the soup
        # comparison means anything. The soup reads ONLY the 20 anchor rows. With
        # flow_exclusion left empty, report_functional falls back to the zoo's
        # holdout["train"] -- 92 members -- and would score a 20-member predictor
        # against a 92-member lookup table. So the ball the soup genuinely honours
        # is declared here.
        #
        # The radius is DERIVED, not hardcoded: L1-to-barycentre is maximal at the
        # vertices, so any radius between the most extreme non-anchor and the
        # anchors themselves selects the anchors by complement. Taking the midpoint
        # keeps it correct if the mixture plan is ever redrawn.
        anchor_rows = sorted(r for d in domains
                             for r in rows_by_mix[f"anchor_{d}"])
        other_rows = [r for r in range(real_all.shape[0]) if r not in set(anchor_rows)]
        bary_v = np.full(len(domains), 1.0 / len(domains))
        d_bary = np.abs(np.asarray(basis_pi, dtype=np.float64) - bary_v).sum(axis=1)
        lo_anchor = float(d_bary[anchor_rows].min())
        hi_other = float(d_bary[other_rows].max()) if other_rows else 0.0
        if hi_other >= lo_anchor:
            raise SystemExit(
                f"--source soup: cannot separate anchors by L1 to the barycentre "
                f"(most extreme non-anchor {hi_other:.4f} >= nearest anchor "
                f"{lo_anchor:.4f}). The blinding ball would be wrong, so refusing "
                f"rather than reporting a rigged comparison.")
        soup_radius = 0.5 * (hi_other + lo_anchor)
        soup_exclusion = {
            "center_pi": [float(x) for x in bary_v],
            "exclude_within_l1": soup_radius,
            "excluded_rows": other_rows,
            "n_rows": len(anchor_rows),
        }
        print(f"[eval_generated] soup blinding: barycentre ball at L1 "
              f"{soup_radius:.4f} (non-anchor max {hi_other:.4f}, anchor min "
              f"{lo_anchor:.4f}); retrieval will keep {len(anchor_rows)} rows, "
              f"excluding {len(other_rows)}")

    row = 0
    for ti, t in enumerate(targets if member_plan is None else []):
        rng = np.random.default_rng(args.seed + ti)
        if args.source == "flow":
            g = torch.Generator(device="cpu").manual_seed(args.seed + ti)
            fv = torch.full((args.n_samples,), fidx_val, dtype=torch.long,
                            device=device)
            kw = {}
            if fm.cond_mode == "pi":
                kw["pi"] = torch.tensor(t["pi"], dtype=torch.float32,
                                        device=device).reshape(1, -1).expand(
                                            args.n_samples, -1).contiguous()
            with torch.no_grad():
                x0 = (torch.randn(args.n_samples, fm.space.dim,
                                  generator=g).to(device) * fm.flow.source_std)
                x = fm.flow.sample(args.n_samples, fv, n_steps=n_steps,
                                   guidance_scale=gs, x0=x0, **kw)
                # from_flow is the denormalisation: flow space is unit-RMS per
                # family, inverse_transform wants raw PCA codes.
                c = fm.space.from_flow(x, fv)
            c = c.double().cpu().numpy()
        elif args.source == "pca_only":
            # The member's OWN code at this rank. Only defined for a target whose
            # mixture is IN the basis -- a sealed member is absent from it by
            # construction, and projecting it back in would defeat the sealing.
            hit = [r for r, i in enumerate(basis_rows)
                   if float(np.abs(np.asarray(zmembers[i]["pi"])
                                   - np.asarray(t["pi"])).sum()) < 1e-6]
            if not hit:
                raise SystemExit(
                    f"--source pca_only: target {t['label']} (pi={t['pi']}) is not "
                    f"a basis member, so it has no code to reconstruct. The "
                    f"reconstruction floor is only defined on members the basis was "
                    f"fitted on; use --targets train.")
            c = np.stack([real_all[hit[j % len(hit)]]
                          for j in range(args.n_samples)])
        elif args.source == "quadratic":
            c = np.tile(_qfeats([t["pi"]]) @ Bq, (args.n_samples, 1))
        elif args.source == "soup":
            # sum_d pi_d * vertex_mean_d. Deterministic, so --n_samples 1 suffices.
            c = np.tile(np.asarray(t["pi"], dtype=np.float64) @ anchor_means,
                        (args.n_samples, 1))
        elif args.source == "trunk":
            # One streaming pass. The cohort loop below then decodes it back, so
            # what gets scored is trunk_proj; the raw trunk is scored separately.
            c = np.tile(pca.transform_vector(trunk_flat, bundle.dataset, arch,
                                             k=args.k).reshape(1, -1),
                        (args.n_samples, 1))
        else:  # gauss
            c = gmu + rng.normal(size=(args.n_samples, args.k)) * gsd

        for j in range(args.n_samples):
            codes[row] = c[j]
            plan.append({"target": t, "sample": j})
            row += 1

    # ---- dispersion, read BEFORE any perplexity ----
    #
    # A generator collapsed toward the family mean scores dPPL ~ 0 -- BETTER than an
    # honest sample -- on a mean-centred ensemble. So this has to be
    # read first. But it has to be read with the RIGHT denominator:
    #
    #   Under pi-conditioning the pooled ratio INVERTS. Samples drawn at one pi
    #   carry the CONDITIONAL spread; the real codes carry the POOLED spread. At
    #   Mini pooled/within = 3.07, so a perfectly conditional flow reads ~0.33
    #   pooled -- numerically identical to the ratio a genuinely collapsed
    #   generator reads. Comparing generated-at-pi against all real codes and
    #   calling the result "collapsed" is therefore a trap.
    #
    # So both are emitted, as train_flow.py does, and only the second is a verdict:
    #   pooled   -- expected to read ~ within/pooled for a correct pi-flow, NOT 1
    #   at_pi    -- generated spread at one mixture against the REAL within-mixture
    #               spread. This one reads ~1 when the conditional width is right.
    # --source member has no generated codes to disperse: the rows ARE the real
    # members, so every ratio below would be the population against itself.
    real = np.asarray(bundle.codes)[:, :args.k]
    real_pooled_rms = float(np.sqrt(real.var(axis=0, ddof=1).sum()))

    # Real within-mixture spread, pooled over every replicate group in the basis.
    sel = bundle.ens_meta.get("member_idxs")
    basis_rows = (list(range(real.shape[0])) if sel is None else [int(i) for i in sel])
    zmembers = {int(m["idx"]): m for m in zmeta["members"]}
    by_mix: Dict[str, List[int]] = {}
    for r, midx in enumerate(basis_rows):
        m = zmembers.get(midx)
        if m is not None:
            by_mix.setdefault(m["mixture_id"], []).append(r)
    ss = np.zeros(args.k)
    dof = 0
    for rws in by_mix.values():
        if len(rws) > 1:
            blk = real[rws]
            ss += ((blk - blk.mean(axis=0)) ** 2).sum(axis=0)
            dof += len(rws) - 1
    real_within_rms = float(np.sqrt((ss / dof).sum())) if dof else float("nan")

    # A deterministic arm (soup, quadratic, trunk) has no dispersion to read, and
    # with one row ddof=1 is a divide-by-zero. Report nan rather than a warning:
    # "this arm has no conditional width" is the honest answer, not a defect.
    expected_pooled = (real_within_rms / real_pooled_rms
                       if real_pooled_rms > 0 else float("nan"))
    if member_plan is not None:
        rms_ratio = at_pi = float("nan")
        print(f"[eval_generated] dispersion: not defined for --source member "
              f"(real within/pooled = {expected_pooled:.3f} is the population's own "
              f"number; these rows ARE the population)")
    else:
        gen_pooled_rms = (float(np.sqrt(codes.var(axis=0, ddof=1).sum()))
                          if n_models > 1 else float("nan"))
        within = codes.reshape(len(targets), args.n_samples, args.k)
        gen_within_rms = (
            float(np.sqrt((((within - within.mean(1, keepdims=True)) ** 2).sum(axis=2)
                           ).sum() / (len(targets) * (args.n_samples - 1))))
            if args.n_samples > 1 else float("nan"))

        rms_ratio = gen_pooled_rms / max(real_pooled_rms, 1e-12)
        at_pi = (gen_within_rms / max(real_within_rms, 1e-12)
                 if real_within_rms == real_within_rms else float("nan"))
        print(f"[eval_generated] dispersion (real within/pooled = "
              f"{expected_pooled:.3f}, {dof} dof over {sum(1 for v in by_mix.values() if len(v) > 1)} groups)")
        print(f"    pooled ratio  {rms_ratio:.4f}   "
              f"(a correct pi-flow reads ~{expected_pooled:.2f} here, NOT 1)")
        print(f"    at_pi  ratio  {at_pi:.4f}   "
              f"(this is the one that should read ~1)")
        if at_pi == at_pi and at_pi < 0.5:
            print("  !! CONDITIONAL COLLAPSE: generated spread at a fixed mixture is "
                  "far below real within-mixture variation. A mixture of memorised "
                  "members would look like this. Do not quote "
                  "any dPPL below at face value.")

    # ---- decode in cohorts: one ensemble pass each, not one per model ----
    rows: List[dict] = []
    t0 = time.time()
    for lo in range(0, n_models, args.cohort):
        hi = min(lo + args.cohort, n_models)
        if member_plan is not None:
            # No PCA involved: these are the originals, read off disk.
            W = np.stack([np.load(os.path.join(args.zoo_dir, f"w_{i}.npy"))
                          for i in member_plan[lo:hi]])
        else:
            W = pca.inverse_transform_many(codes[lo:hi], bundle.dataset, arch,
                                           k=args.k)
        for r in range(hi - lo):
            write_stack_to_model(W[r], model, arch, spec)
            ppl = domain_profile(model, evalsets, device, args.batch_size, domains)
            entry = plan[lo + r]
            t = entry["target"]
            out_row = {"idx": lo + r, "arm": args.arm, "label": t["label"],
                       "kind": t["kind"], "sample": entry["sample"],
                       "pi": t["pi"], "ppl": ppl}
            if "member_idx" in entry:
                out_row["member_idx"] = entry["member_idx"]
            if bench_examples:
                # Same model, same weights, still resident. The decode is what
                # costs; the questions are the cheap part of this loop.
                out_row["bench"] = score_bench(model, bench_tok, bench_examples,
                                               device, bench_max_length)
            rows.append(out_row)
            print(f"  {lo + r:>4}/{n_models} {t['label']:<22} "
                  + "  ".join(f"{d[:4]}={ppl[d]:8.2f}" for d in domains)
                  + ("  " + "  ".join(
                      f"{b[:4]}={out_row['bench'][b]['acc']:.3f}"
                      for b in out_row["bench"]) if bench_examples else ""),
                  flush=True)
            wb.log({**{f"gen_ppl/{d}": ppl[d] for d in domains},
                    "model_idx": lo + r}, step=lo + r)
        del W
    dt = time.time() - t0
    print(f"[eval_generated] {n_models} models in {dt:.0f}s "
          f"({dt / max(n_models, 1):.2f}s each)")

    # ---- trunk: score the RAW weights too, then fan both out over the targets --
    #
    # Two arms in one file, which report_functional groups by row["arm"]. Legal and
    # intended -- --arm's own help says a control can share a file. The fan-out is
    # what lets the pi-independent trunk be compared cell-by-cell against arms that
    # do depend on pi; every target gets the same profile, which is the point.
    if args.source == "trunk":
        ppl_proj = rows[0]["ppl"]
        bench_proj = rows[0].get("bench")
        write_stack_to_model(trunk_flat, model, arch, spec)
        ppl_raw = domain_profile(model, evalsets, device, args.batch_size, domains)
        bench_raw = (score_bench(model, bench_tok, bench_examples, device,
                                 bench_max_length) if bench_examples else None)
        print("[eval_generated] trunk raw  "
              + "  ".join(f"{d[:4]}={ppl_raw[d]:8.2f}" for d in domains))
        print("[eval_generated] trunk proj "
              + "  ".join(f"{d[:4]}={ppl_proj[d]:8.2f}" for d in domains))
        rows = []
        for ti, t in enumerate(report_targets):
            for aname, ppl, bch in (("trunk", ppl_raw, bench_raw),
                                    ("trunk_proj", ppl_proj, bench_proj)):
                r = {"idx": len(rows), "arm": aname, "label": t["label"],
                     "kind": t["kind"], "sample": 0,
                     "pi": t["pi"], "ppl": dict(ppl)}
                if bch is not None:
                    # Deep-copied per row for the same reason ppl is: the fan-out
                    # writes 2 x n_targets rows and a shared mutable would make
                    # every cell alias one dict.
                    r["bench"] = json.loads(json.dumps(bch))
                rows.append(r)
        print(f"[eval_generated] fanned 2 profiles over "
              f"{len(report_targets)} target mixtures -> {len(rows)} rows")

    report = {
        "arch": arch, "run_name": args.run_name, "k": args.k, "arm": args.arm,
        "source": args.source, "flow_dir": fdir,
        "cond_mode": (fm.cond_mode if fm is not None else args.source),
        "domains": domains,
        "n_targets": len(report_targets), "n_samples": args.n_samples,
        "n_evaluated": len(rows),
        "bench": bench_meta,
        "code_rms_ratio": rms_ratio,
        "code_rms_ratio_at_fixed_pi": at_pi,
        "code_rms_reference": "pooled_at_requested_pi",
        "real_within_over_pooled": expected_pooled,
        "within_dof": dof,
        "n_steps": n_steps, "guidance_scale": gs,
        "seed": args.seed,
        "evalset": {"seed": args.evalset_seed, "n_docs": args.evalset_n_docs,
                    "n_ctx": args.evalset_n_ctx},
        "zoo_dir": args.zoo_dir,
        "flow_provenance": (fm.meta.get("provenance", {}) if fm is not None
                            else {}),
        # WHAT THE FLOW WAS BLINDED TO, carried through so the scorer can blind the
        # BASELINES to exactly the same thing. Comparing a flow that never saw a
        # neighbourhood of pi against a retrieval baseline that did is not a
        # comparison, and nothing downstream can reconstruct the exclusion ball from
        # the rows alone.
        # The soup declares its OWN ball (it reads only the anchors); the trunk
        # reads no members at all, so it is a floor and gets no ball -- a matched
        # retrieval comparison against a predictor that consulted nothing would be
        # meaningless, and report_functional's train/holdout fallback is the right
        # reference there.
        "flow_exclusion": ({
            "center_pi": (fm.meta.get("train", {}) or {}).get("center_pi"),
            "exclude_within_l1": (fm.meta.get("train", {}) or {}).get(
                "exclude_within_l1"),
            "excluded_rows": (fm.meta.get("train", {}) or {}).get(
                "excluded_rows", []),
            "n_rows": (fm.meta.get("train", {}) or {}).get("n_rows"),
        } if fm is not None
            else soup_exclusion if args.source == "soup" else {}),
        "rows": rows,
    }
    atomic_write_json(args.out, report)
    print(f"\n[eval_generated] -> {args.out}")
    print(f"  next: python scripts/report_retrieval.py "
          f"--gen {args.arm}={args.out}")
    wb.summary({"gen/n_evaluated": len(rows), "gen/code_rms_ratio": rms_ratio,
                "gen/code_rms_ratio_at_fixed_pi": at_pi, "gen/seconds": dt})
    wb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())

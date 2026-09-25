#!/usr/bin/env python3
"""Run the imported population analyses over one zoo, with N=100-safe flags.

    python scripts/run_population_analysis.py \
        --zoo_dir $ARTIFACT_DIR/zoo_b015_mini_n100/gpt2_zoo_mini \
        --out_dir $ARTIFACT_DIR/population/mini_b015

`src/llmzoo/analysis/` came from a collaborator as five standalone folders with
their own argparse and their own (byte-identical, and deliberately so) copy of
`zoo.py`. They are NOT refactored onto the repo's loaders: the point of keeping
them standalone is that they can be handed back, and diffed against what was
handed over. This driver is the seam -- it knows the zoo layout, the N=100-safe
flag values, and nothing else.

WHY EACH FLAG IS WHAT IT IS
---------------------------
--reference-checkpoint trunk_flat.npy   composition.py otherwise runs with no
    reference at all (it no longer auto-loads trunk.pt, which imported torch
    without asking). Produce the flat trunk ONCE with
    `scripts/diag_zoo_geometry.py --dump_trunk_flat`; it is written through the
    same flatten_model path that wrote every w_i.npy, so the layout matches by
    construction rather than by hope.

--members (all singletons)              composition's default is "all", but
    commd.txt's examples pass `20 21 22 23` -- the four singletons of the
    RANDOM CONTROL. On a real zoo the singletons are 20..99 and that literal
    would silently analyse 4 of 80.

--scope cross-domain                    weight_embeddings with `--include-mixtures`
    and `--scope within-domain` builds one subplot per mixture: at 85 distinct pi
    that is figsize=(425, 4), i.e. a 425-INCH canvas.

STAGES
------
Everything below `--stage weights` is NumPy-only and runs on the `cpu` partition.
The torch import is confined to the one-off trunk flatten, which is why it is a
separate stage you run once per zoo.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ANALYSIS = os.path.join(REPO, "src", "llmzoo", "analysis")


def run(cmd, log_dir, tag):
    """Run one analysis, tee its log, and keep going if it fails.

    A driver that dies on the first failure would throw away the three analyses
    that did work -- and the whole pass over a 100 x 124M zoo that produced them.
    """
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, f"{tag}.log")
    print(f"\n{'=' * 70}\n[{tag}] {' '.join(cmd)}\n{'=' * 70}", flush=True)
    # The analysis folders import `llmzoo.analysis.<pkg>.zoo`, but none of them
    # carries an __init__.py and `[tool.setuptools.packages.find]` only finds
    # packages that do -- so whether the import resolves depends on which
    # editable-install strategy pip happened to pick. Putting src/ on the path
    # makes it resolve the same way every time, as a namespace package.
    env = dict(os.environ)
    src = os.path.join(REPO, "src")
    env["PYTHONPATH"] = (src + os.pathsep + env["PYTHONPATH"]
                         if env.get("PYTHONPATH") else src)
    t0 = time.time()
    # Tee LINE BY LINE rather than subprocess.run(capture_output). A pass over a
    # 100 x 124M zoo is tens of minutes; buffering until exit means the slurm log
    # stays empty for the whole run and a stall is indistinguishable from work.
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            f.write(line)
            f.flush()
            print(line, end="", flush=True)
        proc.wait()
    dt = time.time() - t0
    status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
    print(f"[{tag}] {status}  {dt / 60:.1f} min  -> {log}", flush=True)
    return {"tag": tag, "cmd": cmd, "returncode": proc.returncode,
            "seconds": dt, "log": log}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zoo_dir", required=True, help="the ARCH-level dir, the one "
                                                    "holding zoo_meta.json and w_*.npy")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--stage", default="weights",
                   choices=("trunk", "weights", "all"),
                   help="trunk = flatten trunk.pt + seal the displacement Gram "
                        "(needs torch); weights = the four NumPy-only analyses; "
                        "all = both, in order")
    p.add_argument("--chunk_size", type=int, default=1 << 20,
                   help="elements per streaming block inside the analyses. The "
                        "collaborator's 65536 default is ~50x more Python-level "
                        "iterations than a 124M zoo needs.")
    p.add_argument("--skip", nargs="*", default=[],
                   help="analysis tags to skip: anova geometry containment "
                        "composition embeddings")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    meta_path = os.path.join(args.zoo_dir, "zoo_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    members = meta["members"]
    singletons = [m["idx"] for m in members if m["kind"] == "singleton"]
    anchors = [m["idx"] for m in members if m["kind"] == "anchor"]
    print(f"zoo      {args.zoo_dir}")
    print(f"arch     {meta['arch']}  beta={meta.get('beta')}  D={meta['n_params']:,}")
    print(f"members  {len(members)} = {len(anchors)} anchor + {len(singletons)} singleton")
    print(f"out      {args.out_dir}")

    if not singletons:
        print("\nNo singletons: mixture_composition and subspace_containment both "
              "need mixtures and would be skipped. This is the N=12 beta-calibration "
              "shape, not an N=100 zoo.", file=sys.stderr)

    os.makedirs(args.out_dir, exist_ok=True)
    logs = os.path.join(args.out_dir, "logs")
    trunk_flat = os.path.join(args.out_dir, "trunk_flat.npy")
    results = []

    py = sys.executable
    cs = str(args.chunk_size)

    if args.stage in ("trunk", "all"):
        cmd = [py, "-u", os.path.join(HERE, "diag_zoo_geometry.py"),
               "--zoo_dir", args.zoo_dir,
               "--dump_trunk_flat", trunk_flat,
               "--dump_gram", os.path.join(args.out_dir, "displacement_gram.npy"),
               "--json_out", os.path.join(args.out_dir, "zoo_geometry.json")]
        if args.dry_run:
            print(" ".join(cmd))
        else:
            results.append(run(cmd, logs, "geometry_gram"))

    if args.stage in ("weights", "all"):
        # composition's open_weights() reads zoo_meta.json from the reference's
        # OWN parent, so the flat trunk needs one beside it.
        side = os.path.join(args.out_dir, "zoo_meta.json")
        if os.path.exists(trunk_flat) and not os.path.exists(side) and not args.dry_run:
            with open(side, "w") as f:
                json.dump(meta, f)

        jobs = [
            ("anova", [py, "-u", os.path.join(ANALYSIS, "anovas", "anova.py"),
                       "--zoo-dir", args.zoo_dir, "--chunk-size", cs,
                       "--output-dir", os.path.join(args.out_dir, "anova")]),
            ("geometry", [py, "-u", os.path.join(ANALYSIS, "domain_geometry", "geometry.py"),
                          "--zoo-dir", args.zoo_dir, "--chunk-size", cs,
                          "--output-dir", os.path.join(args.out_dir, "geometry")]),
            ("containment", [py, "-u", os.path.join(ANALYSIS, "subspace_containment", "containment.py"),
                             "--zoo-dir", args.zoo_dir, "--chunk-size", cs,
                             "--output-dir", os.path.join(args.out_dir, "containment")]),
        ]
        comp = [py, "-u", os.path.join(ANALYSIS, "mixture_composition", "composition.py"),
                "--zoo-dir", args.zoo_dir, "--chunk-size", cs,
                "--output-dir", os.path.join(args.out_dir, "composition")]
        if singletons:
            comp += ["--members"] + [str(i) for i in singletons]
        if os.path.exists(trunk_flat) or args.dry_run:
            comp += ["--reference-checkpoint", trunk_flat]
        else:
            print("\nNOTE: no trunk_flat.npy -- run `--stage trunk` first, or "
                  "composition's relative_residual will be null.", file=sys.stderr)
        jobs.append(("composition", comp))

        jobs.append(("embeddings", [
            py, "-u", os.path.join(ANALYSIS, "Embeddings", "weight_embeddings.py"),
            # NB: weight_embeddings.py does NOT take --zoo-dir. Its parser is
            # its own, not zoo.py's add_arguments, and the alias is --zoo-path.
            "--zoo-path", args.zoo_dir, "--chunk-size", cs,
            "--include-mixtures", "--scope", "cross-domain",
            "--per-layer", "--method", "both",
            "--output-dir", os.path.join(args.out_dir, "weight_embeddings")]))

        # Interior predictive R^2. It must run AFTER weight_embeddings, whose
        # per-region centered_gram.npy files it reads -- region Grams are a free
        # by-product of that job, so this one needs no weight pass of its own.
        # Falls back to streaming w_*.npy when they are absent.
        we_dir = os.path.join(args.out_dir, "weight_embeddings")
        pireg = [py, "-u", os.path.join(REPO, "scripts", "diag_pi_regression.py"),
                 "--meta", os.path.join(args.zoo_dir, "zoo_meta.json"),
                 "--json_out", os.path.join(args.out_dir, "pi_regression.json")]
        if os.path.exists(os.path.join(we_dir, "centered_gram.npy")) or args.dry_run:
            pireg += ["--gram_dir", we_dir]
        else:
            pireg += ["--zoo_dir", args.zoo_dir]
        for opt, path in (("--geometry", os.path.join(args.out_dir, "zoo_geometry.json")),
                          ("--disagreement", os.path.join(args.out_dir, "disagreement.npy"))):
            if os.path.exists(path) or args.dry_run:
                pireg += [opt, path]
        jobs.append(("pi_regression", pireg))

        for tag, cmd in jobs:
            if tag in args.skip:
                print(f"[{tag}] skipped")
                continue
            if args.dry_run:
                print(" ".join(cmd))
            else:
                results.append(run(cmd, logs, tag))

    if args.dry_run:
        return 0

    manifest = os.path.join(args.out_dir, "driver_manifest.json")
    with open(manifest, "w") as f:
        json.dump({"zoo_dir": args.zoo_dir, "arch": meta["arch"],
                   "beta": meta.get("beta"), "n_params": meta["n_params"],
                   "n_members": len(members), "chunk_size": args.chunk_size,
                   "stage": args.stage, "results": results}, f, indent=2)
    print(f"\n{'=' * 70}")
    for r in results:
        print(f"  {r['tag']:<16}{'ok' if not r['returncode'] else 'FAILED':<8}"
              f"{r['seconds'] / 60:>7.1f} min")
    print(f"-> {manifest}")
    bad = [r["tag"] for r in results if r["returncode"]]
    if bad:
        print(f"\nFAILED: {', '.join(bad)} -- see {logs}/", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
merge_gen.py -- combine per-arm eval_generated outputs into one scorable file.

PURE STDLIB. A leave-one-mixture-out design trains one flow per held-out mixture and
therefore produces one eval_generated JSON per CELL. report_functional.py scores an
`--gen` file by grouping its rows on `arm`, so the 85 (or 12) files of one
CONFIGURATION have to become one file with one arm label before they can be compared
against another configuration.

  python scripts/merge_gen.py --label c05 --out merged_c05.json gen/c05_lomo_*.json

WHY THE EXCLUSION BALL IS DROPPED, AND WHEN THAT IS SAFE
--------------------------------------------------------
Each input carries `flow_exclusion`, the neighbourhood its flow was blinded to, and
report_functional uses it to blind RETRIEVAL to the same neighbourhood. Merged files
come from different balls, so no single ball describes the result and this script
refuses to invent one.

That is harmless for leave-one-mixture-out and ONLY for it: at radius ~0 the ball is
the target mixture itself, and score_generated already excludes the target from the
retrieval pool at every cell (`l1(m.pi, cell.pi) > 1e-6`). So the blinding is
enforced cell-by-cell whether or not the ball is carried.

It is NOT harmless for the extrapolation-radius sweep, where the ball is the whole
point. This script refuses to merge those: mixing radii would silently compare a
flow blinded to a 0.4-wide ball against retrieval that could see into it.

THE ONE CASE WHERE A WIDE BALL *IS* SAFE TO MERGE: SHARDS OF ONE FLOW
--------------------------------------------------------------------
The refusal above is about merging files from DIFFERENT balls, where no single ball
describes the result. A file set produced by sharding ONE flow's targets across an
array is the opposite situation -- every shard carries the IDENTICAL ball, because
it came from the identical checkpoint. Preserving that ball invents nothing; it is
the ball, and dropping it would silently hand retrieval the whole zoo while the flow
saw a fifth of it.

So: if every input agrees on (center_pi, exclude_within_l1), the ball is carried
through and --max_radius is not consulted. If they disagree, the old behaviour
stands -- blank the ball and refuse anything wider than --max_radius. Sharding a
20-row anchors-only flow over 9 array tasks is why this exists; a 2-hour
unbackfillable single job was the alternative.
"""
from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True, help="arm name for every merged row")
    p.add_argument("--out", required=True)
    p.add_argument("--max_radius", type=float, default=1e-3,
                   help="Refuse any input whose exclusion ball is wider than this. "
                        "The default admits leave-one-mixture-out (1e-6) and "
                        "nothing else.")
    p.add_argument("inputs", nargs="+")
    args = p.parse_args()

    # Read every ball FIRST, so "do these all agree?" can be answered before
    # deciding whether a wide radius is a shard set or a mixed-radius mistake.
    balls = []
    for path in args.inputs:
        with open(path) as f:
            balls.append((path, (json.load(f).get("flow_exclusion") or {})))
    keys = {(json.dumps(b.get("center_pi"), sort_keys=True),
             b.get("exclude_within_l1")) for _, b in balls}
    one_ball = len(keys) == 1 and next(iter(keys))[1] is not None
    shared: dict = {}
    if one_ball:
        shared = next(b for _, b in balls if b.get("exclude_within_l1") is not None)
        print(f"one shared exclusion ball across all {len(args.inputs)} input(s) "
              f"(L1 {float(shared['exclude_within_l1']):g}, "
              f"{len(shared.get('excluded_rows') or [])} rows excluded) -- carried "
              f"through, so retrieval stays blinded to it downstream.")

    base, rows, radii, cells = None, [], set(), set()
    for path in args.inputs:
        with open(path) as f:
            d = json.load(f)
        r = (d.get("flow_exclusion") or {}).get("exclude_within_l1")
        if r is not None:
            radii.add(float(r))
            if float(r) > args.max_radius and not one_ball:
                raise SystemExit(
                    f"{path}: exclusion radius {float(r):g} exceeds --max_radius "
                    f"{args.max_radius:g} and the inputs do NOT share one ball "
                    f"({len(keys)} distinct). Merging arms with different exclusion "
                    f"balls would compare a blinded flow against an unblinded "
                    f"retrieval baseline. Score those files one at a time.")
        if base is None:
            base = {k: v for k, v in d.items() if k != "rows"}
        elif base.get("domains") != d.get("domains"):
            raise SystemExit(f"{path}: domains differ from the first input")
        for row in d["rows"]:
            row = dict(row)
            row["arm"] = args.label
            row["source_file"] = os.path.basename(path)
            cells.add(tuple(round(float(x), 9) for x in row["pi"]))
            rows.append(row)

    if base is None:
        raise SystemExit("no inputs")
    base["arm"] = args.label
    base["n_inputs"] = len(args.inputs)
    base["n_evaluated"] = len(rows)
    base["merged_radii"] = sorted(radii)
    # Carried through when every shard agrees on it, blanked otherwise. See the
    # module docstring: preserving one shared ball is not inventing one.
    base["flow_exclusion"] = shared if one_ball else {}
    base["rows"] = rows

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(base, f)
    print(f"{args.label}: {len(args.inputs)} file(s), {len(rows)} rows, "
          f"{len(cells)} distinct mixtures -> {args.out}")
    if len(cells) != len(args.inputs):
        print(f"  NOTE: {len(args.inputs)} inputs but {len(cells)} distinct "
              f"mixtures -- two arms scored the same cell.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

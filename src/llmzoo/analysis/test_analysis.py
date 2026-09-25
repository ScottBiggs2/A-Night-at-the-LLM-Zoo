"""Independent local audit: NumPy only, no torch, network or cluster access.

Run: python analysis/test_analysis.py
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parent
for folder in ("anovas", "domain_geometry", "mixture_composition"):
    sys.path.insert(0, str(ROOT / folder))

from zoo import anchor_groups, analyze, tensor_layout, weight_dataset
from geometry import analyze_geometry
from composition import analyze_composition, projection_energy


class AnalysisAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rng = np.random.default_rng(173)
        # Extra=5 coordinates; two blocks each contain 6 matrix + 2 bias entries.
        self.meta = dict(layout_version=2, arch="gpt2_zoo_mini", beta=.15,
                         trunk_seed=1234, n_layers=2, block_size=8, extra_size=5,
                         n_params=21, schema=[["weight", [2, 3]], ["bias", [2]]],
                         extra_schema=[["embedding", [5]]], n_anchor_branches=4,
                         domains=["web", "code", "math", "books", "multilingual"],
                         members=[dict(idx=4*d+s, kind="anchor", pi=np.eye(5)[d].tolist())
                                  for d in range(5) for s in range(4)])
        self.pi = self.rng.dirichlet(np.ones(5), size=3)
        self.meta["members"] += [dict(idx=20+i, kind="singleton", pi=p.tolist())
                                 for i, p in enumerate(self.pi)]
        self.x = (self.rng.normal(size=(5, 1, 21))*3
                  + self.rng.normal(size=(5, 4, 21))).astype(np.float32)
        self.y = (self.pi @ self.x.mean(axis=1, dtype=np.float64)
                  + self.rng.normal(size=(3, 21))).astype(np.float32)
        for i, vector in enumerate(np.concatenate([self.x.reshape(20, 21), self.y])):
            np.save(self.root / f"w_{i}.npy", vector)
        (self.root / "zoo_meta.json").write_text(json.dumps(self.meta))

    def regions(self):
        regions = {( "tensor", name): slice(start, stop)
                   for name, _, start, stop, _ in tensor_layout(self.meta)}
        regions.update({("layer", "extra"): slice(0, 5),
                        ("layer", "layer_0"): slice(5, 13),
                        ("layer", "layer_1"): slice(13, 21),
                        ("model", "whole_model"): slice(0, 21)})
        return regions

    def test_all_regions_against_full_array_and_direct_svd(self):
        regions = self.regions()
        for chunk_size in (1, 7, 100):
            _, anova = analyze(self.root, self.meta, chunk_size)
            geometry = analyze_geometry(self.root, self.meta, chunk_size)
            composition, _ = analyze_composition(self.root, self.meta, chunk_size=chunk_size)
            for row in anova:
                x = self.x[..., regions[row["level"], row["name"]]].astype(np.float64)
                means, grand = x.mean(axis=1), x.mean(axis=(0, 1))
                between = 4*np.sum((means-grand)**2)
                within = np.sum((x-means[:, None])**2)
                total = np.sum((x-grand)**2)
                np.testing.assert_allclose([row[k] for k in ("ss_between", "ss_within", "ss_total")],
                                           [between, within, total], rtol=1e-12, atol=1e-12)
                self.assertAlmostEqual(row["domain_variance_corrected"], (between-4*within/15)/20)
            for row in geometry:
                x = self.x[..., regions[row["level"], row["name"]]].astype(np.float64)
                means = x.mean(axis=1)
                singular = np.linalg.svd((means-means.mean(axis=0)).T, compute_uv=False)
                expected = np.zeros(4)
                expected[:min(4, len(singular))] = singular[:4]**2
                np.testing.assert_allclose([row[f"eigenvalue_{i}"] for i in range(1, 5)], expected,
                                           rtol=1e-10, atol=1e-11)
            for row in composition:
                region = regions[row["level"], row["name"]]
                x = self.x[..., region].astype(np.float64)
                means = x.mean(axis=1)
                j = row["member_idx"]-20
                residual = self.y[j, region].astype(np.float64)-self.pi[j]@means
                u, singular, _ = np.linalg.svd((means-means.mean(axis=0)).T, full_matrices=False)
                keep = singular**2 > singular[0]**2*1e-8
                projected = u[:, keep]@(u[:, keep].T@residual)
                np.testing.assert_allclose([row[k] for k in ("residual_squared", "parallel_squared", "perpendicular_squared")],
                                           [residual@residual, projected@projected,
                                            (residual-projected)@(residual-projected)], rtol=1e-10, atol=1e-10)

    def test_duplicate_singleton_and_anchor_collision_rejected(self):
        for idx in (20, 0):
            meta = copy.deepcopy(self.meta)
            meta["members"][-1]["idx"] = idx
            with self.assertRaises(ValueError):
                anchor_groups(meta)

    def test_reference_shape_and_nonfinite_rejected(self):
        for ref in (np.zeros(1), np.zeros((1, 21)), np.zeros(22)):
            with self.assertRaises(ValueError):
                analyze_composition(self.root, self.meta, reference=ref)
        with self.assertRaises(ValueError):
            analyze_composition(self.root, self.meta, reference=np.full(21, np.nan))

    def test_random_source_is_not_relabeled(self):
        meta = {**self.meta, "source": "random_initialization"}
        (self.root / "zoo_meta.json").write_text(json.dumps(meta))
        args = argparse.Namespace(chunk_size=10, random_mixtures=4, work_dir=None,
                                  zoo_dir=self.root, config=None)
        with weight_dataset(args) as (_, _, source):
            self.assertEqual(source, "random_initialization")

    def test_rounding_in_coefficients_is_normalized(self):
        weights = {str(20+i): (pi*(1+1e-9)).tolist() for i, pi in enumerate(self.pi)}
        actual, used = analyze_composition(self.root, self.meta, token_weights=weights)
        expected, _ = analyze_composition(self.root, self.meta)
        np.testing.assert_allclose([r["residual_squared"] for r in actual],
                                   [r["residual_squared"] for r in expected], rtol=1e-12, atol=1e-12)
        for pi in used.values():
            self.assertAlmostEqual(sum(pi), 1)

    def test_inconsistent_projection_not_silently_clipped(self):
        with self.assertRaises(ValueError):
            projection_energy(np.eye(2), np.array([10., 10.]), 1.)

    def test_helpers_identical_and_cli_outputs(self):
        helpers = [(ROOT / folder / "zoo.py").read_bytes()
                   for folder in ("anovas", "domain_geometry", "mixture_composition")]
        self.assertEqual(helpers[0], helpers[1])
        self.assertEqual(helpers[1], helpers[2])
        for folder, script, name in (("anovas", "anova.py", "anova"),
                                     ("domain_geometry", "geometry.py", "geometry"),
                                     ("mixture_composition", "composition.py", "composition")):
            output = self.root / name
            subprocess.run([sys.executable, str(ROOT / folder / script), "--dataset-path",
                            str(self.root), "--output-dir", str(output), "--chunk-size", "7"],
                           cwd=self.root, check=True, capture_output=True, text=True)
            report = json.loads((output / f"{name}.json").read_text())
            self.assertEqual(report["source"], "pretrained_zoo")
            self.assertTrue(report["rows"])
            self.assertTrue((output / f"{name}.csv").stat().st_size)
            ET.parse(output / "figure.svg")


if __name__ == "__main__":
    unittest.main()

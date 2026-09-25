"""NumPy-only tests of coefficient error versus new-direction error."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from llmzoo.analysis.mixture_composition.composition import analyze_composition, projection_energy


class CompositionTests(unittest.TestCase):
    def test_projection(self):
        effects = np.array([[-1., 0.], [1., 0.]])
        residual = np.array([2., 3.])
        parallel, perpendicular, rank = projection_energy(effects @ effects.T,
                                                          effects @ residual, 13)
        self.assertAlmostEqual(parallel, 4)
        self.assertAlmostEqual(perpendicular, 9)
        self.assertEqual(rank, 1)
        self.assertEqual(projection_energy(np.zeros((2, 2)), np.zeros(2), 9), (0, 9, 0))

    def test_mixtures_and_noise(self):
        meta = dict(layout_version=2, n_layers=1, block_size=2, extra_size=0,
                    n_params=2, schema=[["weight", [2]]], extra_schema=[],
                    domains=["web", "code"], n_anchor_branches=3,
                    members=[dict(idx=i, kind="anchor", pi=[1, 0] if i < 3 else [0, 1]) for i in range(6)]
                    + [dict(idx=i, kind="singleton", pi=[0.5, 0.5]) for i in range(6, 9)])
        values = [[x, 0] for x in [9, 10, 11, 19, 20, 21]] + [[15, 0], [17, 0], [15, 3]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, value in enumerate(values):
                np.save(root / f"w_{i}.npy", np.array(value, dtype=np.float32))
            rows, _ = analyze_composition(root, meta, chunk_size=1)
            self.assertEqual(rows, analyze_composition(root, meta, chunk_size=100)[0])
            whole = [r for r in rows if r["level"] == "model"]
            self.assertEqual(whole[0]["residual_squared"], 0)
            self.assertAlmostEqual(whole[1]["parallel_squared"], 4)
            self.assertAlmostEqual(whole[1]["perpendicular_squared"], 0)
            self.assertAlmostEqual(whole[2]["parallel_squared"], 0)
            self.assertAlmostEqual(whole[2]["perpendicular_squared"], 9)
            self.assertAlmostEqual(whole[0]["expected_noise_squared"], 1 + 0.5 / 3)
            self.assertIsNone(whole[0]["relative_residual"])
            # A changed common reference must not change the prediction residual.
            ref_rows, _ = analyze_composition(root, meta, reference=np.array([5, 0]))
            last = next(r for r in ref_rows if r["level"] == "model" and r["member_idx"] == 8)
            self.assertAlmostEqual(last["relative_residual"], np.sqrt(9 / 109))
            self.assertEqual(last["perpendicular_squared"], whole[2]["perpendicular_squared"])
            with self.assertRaises(ValueError):
                analyze_composition(root, meta, member_ids=[0])
            with self.assertRaises(ValueError):
                analyze_composition(root, meta, token_weights={str(i): [0.9, 0.9] for i in range(6, 9)})


if __name__ == "__main__":
    unittest.main()

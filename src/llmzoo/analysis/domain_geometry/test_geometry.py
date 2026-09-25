"""NumPy-only tests; no model construction."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from llmzoo.analysis.domain_geometry.geometry import analyze_geometry, spectrum


class GeometryTests(unittest.TestCase):
    def test_spectrum_and_zero(self):
        effects = np.array([[-1., 0.], [1., 0.]])
        values, fractions, rank = spectrum(effects @ effects.T)
        np.testing.assert_allclose(values, [2])
        np.testing.assert_allclose(fractions, [1])
        self.assertEqual(rank, 1)
        self.assertIsNone(spectrum(np.zeros((5, 5)))[2])

    def test_saved_data_and_chunking(self):
        meta = dict(layout_version=2, n_layers=1, block_size=2, extra_size=0,
                    n_params=2, schema=[["weight", [2]]], extra_schema=[],
                    domains=["web", "code"], n_anchor_branches=3,
                    members=[dict(idx=i, kind="anchor", pi=[1, 0] if i < 3 else [0, 1]) for i in range(6)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, value in enumerate([9, 10, 11, 19, 20, 21]):
                np.save(root / f"w_{i}.npy", np.array([value, 0], dtype=np.float32))
            rows = analyze_geometry(root, meta, 1)
            self.assertEqual(rows, analyze_geometry(root, meta, 100))
            whole = next(r for r in rows if r["level"] == "model")
            self.assertEqual(whole["effective_rank"], 1)
            self.assertAlmostEqual(whole["ss_between"], 150)
            self.assertAlmostEqual(whole["ss_within"], 4)


if __name__ == "__main__":
    unittest.main()

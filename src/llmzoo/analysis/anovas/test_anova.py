"""NumPy-only checks; never imports torch or transformers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from llmzoo.analysis.anovas.anova import analyze, anchor_groups, sums_of_squares, summarize


class AnovaTests(unittest.TestCase):
    def test_known_example_and_translation(self):
        values = np.array([[[10], [11], [9]], [[20], [22], [18]]], dtype=float)
        np.testing.assert_allclose(sums_of_squares(values), [150, 10, 160])
        np.testing.assert_allclose(sums_of_squares(values - 1000), [150, 10, 160])
        row = summarize("example", "tensor", 1, [150, 10, 160], 2, 3)
        self.assertEqual(row["between_fraction"], 0.9375)

    def test_zero_variation_and_nonfinite(self):
        ss = sums_of_squares(np.ones((5, 4, 3)))
        self.assertEqual(summarize("zero", "tensor", 3, ss, 5, 4)["between_fraction"], None)
        with self.assertRaises(ValueError):
            sums_of_squares(np.full((5, 4, 3), np.nan))

    def test_saved_layout_chunking_and_groups(self):
        meta = dict(
            layout_version=2, n_layers=2, block_size=3, extra_size=2, n_params=8,
            schema=[["weight", [2]], ["bias", [1]]],
            extra_schema=[["embedding", [2]]], domains=["web", "math"],
            n_anchor_branches=3,
            members=[dict(idx=i + 10, kind="anchor", pi=[1, 0] if i < 3 else [0, 1])
                     for i in range(6)] + [dict(idx=99, kind="singleton", pi=[0.5, 0.5])],
        )
        values = np.array([10, 11, 9, 20, 22, 18], dtype=np.float32)[:, None]
        weights = values * np.arange(1, 9, dtype=np.float32)[None, :]
        with tempfile.TemporaryDirectory() as directory:
            for idx, vector in enumerate(weights):
                np.save(Path(directory) / f"w_{idx + 10}.npy", vector)
            groups, rows = analyze(directory, meta, chunk_size=1)
            self.assertEqual(groups, [[10, 11, 12], [13, 14, 15]])
            _, big_chunks = analyze(directory, meta, chunk_size=100)
            self.assertEqual(rows, big_chunks)
            whole = next(row for row in rows if row["name"] == "whole_model")
            expected = sums_of_squares(weights.reshape(2, 3, 8))
            np.testing.assert_allclose([whole[k] for k in ("ss_between", "ss_within", "ss_total")], expected)
            self.assertEqual(whole["n_params"], 8)
            Path(directory, "w_10.npy").unlink()
            with self.assertRaises(FileNotFoundError):
                analyze(directory, meta)
        meta["members"].pop(0)
        with self.assertRaises(ValueError):
            anchor_groups(meta)


if __name__ == "__main__":
    unittest.main()

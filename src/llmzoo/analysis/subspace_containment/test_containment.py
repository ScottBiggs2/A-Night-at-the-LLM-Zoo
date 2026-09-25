"""NumPy-only checks; never imports torch, transformers or the repository."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from llmzoo.analysis.subspace_containment.containment import (analyze, coefficient_matrices, containment, debias,
                         member_tables, region_grams, tensor_layout)

D, S, NM, P = 5, 4, 40, 600


def make_meta(n_singletons=NM, branches=S):
    rng = np.random.default_rng(1)
    members = [dict(idx=d * branches + r, kind="anchor", pi=np.eye(D)[d].tolist(),
                    mixture_id=f"anchor_{d}", branch=r)
               for d in range(D) for r in range(branches)]
    base = D * branches
    for j in range(n_singletons):
        p = rng.dirichlet(np.ones(D))
        members.append(dict(idx=base + j, kind="singleton", pi=p.tolist(),
                            mixture_id=f"dirichlet_{j:03d}", branch=0))
    # P = extra_size + n_layers * block_size = 200 + 2 * 200
    return dict(layout_version=2, arch="gpt2_zoo_mini", n_layers=2,
                block_size=200, extra_size=200, n_params=P,
                schema=[["w", [100]], ["b", [100]]],
                extra_schema=[["transformer.wte.weight", [200]]],
                exclude_1d=False, include_extra=True, n_anchor_branches=branches,
                domains=["web", "code", "math", "books", "multilingual"],
                members=members)


def make_zoo(directory, meta, *, noise=0.30, curvature=0.0, seed=0):
    """Anchors around D domain means; mixtures linear in pi plus optional curvature.

    curvature=0 puts every mixture exactly in the anchor affine span, so
    containment holds BY CONSTRUCTION and the debiased test must say so.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((D, P)) * 3.0
    quad = rng.standard_normal((D, D, P))
    for member in meta["members"]:
        pi = np.asarray(member["pi"])
        mean = pi @ base
        if member["kind"] == "singleton" and curvature:
            mean = mean + curvature * np.einsum("a,b,abp->p", pi, pi, quad)
        vector = mean + rng.standard_normal(P) * noise
        np.save(Path(directory) / f"w_{member['idx']}.npy", vector.astype(np.float32))
    (Path(directory) / "zoo_meta.json").write_text(json.dumps(meta))
    return base


class LayoutTests(unittest.TestCase):
    def test_layout_tiles_and_names(self):
        meta = make_meta()
        entries = list(tensor_layout(meta))
        self.assertEqual(entries[0][2], 0)
        self.assertEqual(entries[-1][3], meta["n_params"])
        for a, b in zip(entries, entries[1:]):
            self.assertEqual(a[3], b[2])
        self.assertEqual({e[1] for e in entries}, {"extra", "layer_0", "layer_1"})
        self.assertEqual(len({e[0] for e in entries}), len(entries))

    def test_member_tables_and_refusals(self):
        meta = make_meta()
        order, sizes, mixtures, pi = member_tables(meta)
        self.assertEqual(order, list(range(D * S + NM)))
        self.assertEqual(sizes, [S] * D)
        self.assertEqual(mixtures, list(range(D * S, D * S + NM)))
        np.testing.assert_allclose(pi.sum(axis=1), 1.0)
        thin = make_meta(n_singletons=3)                  # fewer mixtures than dim
        with self.assertRaises(ValueError):
            member_tables(thin)
        ragged = make_meta()
        ragged["members"] = [m for m in ragged["members"] if m["idx"] != 0]
        with self.assertRaises(ValueError):
            member_tables(ragged)

    def test_coefficient_rows_sum_to_zero(self):
        la, lm, lw, d, s = coefficient_matrices(D * S + NM, [S] * D,
                                                list(range(D * S, D * S + NM)))
        for name, matrix in (("L_A", la), ("L_M", lm), ("L_W", lw)):
            np.testing.assert_allclose(matrix.sum(axis=1), 0, atol=1e-12,
                                       err_msg=f"{name} rows must sum to zero")
        self.assertEqual((d, s), (D, S))


class GeometryTests(unittest.TestCase):
    def test_gram_route_matches_direct_vectors(self):
        meta = make_meta()
        with tempfile.TemporaryDirectory() as directory:
            make_zoo(directory, meta, curvature=0.8)
            order, sizes, mixtures, _ = member_tables(meta)
            grams, widths = region_grams(directory, meta, order, chunk_size=37)
            X = np.array([np.load(Path(directory) / f"w_{i}.npy") for i in order],
                         dtype=np.float64)
        la, lm, lw, _, _ = coefficient_matrices(len(order), sizes, mixtures)
        gram = grams["whole_model"]
        self.assertEqual(widths["whole_model"], P)
        # a Gram of displacements must give the same blocks as one of raw vectors,
        # because every coefficient row sums to zero
        direct = (X - X.mean(0)) @ (X - X.mean(0)).T
        for L in (la, lm, lw):
            np.testing.assert_allclose(L @ gram @ L.T, L @ direct @ L.T,
                                       rtol=1e-8, atol=1e-8)
        # principal cosines from blocks == from explicit orthonormal bases
        A, M = la @ X, lm @ X
        got = containment(la @ gram @ la.T, lm @ gram @ lm.T, la @ gram @ lm.T, P)
        qa = np.linalg.qr(A.T)[0]
        qb = np.linalg.qr(M.T)[0]
        want = np.linalg.svd(qa.T @ qb, compute_uv=False)
        # atol 2e-3, not machine precision, and the reason is structural: forming
        # G = X X' squares the condition number, which is the price of never
        # materialising a P-dimensional basis. The QR route here is the accurate
        # reference and is only available because this fixture is tiny.
        np.testing.assert_allclose(got["principal_cosines"],
                                   want[:len(got["principal_cosines"])], atol=2e-3)
        # energy fraction == explicit projection
        projector = qb @ qb.T
        np.testing.assert_allclose(got["anchor_energy_in_mixture_span"],
                                   ((A @ projector) ** 2).sum() / (A ** 2).sum(),
                                   rtol=5e-3)   # same squared-condition-number price

    def test_chunking_is_invariant(self):
        meta = make_meta()
        with tempfile.TemporaryDirectory() as directory:
            make_zoo(directory, meta)
            order, _, _, _ = member_tables(meta)
            a, _ = region_grams(directory, meta, order, chunk_size=1)
            b, _ = region_grams(directory, meta, order, chunk_size=10 ** 6)
        for key in a:
            np.testing.assert_allclose(a[key], b[key], rtol=1e-9, atol=1e-9)

    def test_debias_removes_the_noise_floor(self):
        """The floor is what stops containment reading 1.0 when it truly holds."""
        rng = np.random.default_rng(3)
        effects = rng.standard_normal((D, 400)) * 2.0
        effects -= effects.mean(0)
        noise = rng.standard_normal((D, 400)) * 0.5 / np.sqrt(S)
        ga = (effects + noise) @ (effects + noise).T
        trace_within = 0.25 * 400
        gad, _ = debias(ga, np.eye(3), trace_within, S)
        clean = effects @ effects.T
        self.assertLess(abs(np.trace(gad) - np.trace(clean)),
                        abs(np.trace(ga) - np.trace(clean)))


class VerdictTests(unittest.TestCase):
    def test_linear_mixtures_are_contained_and_stay_within_the_simplex(self):
        meta = make_meta()
        with tempfile.TemporaryDirectory() as directory:
            make_zoo(directory, meta, curvature=0.0)
            rows, _, _ = analyze(directory, meta, chunk_size=101)
        whole = next(r for r in rows if r["region"] == "whole_model")
        self.assertEqual(whole["simplex_dimension"], D - 1)
        # debiasing must beat the raw statistic on data where containment holds
        self.assertGreater(whole["debiased_smallest_principal_cosine"],
                           whole["raw_smallest_principal_cosine"])
        self.assertGreater(whole["debiased_anchor_energy_in_mixture_span"], 0.95)
        self.assertLessEqual(whole["debiased_mixture_dimension_above_noise"], D - 1)
        self.assertFalse(whole["nonlinear_beyond_simplex"])
        self.assertGreater(whole["debiased_anchor_energy_in_mixture_span"],
                           whole["debiased_random_subspace_null_anchor_in_mixture"])

    def test_curved_mixtures_exceed_the_simplex_dimension(self):
        meta = make_meta()
        with tempfile.TemporaryDirectory() as directory:
            make_zoo(directory, meta, curvature=2.5)
            rows, _, _ = analyze(directory, meta, chunk_size=101)
        whole = next(r for r in rows if r["region"] == "whole_model")
        self.assertGreater(whole["debiased_mixture_dimension_above_noise"], D - 1)
        self.assertTrue(whole["nonlinear_beyond_simplex"])
        # The anchor span stays far more reachable than a random subspace of the
        # same rank, but not ~1.0: with curvature the mixture span is estimated
        # from points spread over many more directions, so it is noisier.
        self.assertGreater(whole["debiased_anchor_energy_in_mixture_span"], 0.5)
        self.assertGreater(whole["debiased_anchor_energy_in_mixture_span"],
                           20 * whole["debiased_random_subspace_null_anchor_in_mixture"])

    def test_every_region_is_reported(self):
        meta = make_meta()
        with tempfile.TemporaryDirectory() as directory:
            make_zoo(directory, meta)
            rows, order, mixtures = analyze(directory, meta)
        self.assertEqual({r["region"] for r in rows},
                         {"extra", "layer_0", "layer_1", "whole_model"})
        self.assertEqual({r["level"] for r in rows}, {"extra", "layer", "model"})
        self.assertEqual(sum(r["n_params"] for r in rows if r["region"] != "whole_model"), P)
        self.assertEqual(len(order), D * S + NM)
        self.assertEqual(len(mixtures), NM)


if __name__ == "__main__":
    unittest.main()

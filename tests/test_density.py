#!/usr/bin/env python
"""
Known-answer tests for llmzoo.gen.density.

The point of a density module is that its output is not checkable by eye: a
confident, wrong log-likelihood looks exactly like a correct one. So every test here
compares against a CLOSED FORM, using stub velocity fields whose flow map and
divergence are known analytically, plus one consistency test on a real MLP.

  python tests/test_density.py           # cpu is fine; the nets here are toys
"""

from __future__ import annotations

import math
import sys

import torch

from llmzoo.gen.density import (assert_consistent, identification, log_prob,
                                logp_matrix)

FAILS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def close(name: str, got: float, want: float, tol: float) -> None:
    check(name, abs(got - want) <= tol, f"got {got:.6f} want {want:.6f} (tol {tol})")


class _Net:
    """Duck-types FlowVelocityNet: log_prob only ever calls forward_cfg."""

    def __init__(self, fn):
        self.fn = fn

    def forward_cfg(self, x, t, family_idx, guidance_scale=1.0, pi=None):
        return self.fn(x, t, pi)


class _Flow:
    def __init__(self, fn, source_std=1.0):
        self.net = _Net(fn)
        self.source_std = source_std


def _gauss_logp(x, sd):
    d = x.shape[1]
    return (-0.5 * (x ** 2).sum(1) / sd ** 2
            - 0.5 * d * math.log(2 * math.pi * sd * sd))


def test_translation() -> None:
    """v = a constant. The map is a translation, divergence is exactly 0."""
    print("\n--- constant field: pure translation, zero divergence ---")
    d, B = 4, 6
    a = torch.tensor([0.3, -0.7, 1.1, 0.05])
    flow = _Flow(lambda x, t, pi: a.expand_as(x))
    x1 = torch.randn(B, d)
    fam = torch.zeros(B, dtype=torch.long)
    for method in ("trace", "map"):
        lp, diag = log_prob(flow, x1, fam, n_steps=20, method=method)
        want = _gauss_logp(x1 - a, 1.0)
        close(f"{method}: log p matches the translated Gaussian",
              float((lp - want).abs().max()), 0.0, 1e-4)
        close(f"{method}: accumulated log-det is 0", diag["logdet_mean"], 0.0, 1e-6)


def test_linear_contraction() -> None:
    """
    v = lam * x. Then x(t) = e^(lam t) x(0), so x0 = e^-lam x1, and the continuous
    divergence integral is exactly lam*d.
    """
    print("\n--- linear field v = lam*x: closed-form flow and divergence ---")
    d, B, lam = 5, 8, 0.4
    flow = _Flow(lambda x, t, pi: lam * x)
    x1 = torch.randn(B, d)
    fam = torch.zeros(B, dtype=torch.long)
    lp, diag = log_prob(flow, x1, fam, n_steps=4000, method="trace")
    want = _gauss_logp(x1 * math.exp(-lam), 1.0) - lam * d
    close("trace: log p matches the analytic linear-flow density",
          float((lp - want).abs().max()), 0.0, 2e-3)
    close("trace: divergence integral is lam*d", diag["logdet_mean"], lam * d, 2e-3)

    # The same field with a coarse grid: trace and map must diverge by O(dt),
    # which is what makes assert_consistent a real check rather than a tautology.
    a_c, _ = log_prob(flow, x1, fam, n_steps=8, method="trace")
    b_c, _ = log_prob(flow, x1, fam, n_steps=8, method="map")
    a_f, _ = log_prob(flow, x1, fam, n_steps=800, method="trace")
    b_f, _ = log_prob(flow, x1, fam, n_steps=800, method="map")
    coarse = float((a_c - b_c).abs().mean())
    fine = float((a_f - b_f).abs().mean())
    check("trace vs map gap shrinks with n_steps (it is an O(dt) discretisation "
          "difference, not a bug)", fine < coarse / 10.0,
          f"n_steps=8 gap {coarse:.5f} -> n_steps=800 gap {fine:.6f}")


def test_source_std_is_honoured() -> None:
    """A wider source must change the density by exactly the Gaussian constant."""
    print("\n--- source_std enters the base density, not the integrator ---")
    d, B = 3, 5
    x1 = torch.randn(B, d)
    fam = torch.zeros(B, dtype=torch.long)
    f1 = _Flow(lambda x, t, pi: torch.zeros_like(x), source_std=1.0)
    f2 = _Flow(lambda x, t, pi: torch.zeros_like(x), source_std=2.0)
    l1, _ = log_prob(f1, x1, fam, n_steps=10)
    l2, _ = log_prob(f2, x1, fam, n_steps=10)
    want = _gauss_logp(x1, 2.0) - _gauss_logp(x1, 1.0)
    close("log p shifts by exactly the base-Gaussian difference",
          float((l2 - l1 - want).abs().max()), 0.0, 1e-5)


def test_conditioning_is_identified() -> None:
    """
    A field that genuinely depends on pi must let logp_matrix recover which pi
    produced a point; a field that ignores pi must NOT.

    This is the control the real measurement rests on: if the identification test
    cannot separate a pi-dependent field from a pi-blind one on a toy, it cannot be
    trusted on the flow.
    """
    print("\n--- identification: a pi-blind field must read as chance ---")
    d, n_pi = 4, 12
    torch.manual_seed(0)
    # Sparse mixtures (Dirichlet with small alpha) so the conditional means are
    # actually far apart; with near-uniform pi every pi @ W lands near the mean of
    # W's rows and the toy is unidentifiable for reasons that say nothing about the
    # estimator.
    gam = torch._standard_gamma(torch.full((n_pi, 5), 0.25))
    pis = gam / gam.sum(1, keepdim=True)
    W = torch.randn(5, d) * 6.0

    cond = _Flow(lambda x, t, pi: (pi @ W) - x)          # pulls toward pi @ W
    blind = _Flow(lambda x, t, pi: -x)                   # same field, no pi

    # Draw one point per mixture by running the conditional field forward.
    xs = []
    for j in range(n_pi):
        x = torch.randn(1, d)
        for i in range(200):
            t = torch.full((1,), i / 200.0)
            x = x + (1.0 / 200.0) * cond.net.forward_cfg(
                x, t, torch.zeros(1, dtype=torch.long), pi=pis[j:j + 1])
        xs.append(x)
    X = torch.cat(xs, 0)
    fam = torch.zeros(n_pi, dtype=torch.long)

    Mc = logp_matrix(cond, X, fam, pis, n_steps=60)
    Mb = logp_matrix(blind, X, fam, pis, n_steps=60)
    ic = identification(Mc, list(range(n_pi)))
    ib = identification(Mb, list(range(n_pi)))
    print(f"    conditional: top1 {ic['top1']:.2f}  median rank {ic['median_rank']}"
          f"  InfoNCE {ic['infonce_nats']:+.3f} / {ic['max_infonce']:.3f}")
    print(f"    pi-blind   : top1 {ib['top1']:.2f}  median rank {ib['median_rank']}"
          f"  InfoNCE {ib['infonce_nats']:+.3f}")
    check("conditional field: true pi is identified far above chance",
          ic["top1"] > 0.6, f"top1 {ic['top1']:.2f} vs chance {ic['chance_top1']:.2f}")
    check("conditional field: InfoNCE is a large positive number of nats",
          ic["infonce_nats"] > 0.5 * ic["max_infonce"],
          f"{ic['infonce_nats']:.3f} of a possible {ic['max_infonce']:.3f}")
    check("pi-BLIND field: InfoNCE is ~0 — the ablation the real run must "
          "reproduce with cond_mode=family", abs(ib["infonce_nats"]) < 1e-6,
          f"{ib['infonce_nats']:.2e}")
    # The tie convention is what makes this meaningful: every candidate scores
    # identically, so the honest answer is the MIDDLE rank, not rank 1.
    check("pi-blind field: every candidate ties", ib["tie_fraction"] > 0.99,
          f"tie fraction {ib['tie_fraction']:.3f}")
    check("pi-blind field: median rank is chance, NOT 1 (mid-rank ties)",
          abs(ib["median_rank"] - ib["chance_median_rank"]) < 1e-6,
          f"median rank {ib['median_rank']:.1f} vs chance "
          f"{ib['chance_median_rank']:.1f}")
    check("pi-blind field: top-1 is not inflated by ties", ib["top1"] < 0.2,
          f"top1 {ib['top1']:.3f}")


def test_hutchinson_is_unbiased() -> None:
    """The estimator must converge to the exact trace as probes accumulate."""
    print("\n--- Hutchinson converges to the exact trace ---")
    d, B = 6, 4
    torch.manual_seed(1)
    A = torch.randn(d, d) * 0.3
    flow = _Flow(lambda x, t, pi: x @ A.T)
    x1 = torch.randn(B, d)
    fam = torch.zeros(B, dtype=torch.long)
    exact, _ = log_prob(flow, x1, fam, n_steps=40, method="trace")
    g = torch.Generator().manual_seed(7)
    approx, _ = log_prob(flow, x1, fam, n_steps=40, method="trace",
                         n_probe=256, generator=g)
    close("Hutchinson(256) tracks the exact trace",
          float((exact - approx).abs().mean()), 0.0, 0.05)


def test_assert_consistent() -> None:
    print("\n--- assert_consistent on a real FlowVelocityNet ---")
    from llmzoo.gen.flow import FlowVelocityNet, RectifiedFlow
    torch.manual_seed(3)
    net = FlowVelocityNet(dim=8, cond_mode="pi", pi_dim=5, hidden_dims=(16, 16))
    flow = RectifiedFlow(net)
    x1 = torch.randn(5, 8)
    fam = torch.zeros(5, dtype=torch.long)
    pi = torch.full((5, 5), 0.2)
    fine = assert_consistent(flow, x1, fam, pi=pi, n_steps=200)
    print(f"    n_steps=200: gap {fine['mean_abs_gap']:.5f}  "
          f"logp sd {fine['logp_std']:.4f}  relative {fine['relative']:.5f}")
    check("a real velocity net is consistent between the two densities at "
          "n_steps=200", fine["ok"], f"relative gap {fine['relative']:.5f}")


def main() -> int:
    print("=" * 78)
    print("llmzoo.gen.density — closed-form checks")
    print("=" * 78)
    test_translation()
    test_linear_contraction()
    test_source_std_is_honoured()
    test_conditioning_is_identified()
    test_hutchinson_is_unbiased()
    test_assert_consistent()
    print("\n" + "=" * 78)
    if FAILS:
        print(f"FAILED ({len(FAILS)}):")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("All density checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

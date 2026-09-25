"""
Exact conditional log-likelihood for the rectified flow.

WHY
---
Every conditioning statistic in this repo so far is a MOMENT -- a width, a squared
bias, a mean distance -- because the flow had no density. That is also why the
headline one is broken: `roundtrip_l1` is a mean per-sample distance to the target,
and by Jensen E||f(X) - pi||_1 is minimised by a point mass, so a memorising
generator beats the correct conditional on it (measured 0.131 vs 0.157 on the real
Mini code matrix).

A log score has the opposite pathology, in the useful direction: it assigns a point
mass INFINITE loss. So `log p(x | pi)` is the sharpest available collapse control,
and it buys two things no moment can:

  * held-out conditional NLL in nats/dim -- the standard generative number;
  * the log p(x_i | pi_j) MATRIX, whose row-wise rank of the true pi is a
    calibration-free identification test (chance is 1/85 here), and whose InfoNCE
    bound on I(x; pi) is valid because a flow likelihood is normalised. A Gaussian
    surrogate's is not: fitted on this zoo it reports -15,335 nats.

THE MATH
--------
The flow is an ODE dx/dt = v(x, t, c) carrying N(0, sigma^2 I) at t=0 to the data at
t=1. The instantaneous change of variables gives

    d log p(x_t)/dt = -tr(dv/dx)   =>   log p_1(x_1) = log p_0(x_0) - I,
    I = integral_0^1 tr(dv/dx)(x_t, t) dt.

So: integrate BACKWARD from x_1 to x_0 along the same Euler path `integrate()` walks
forward, accumulating the trace. At k <= 99 on a 51k-parameter MLP the EXACT trace
is affordable -- one jacrev per step -- so Hutchinson is offered but is not the
default. Estimating a divergence when you can compute it is how a variance artifact
becomes a result.

TWO DENSITIES, AND THEY ARE NOT THE SAME
----------------------------------------
`trace` integrates the CONTINUOUS ODE, which is what "the flow's likelihood" means in
the literature (FFJORD, and the probability-flow ODE). `map` instead takes the exact
log-determinant of the discrete Euler step actually used to sample,
log|det(I + dt * dv/dx)|, which is the density of the sampler you really run. They
agree to O(dt^2) and their gap is a direct read on whether n_steps is large enough --
`assert_consistent` uses exactly that, and it is the self-test for this module.

SPACE
-----
Everything here is in FLOW space (per-family unit-RMS codes). Converting to raw PCA
code space adds a constant -k*log(std_family), which cancels in every comparison
made here -- across conditions, across members, between flows sharing a CodeStats.
`logp_codes` applies it anyway, so a number quoted outside that context is right.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch


def _velocity(flow, x: torch.Tensor, t: torch.Tensor, family_idx: torch.Tensor,
              guidance_scale: float, pi: Optional[torch.Tensor]) -> torch.Tensor:
    return flow.net.forward_cfg(x, t, family_idx, guidance_scale=guidance_scale,
                                pi=pi)


def _jacobian(flow, x: torch.Tensor, t: torch.Tensor, family_idx: torch.Tensor,
              guidance_scale: float, pi: Optional[torch.Tensor]) -> torch.Tensor:
    """(B, d, d) batch Jacobian dv/dx. Exact, via one vjp per output coordinate."""
    B, d = x.shape
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        v = _velocity(flow, x, t, family_idx, guidance_scale, pi)
        # A field that does not depend on x at all (a pure translation) produces a
        # tensor with no grad_fn, and autograd.grad raises rather than returning
        # zeros. That is a legitimate velocity field with divergence exactly 0, and
        # it is the first closed-form case any test of this module will try.
        if not v.requires_grad:
            return torch.zeros(B, d, d, device=x.device, dtype=x.dtype)
        rows = []
        for j in range(d):
            g = torch.autograd.grad(
                v[:, j].sum(), x, retain_graph=(j < d - 1), create_graph=False,
                allow_unused=True)[0]
            rows.append(torch.zeros_like(x) if g is None else g)
    return torch.stack(rows, dim=1)                      # (B, d, d): [:, j, :] = dv_j/dx


def _divergence_hutchinson(flow, x, t, family_idx, guidance_scale, pi,
                           n_probe: int, generator=None) -> torch.Tensor:
    """Rademacher estimator of tr(dv/dx). Unbiased; variance falls as 1/n_probe."""
    x = x.detach().requires_grad_(True)
    acc = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    with torch.enable_grad():
        v = _velocity(flow, x, t, family_idx, guidance_scale, pi)
        if not v.requires_grad:          # x-independent field: divergence is 0
            return acc
        for i in range(n_probe):
            eps = torch.randint(0, 2, x.shape, generator=generator,
                                device="cpu").to(x.device).to(x.dtype) * 2 - 1
            g = torch.autograd.grad(v, x, grad_outputs=eps,
                                    retain_graph=(i < n_probe - 1),
                                    allow_unused=True)[0]
            if g is not None:
                acc = acc + (g * eps).sum(dim=1)
    return acc / n_probe


def log_prob(flow, x1: torch.Tensor, family_idx: torch.Tensor, *,
             n_steps: int = 50, guidance_scale: float = 1.0,
             pi: Optional[torch.Tensor] = None,
             method: str = "trace", n_probe: int = 0,
             generator=None) -> Tuple[torch.Tensor, dict]:
    """
    log p(x1 | family, pi) under the flow, in FLOW space. Returns (logp, diag).

    method
        "trace" -- integrate the continuous ODE's divergence (the standard CNF
                   likelihood). Exact trace unless n_probe > 0.
        "map"   -- exact log-det of the discrete Euler map actually sampled. Ignores
                   n_probe; there is nothing to estimate.

    Guidance is applied as at sampling time. Note that with guidance_scale != 1 the
    integrand is the GUIDED field, which is not the model's own density -- it is the
    density of the thing you actually sample. Both are defensible; they are not
    interchangeable, so `diag` records which was used.
    """
    if method not in ("trace", "map"):
        raise ValueError(f"method must be 'trace' or 'map', got {method!r}")
    B, d = x1.shape
    dev = x1.device
    dt = 1.0 / n_steps
    x = x1.detach().clone()
    acc = torch.zeros(B, device=dev, dtype=x1.dtype)

    # Walk t = 1 -> 0 with the same uniform grid integrate() uses forward. The
    # backward Euler step is x_{t-dt} = x_t - dt*v(x_t, t); using v at the ARRIVING
    # time (rather than a midpoint) keeps this the exact algebraic inverse of the
    # forward step at the same grid, which is what makes `map` a true log-det.
    for i in range(n_steps, 0, -1):
        t_val = i * dt
        t = torch.full((B,), t_val, device=dev, dtype=x1.dtype)
        if method == "map":
            J = _jacobian(flow, x, t, family_idx, guidance_scale, pi)
            eye = torch.eye(d, device=dev, dtype=x1.dtype).expand(B, d, d)
            acc = acc + torch.linalg.slogdet(eye + dt * J)[1]
        elif n_probe > 0:
            acc = acc + dt * _divergence_hutchinson(
                flow, x, t, family_idx, guidance_scale, pi, n_probe, generator)
        else:
            J = _jacobian(flow, x, t, family_idx, guidance_scale, pi)
            acc = acc + dt * torch.diagonal(J, dim1=1, dim2=2).sum(dim=1)
        with torch.no_grad():
            x = x - dt * _velocity(flow, x, t, family_idx, guidance_scale, pi)

    x0 = x.detach()
    sd = float(flow.source_std)
    logp0 = (-0.5 * (x0 ** 2).sum(dim=1) / (sd ** 2)
             - 0.5 * d * math.log(2.0 * math.pi * sd * sd))
    logp = logp0 - acc
    return logp, {"method": method, "n_steps": n_steps, "dim": d,
                  "guidance_scale": guidance_scale,
                  "n_probe": (n_probe if method == "trace" else 0),
                  "logp0_mean": float(logp0.mean()),
                  "logdet_mean": float(acc.mean()),
                  "x0_rms": float(x0.pow(2).mean().sqrt())}


def log_prob_codes(flow, space, codes: torch.Tensor, family_idx: torch.Tensor,
                   **kw) -> Tuple[torch.Tensor, dict]:
    """
    log p of RAW PCA codes, i.e. flow-space log p plus the normalisation Jacobian.

    CodeStats divides by ONE scalar per family, so the correction is -k*log(std)
    and is constant across conditions. It changes no comparison made here; it is
    applied so a number quoted on its own is still right.
    """
    x = space.to_flow(codes, family_idx)
    logp, diag = log_prob(flow, x, family_idx, **kw)
    std = space.code_stats.torch_stds(codes.device)[family_idx].reshape(-1)
    corr = -codes.shape[1] * torch.log(std)
    diag["jacobian_correction_mean"] = float(corr.mean())
    return logp + corr, diag


def logp_matrix(flow, x: torch.Tensor, family_idx: torch.Tensor,
                pis: torch.Tensor, **kw) -> torch.Tensor:
    """
    (n_x, n_pi) matrix of log p(x_i | pi_j): the conditioning identification test.

    Row-wise, the RANK of the true mixture among all candidates is calibration-free
    and needs no threshold -- chance is n_pi/2 median rank. It is the flow-likelihood
    analogue of a classification-accuracy score, but it needs no auxiliary
    classifier, which is the usual place such a test springs a leak.
    """
    n_x, n_pi = x.shape[0], pis.shape[0]
    out = torch.empty(n_x, n_pi, dtype=torch.float64)
    for j in range(n_pi):
        pj = pis[j].reshape(1, -1).expand(n_x, -1).contiguous()
        lp, _ = log_prob(flow, x, family_idx, pi=pj, **kw)
        out[:, j] = lp.double().cpu()
    return out


def identification(M: torch.Tensor, true_idx: Sequence[int]) -> dict:
    """
    Read the matrix: rank of the true pi, and the InfoNCE lower bound on I(x; pi).

    The bound is E[ log p(x_i|pi_i) - log mean_j p(x_i|pi_j) ] and is capped at
    log(n_pi), which is why it is a LOWER bound and why saturating it is the most
    that can be claimed from a finite candidate set.
    """
    n_x, n_pi = M.shape
    ranks, nce, ties = [], [], []
    for i in range(n_x):
        row = M[i]
        t = int(true_idx[i])
        # MID-RANK ties. A flow that ignores pi gives every candidate the identical
        # log-likelihood, and counting only STRICTLY greater entries would then
        # score it rank 1 on every row -- top-1 of 100%, from a model carrying no
        # information at all. That is the exact failure this test family exists to
        # catch, so the tie convention is load-bearing, not cosmetic.
        gt = int((row > row[t]).sum().item())
        eq = int((row == row[t]).sum().item()) - 1        # excluding itself
        ranks.append(1.0 + gt + 0.5 * eq)
        ties.append(eq / max(n_pi - 1, 1))
        nce.append(float(row[t] - torch.logsumexp(row, 0) + math.log(n_pi)))
    top1 = sum(1 for r in ranks if r <= 1.0) / n_x
    top5 = sum(1 for r in ranks if r <= 5.0) / n_x
    srt = sorted(ranks)
    return {"n_x": n_x, "n_pi": n_pi,
            "top1": top1, "top5": top5,
            # The full rank vector, kept so the rank CDF can be drawn downstream
            # without re-running the flow. n_candidates is an alias of n_pi that
            # names what the axis of that figure actually is.
            "ranks": ranks, "n_candidates": n_pi,
            "median_rank": srt[len(srt) // 2],
            "chance_top1": 1.0 / n_pi, "chance_median_rank": (n_pi + 1) / 2,
            "tie_fraction": sum(ties) / len(ties),
            "infonce_nats": sum(nce) / len(nce), "max_infonce": math.log(n_pi)}


def assert_consistent(flow, x1, family_idx, *, pi=None, n_steps=50,
                      tol=0.05) -> dict:
    """
    The self-test: the continuous trace and the discrete map log-det must agree.

    log|det(I + dt J)| = dt*tr(J) + O(dt^2), so a gap large relative to the spread of
    log p across the batch means n_steps is too small for the density to mean
    anything -- exactly the failure that would otherwise show up as a confident,
    wrong NLL.
    """
    a, _ = log_prob(flow, x1, family_idx, pi=pi, n_steps=n_steps, method="trace")
    b, _ = log_prob(flow, x1, family_idx, pi=pi, n_steps=n_steps, method="map")
    gap = float((a - b).abs().mean())
    scale = float(a.std()) if a.numel() > 1 else max(abs(float(a.mean())), 1.0)
    rel = gap / max(scale, 1e-12)
    return {"mean_abs_gap": gap, "logp_std": scale, "relative": rel,
            "ok": rel < tol, "n_steps": n_steps}

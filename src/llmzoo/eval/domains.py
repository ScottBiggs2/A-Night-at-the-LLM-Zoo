"""
Per-domain perplexity scoring, shared by the gate and by generated models.

WHY THIS MODULE EXISTS
----------------------
`scripts/eval_domains.py` measures the five per-domain perplexities of every REAL
zoo member. `scripts/eval_generated.py` measures the same five numbers for models
the flow produced. Those two sets of numbers are subtracted from each other in
`scripts/report_retrieval.py` and `scripts/report_functional.py`, and the whole
retrieval verdict is a difference of ~0.01-0.20 nats against a measured noise floor
of ~0.010 nats.

At that resolution the scoring path cannot be re-implemented twice. A different
batch size changes the bf16 reduction order; a different token count changes the
denominator; a rebuilt eval slice changes the text. Any of those is larger than the
effect being measured. So there is exactly one `perplexity` and exactly one loader,
and `load_evalsets` REFUSES to build a slice rather than silently making a new one.

The cached slice is keyed on (seed, n_docs, n_ctx), which are the only inputs that
change its contents, and lives in $ARTIFACT_DIR root rather than under a zoo_dir --
every beta and every scale scores on the same bytes.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from llmzoo.data.mixtures import DOMAINS


def evalset_path(cache_dir: str, seed: int = 7, n_docs: int = 64,
                 n_ctx: int = 1024) -> str:
    return os.path.join(cache_dir,
                        f"domain_evalset_s{seed}_n{n_docs}_c{n_ctx}.npz")


def load_evalsets(cache_dir: str, seed: int = 7, n_docs: int = 64,
                  n_ctx: int = 1024,
                  domains: Sequence[str] = DOMAINS) -> Dict[str, torch.Tensor]:
    """
    Load the cached held-out slices. Refuses to build them.

    Deliberately load-only. `eval_domains.build_evalsets` is allowed to construct a
    slice because it is the thing that defines one; anything scoring a GENERATED
    model must reuse the identical bytes the real members were scored on, and a
    silent rebuild here would compare two different test sets and report the
    difference as a generative result.
    """
    path = evalset_path(cache_dir, seed, n_docs, n_ctx)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no cached eval slice at {path}. Generated models must be scored on "
            f"byte-identical text to the real members, so this module will not "
            f"build one. Run scripts/eval_domains.py on the zoo first (it caches "
            f"the slice), or point --evalset_cache at the ARTIFACT_DIR that holds "
            f"it.")
    z = np.load(path)
    missing = [d for d in domains if d not in z.files]
    if missing:
        raise RuntimeError(
            f"{path} lacks domains {missing}; it was built for a different domain "
            f"set. Do not regenerate it in place -- the zoo was scored on the old "
            f"one.")
    return {d: torch.from_numpy(z[d]).long() for d in domains}


@torch.no_grad()
def perplexity(model, blocks, device, batch_size: int = 8) -> float:
    """
    Token-level perplexity over a fixed block tensor.

    Byte-identical to scripts/eval_domains.perplexity, which imports this. Keep the
    autocast, the float() upcast before cross_entropy and the sum-then-divide
    exactly as they are: they set the reduction order, and the numbers this
    produces are differenced against a 0.010-nat noise floor.
    """
    model.eval()
    tot_nll, tot_tok = 0.0, 0
    for i in range(0, len(blocks), batch_size):
        b = blocks[i:i + batch_size].to(device)
        x, y = b[:, :-1], b[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            logits = model(input_ids=x).logits
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1),
            reduction="sum")
        tot_nll += float(nll)
        tot_tok += y.numel()
    return float(np.exp(tot_nll / max(tot_tok, 1)))


def domain_profile(model, evalsets: Dict[str, torch.Tensor], device,
                   batch_size: int = 8,
                   domains: Sequence[str] = DOMAINS) -> Dict[str, float]:
    """The five-vector: {domain: perplexity}. This is the functional observation."""
    return {d: perplexity(model, evalsets[d], device, batch_size) for d in domains}


@torch.no_grad()
def top1(model, blocks, device, batch_size: int = 8) -> np.ndarray:
    """
    argmax over the vocabulary at every next-token position, as uint16.

    Mirrors `perplexity`'s loop exactly -- same batching, same autocast -- so a
    position here is the same position there. An argmax is far less sensitive to
    reduction order than a summed NLL, but there is no reason to introduce a second
    convention.

    uint16 is safe and deliberate: the zoo vocab is GPT-2 BPE at 50257, which fits
    under 65535. Storing logits instead would be 50257x larger.

    Lives here rather than in scripts/diag_predictions.py for the reason in this
    module's header: generated models are scored on this too (the
    generated-vs-real disagreement measurement), and the two sets of argmax tokens are compared directly against each other. A second
    implementation would be a second reduction order.
    """
    model.eval()
    out = []
    for i in range(0, len(blocks), batch_size):
        b = blocks[i:i + batch_size].to(device)
        x = b[:, :-1]
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            logits = model(input_ids=x).logits
        out.append(logits.argmax(-1).to(torch.int32).cpu().numpy().astype(np.uint16))
    return np.concatenate(out, axis=0).ravel()


def top1_profile(model, evalsets: Dict[str, torch.Tensor], device,
                 batch_size: int = 8,
                 domains: Sequence[str] = DOMAINS) -> np.ndarray:
    """
    One flat uint16 vector of argmax tokens over every domain, in `domains` order.

    The concatenation order IS the contract: `domain_spans` in a predictions.json
    slices this vector back per domain, and a consumer joining two of these must
    have built them with the same `domains` sequence.
    """
    return np.concatenate([top1(model, evalsets[d], device, batch_size)
                           for d in domains])

"""
Multiple-choice benchmark harness: question loading, tokenizers, and scoring.

These helpers live outside scripts/eval_stack.py so that compute_mc_accuracy is
not reachable only through the ENSEMBLE-ROW evaluator: a zoo member read back from
w_<i>.npy and a flow-generated model decoded from codes can be benchmarked too.
eval_stack.py imports the same names, and so does scripts/eval_generated.py.

The scoring primitives themselves stay in llmzoo.eval.core: that module takes any
nn.Module and knows nothing about PCA, flows, bundles or zoos, which is what lets a
measurement sit inside an arm loop on a model that has just had a reconstruction
written into it.

WHAT TO READ OFF A RESULT
  acc / acc_norm      argmax over summed / length-normalised choice log-likelihood.
                      Quantised at 1/n_examples. At the 51-355M scale of this zoo a
                      4-way suite sits near 0.25 and a delta smaller than a few
                      questions is not a measurement.
  gold_mean_logprob   the log score of the CORRECT continuation, per question, in
                      nats/token. Continuous, and paired across models because every
                      model sees the identical question list in the identical order.
                      This is the statistic with the power; accuracy is the one with
                      the external comparability. Seal both.
"""

from __future__ import annotations

import hashlib
import zlib
from typing import Dict, Iterable, Optional

ALL_BENCHMARKS = ("mmlu", "hellaswag", "gpqa")

# The zoo vocab IS GPT-2 BPE at 50257, even though every zoo arch is from_scratch.
# Same id as eval_domains.py's TOKENIZER_ID and eval_stack.py's from-scratch branch;
# scoring a zoo model with anything else silently measures a different tokenisation.
ZOO_TOKENIZER_ID = "gpt2"


# ---------------------------------------------------------------------------
# synthetic smoke-test questions
# ---------------------------------------------------------------------------

def _synthetic_examples(n_questions: int, seed: int) -> list:
    """
    Fixed nonsense MCExamples, for smoke tests only.

    A random-init tiny model scores at chance on any real benchmark, so reporting
    an mmlu delta there is noise dressed as a measurement -- eval_stack.main()
    refuses that. But the plumbing still needs a cheap regression test, and this
    provides one with no download and no meaning.
    """
    import random
    from llmzoo.data.mc_loader import MCExample
    rng = random.Random(seed)
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    out = []
    for _ in range(n_questions):
        ctx = " ".join(rng.choice(vocab) for _ in range(8)) + " ?"
        out.append(MCExample(context=ctx,
                             choices=[" " + rng.choice(vocab) for _ in range(4)],
                             gold_idx=rng.randrange(4)))
    return out


class _SyntheticTokenizer:
    """
    Whitespace tokenizer whose ids are guaranteed to fit a tiny model's vocab.

    A tiny model has vocab_size=1000 while the real GPT-2 tokenizer emits ids up to
    50256, so scoring synthetic questions with the real tokenizer indexes the
    embedding out of range and dies on a CUDA device-side assert. score_choices only
    ever calls tokenizer.encode(text) -> list[int], so a stub is enough.

    crc32 rather than hash(): Python salts hash() per process, and a smoke test that
    tokenizes differently on every run is not a regression test.
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = max(int(vocab_size), 2)

    def encode(self, text: str):
        # Whitespace splitting preserves the prefix property score_choices relies
        # on: encode(context + " word") starts with encode(context).
        return [1 + (zlib.crc32(w.encode()) % (self.vocab_size - 1))
                for w in text.split()]


# Public alias. The underscore names are kept because eval_stack.py already uses
# them and a rename there would show up as churn in a diff that is meant to be a
# pure move.
SyntheticTokenizer = _SyntheticTokenizer


def load_zoo_tokenizer(cache_dir: Optional[str] = None):
    """The GPT-2 BPE tokenizer every zoo architecture was trained against."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(ZOO_TOKENIZER_ID, cache_dir=cache_dir)


# ---------------------------------------------------------------------------
# question loading
# ---------------------------------------------------------------------------

# Which loaders accept a few-shot count at all. hellaswag is 0-shot by
# construction, so passing n_shots to it is a TypeError rather than a no-op.
_TAKES_N_SHOTS = {"mmlu", "gpqa"}


def load_bench_examples(benchmarks: Iterable[str], n_questions: int,
                        hf_cache: Optional[str], seed: int,
                        n_shots: Optional[int] = None) -> Dict[str, list]:
    """
    Load each benchmark's MCExample list once per process.

    Called from main(), not from a per-model loop: the questions do not depend on
    the model, so loading them per-model would re-read the datasets dozens of times.

    n_shots=None keeps each loader's own default (5 for mmlu, 3 for gpqa). Pass 0 to
    force zero-shot. That matters at this scale: GPT-2's n_positions is 1024 and
    score_choices left-truncates the context to fit, so a nominal 5-shot MMLU prompt
    is silently 1-2 shot on these models. Better to choose it than to inherit it.
    """
    from llmzoo.data.mc_loader import LOADERS
    out: Dict[str, list] = {}
    for name in benchmarks:
        if name == "synthetic":
            out[name] = _synthetic_examples(n_questions, seed)
        else:
            kw = {"n_questions": n_questions, "seed": seed, "cache_dir": hf_cache}
            if n_shots is not None and name in _TAKES_N_SHOTS:
                kw["n_shots"] = int(n_shots)
            out[name] = LOADERS[name](**kw)
        print(f"  [bench] {name}: {len(out[name])} examples")
    return out


def examples_fingerprint(examples: Iterable) -> str:
    """
    Hash the question list so a mis-paired join is detectable.

    The paired log score is only a paired statistic if every model saw the same
    questions in the same order. The loaders are deterministic given (n_questions,
    seed, n_shots), but "deterministic" is an argument and a fingerprint is a check
    -- the same reason train_flow.py seals pi_fingerprint.
    """
    h = hashlib.blake2b(digest_size=8)
    for ex in examples:
        h.update(ex.context.encode("utf-8", "replace"))
        for c in ex.choices:
            h.update(b"\x00")
            h.update(c.encode("utf-8", "replace"))
        h.update(b"\x01%d" % int(ex.gold_idx))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_bench(model, tokenizer, examples_by_bench: Dict[str, list], device,
                max_length: int, gold_logprobs: bool = True) -> Dict[str, dict]:
    """
    Every suite, scored on the model AS IT CURRENTLY IS.

    gold_logprobs=True adds the per-question log score of the correct continuation
    beside acc / acc_norm. One pass produces both.
    """
    from llmzoo.eval.core import compute_mc_accuracy
    return {name: compute_mc_accuracy(model, tokenizer, ex, device,
                                      max_length=max_length,
                                      gold_logprobs=gold_logprobs)
            for name, ex in examples_by_bench.items()}


def measure_benchmarks(model, tokenizer, examples_by_bench: Dict[str, list],
                       device, max_length: int) -> Dict[str, dict]:
    """
    eval_stack.py's view: acc / acc_norm / n_examples and nothing else.

    gold_logprobs=False keeps compute_mc_accuracy's output dict exactly as it was
    before this module existed, so eval_stack.py's JSON is byte-identical across
    the refactor.
    """
    return score_bench(model, tokenizer, examples_by_bench, device, max_length,
                       gold_logprobs=False)


def bench_deltas(original: Dict[str, dict], arm: Dict[str, dict]) -> Dict[str, dict]:
    """Pair an arm's scores with the baseline's. The delta is the signal, not the
    absolute score -- these are small base models near chance on MMLU/GPQA."""
    out: Dict[str, dict] = {}
    for name, a in arm.items():
        o = original.get(name, {})
        out[name] = {
            "acc": a["acc"], "acc_norm": a["acc_norm"],
            "acc_delta": a["acc"] - o.get("acc", float("nan")),
            "acc_norm_delta": a["acc_norm"] - o.get("acc_norm", float("nan")),
            "n_examples": a["n_examples"],
        }
    return out

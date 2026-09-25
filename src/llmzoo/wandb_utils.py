"""
Thin Weights & Biases integration, shared by every training and evaluation
entry point.

Degrades to a no-op if wandb isn't installed or logging is disabled
(--no_wandb), so nothing in the pipeline ever depends on it to run.

Usage
-----
Each standalone entrypoint's main() calls init_run() once; shared stage
functions (evaluate_family, evaluate_family_mc, ...) just call
log() unconditionally — it no-ops if no run is active, so those functions
work the same whether invoked standalone or from a multi-stage sbatch chain.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

# Used when neither project= nor $WANDB_PROJECT is given.
DEFAULT_PROJECT = "llmzoo"

# `project` is PASSED to wandb.init() below, and an explicit argument beats wandb's own
# WANDB_PROJECT env handling — so the env var has to be read here or it silently does
# nothing. Precedence: the caller's project= argument, then $WANDB_PROJECT, then the
# default.
PROJECT = os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT)

# wandb.init() silently falls back to the account's server-side "default
# team" when entity isn't passed explicitly. Set WANDB_ENTITY to pin it; None
# leaves the choice to wandb.
ENTITY = os.environ.get("WANDB_ENTITY")


def resolve_project(project: Optional[str] = None) -> str:
    """The one place the precedence rule lives, so no caller can reimplement it."""
    return project or os.environ.get("WANDB_PROJECT") or DEFAULT_PROJECT


def add_wandb_args(p) -> None:
    """
    The two routing flags, defined once so every entry point spells them the same.

    Both default to None: the project
    falls back to $WANDB_PROJECT then "llmzoo", and the group falls back to whatever
    the call site already passed (usually the Slurm job id).
    """
    p.add_argument("--wandb_project", default=None,
                   help="W&B project. Default: $WANDB_PROJECT, then 'llmzoo'. "
                        "Split by workstream, e.g. llmzoo-zoo / llmzoo-genverify.")
    p.add_argument("--wandb_group", default=None,
                   help="W&B run group. Default: the Slurm job id. Pass the "
                        "EXPERIMENT CELL when one experiment spans several jobs, "
                        "e.g. b015_k99 or zoo_gpt2_zoo_mini_b015_n100.")


def init_run(
    job_type: str,
    config: dict,
    tags: Optional[list] = None,
    enabled: bool = True,
    artifact_dir: Optional[str] = None,
    name_suffix: Optional[str] = None,
    group: Optional[str] = None,
    project: Optional[str] = None,
) -> bool:
    """
    Start a W&B run. No-ops (returns False) if wandb is missing, disabled,
    or a run is already active in this process (e.g. an orchestrator already
    opened one before calling into train.py's stage functions).

    name_suffix
        Appended to the run name. One Slurm job routinely invokes the same
        script several times -- slurm/stack_run.sbatch fits codes at k=N-1 and
        again at k=N/2, and the flow scripts add one run per rank on top of that --
        and each invocation is a separate process, so each opens its own W&B
        run. Without a suffix they all land under the identical name
        `train_stack_<jobid>` and become impossible to tell apart in the UI.
        Pass something like "k99" or "k99_codes".

    group
        W&B run group, so every run from one pipeline invocation collapses into
        a single expandable row. Defaults to the Slurm job id, which is exactly
        the right granularity: one sbatch, one group.

        Pass an EXPERIMENT CELL instead when one logical experiment spans several
        sbatch jobs -- `zoo_<arch>_<tag>_n<N>` for a zoo, `b015_k99` for a
        generative-verification cell. The job-id default cannot express that.

    project
        W&B project. Defaults to $WANDB_PROJECT, then "llmzoo".
        Split by workstream rather than by run: one project per phase keeps the runs
        table scannable, where a single project mixes zoo training, spectra and
        generative evaluation into one undifferentiated list.
    """
    if not enabled:
        return False
    if _wandb is None:
        print("[wandb] not installed — skipping W&B logging (pip install wandb to enable)")
        return False
    if _wandb.run is not None:
        return True

    # All wandb state off $HOME: home quotas are typically small, and a full
    # home fails jobs in about a second, with an opaque exit code.
    scratch = artifact_dir or os.environ.get("ARTIFACT_DIR", "./artifacts")
    os.environ.setdefault("WANDB_DIR", scratch)

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    # THE DESCRIPTIVE PART GOES FIRST, THE JOB ID LAST.
    #
    # The runs table sorts by name, and with the job id leading, a 100-way branch
    # array sorts by submission order -- which is close to random with respect to
    # the experiment, and puts Mini, Small and Medium runs interleaved. Leading
    # with the suffix makes `small_b015_member042` sit beside
    # `small_b015_member043`, and a whole scale sit together. That is the only
    # thing that makes a three-scale ladder scannable without opening runs.
    #
    # Historical runs keep the names they were created with; this only affects
    # runs created from here on.
    name = (f"{job_type}_{name_suffix}_{job_id}" if name_suffix
            else f"{job_type}_{job_id}")
    _wandb.init(
        entity=ENTITY,
        project=resolve_project(project),
        name=name,
        job_type=job_type,
        group=group or f"job_{job_id}",
        config=config,
        tags=tags or [],
    )
    return True


def log(data: dict, step: Optional[int] = None) -> None:
    if _wandb is not None and _wandb.run is not None:
        _wandb.log(data, step=step)


def summary(data: dict) -> None:
    """
    Write final scalars into the run summary rather than the history.

    Summary values are what the W&B runs TABLE shows as columns, so this is how a
    sweep over ranks and flow spaces becomes scannable at a glance instead of
    requiring each run to be opened. Use it for terminal values (final KL, best
    loss, fitted rank), and log() for anything that varies per epoch.
    """
    if _wandb is not None and _wandb.run is not None:
        _wandb.run.summary.update(data)


def run_id() -> Optional[str]:
    """
    The active run's W&B id, or None.

    Exists so a checkpointed job can seal the id of the run that wrote the
    checkpoint, and the resumed job can seal the ids that preceded it -- which
    makes a member that spans several Slurm jobs walkable in both directions
    instead of appearing as unrelated duplicates in the same group.

    No-ops to None exactly like log() and summary(), so callers never have to
    know whether W&B is active.
    """
    if _wandb is not None and _wandb.run is not None:
        return _wandb.run.id
    return None


def finish() -> None:
    if _wandb is not None and _wandb.run is not None:
        _wandb.finish()

from collections.abc import Callable

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jax import Array
from tqdm import trange

from wasabi.fitting import REFIT_FIELDS, TrialBuffer, refit
from wasabi.inference.ekf import NUM_ITERS
from wasabi.inference.joint import BPosterior, filter_trial
from wasabi.model import StateSpaceModel
from wasabi.selection import SelectionStrategy, TrialDesign


def observe_trial(
    model: StateSpaceModel,
    prior: BPosterior,
    buffer: TrialBuffer,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    ys: Array,
    ss: Array,
    keep: Array,
    refit_steps: int = 20,
    refit_fields: tuple[str, ...] = REFIT_FIELDS,
    filter_iters: int = NUM_ITERS,
) -> tuple[StateSpaceModel, BPosterior, TrialBuffer, optax.OptState, Array]:
    """Incorporate one trial.

    Runs the joint EKF over [x; vec(B)] with the dynamics fixed, refits A, C, d, Q, b on
    every trial so far with B fixed at its posterior mean, then re-expresses the B
    posterior in the refit latent basis.
    """
    _, prior, _ = filter_trial(model, prior, ys, ss, keep, filter_iters)
    buffer = buffer.add(ys, ss, keep)
    refit_model, opt_state, losses = refit(
        model, opt_state, prior.B, buffer, optimizer, refit_steps, refit_fields
    )
    prior = prior.realign(refit_model.basis_change(model))

    return refit_model, prior, buffer, opt_state, losses


def _record(history, choice, prior, losses, model, evaluate):
    history["choice"].append(choice)
    history["logdet"].append(float(prior.logdet()))
    history["refit_loss"].append(float(losses[-1]) if losses.size else float("nan"))
    if evaluate is not None:
        for name, value in evaluate(model, prior).items():
            history.setdefault(name, []).append(value)


def run_adaptive(
    model: StateSpaceModel,
    prior: BPosterior,
    strategy: SelectionStrategy,
    candidates: np.ndarray,
    design: TrialDesign,
    simulate: Callable[[np.ndarray], np.ndarray],
    buffer: TrialBuffer,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    num_trials: int,
    key: Array,
    refit_steps: int = 20,
    refit_fields: tuple[str, ...] = REFIT_FIELDS,
    filter_iters: int = NUM_ITERS,
    evaluate: Callable[[StateSpaceModel, BPosterior], dict] | None = None,
) -> tuple[StateSpaceModel, BPosterior, TrialBuffer, dict]:
    """Closed-loop identification of B: pick a pattern, observe it, `observe_trial`.

    `optimizer`/`opt_state` must come from `init_optimizer` with the same `refit_fields`.

    `simulate(pattern)` returns the (T, N) counts of one trial, NaN where blanked.
    `evaluate(model, prior)` can return extra per-trial diagnostics, e.g. B error.
    """
    history = {"choice": [], "logdet": [], "refit_loss": []}

    for trial in (progress_bar := trange(num_trials)):
        key, subkey = jr.split(key)
        choice = strategy.select(subkey, model, prior, candidates, design, trial)

        ss = design.schedule(candidates[choice])
        ys = jnp.asarray(simulate(candidates[choice]))
        model, prior, buffer, opt_state, losses = observe_trial(
            model, prior, buffer, optimizer, opt_state, ys, ss, design.keep,
            refit_steps, refit_fields, filter_iters,
        )

        _record(history, choice, prior, losses, model, evaluate)
        progress_bar.set_postfix_str(
            f"logdet {history['logdet'][-1]:.1f}, NLL {history['refit_loss'][-1]:.4f}"
        )

    return model, prior, buffer, {name: np.array(v) for name, v in history.items()}


def run_batch(
    model: StateSpaceModel,
    prior: BPosterior,
    Ys: Array,
    Ss: Array,
    keep: Array,
    buffer: TrialBuffer,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    refit_steps: int = 20,
    refit_fields: tuple[str, ...] = REFIT_FIELDS,
    filter_iters: int = NUM_ITERS,
    order: np.ndarray | None = None,
    evaluate: Callable[[StateSpaceModel, BPosterior], dict] | None = None,
) -> tuple[StateSpaceModel, BPosterior, TrialBuffer, dict]:
    """Replay recorded trials through `observe_trial`, in `order` (default: as given).

    keep is (T,) shared or (n_trials, T). The "choice" history holds trial indices.
    """
    keep = jnp.broadcast_to(keep, Ys.shape[:2])
    order = np.arange(len(Ys)) if order is None else np.asarray(order)
    history = {"choice": [], "logdet": [], "refit_loss": []}

    for i in (progress_bar := trange(len(order))):
        j = int(order[i])
        model, prior, buffer, opt_state, losses = observe_trial(
            model, prior, buffer, optimizer, opt_state, Ys[j], Ss[j], keep[j],
            refit_steps, refit_fields, filter_iters,
        )

        _record(history, j, prior, losses, model, evaluate)
        progress_bar.set_postfix_str(
            f"logdet {history['logdet'][-1]:.1f}, NLL {history['refit_loss'][-1]:.4f}"
        )

    return model, prior, buffer, {name: np.array(v) for name, v in history.items()}

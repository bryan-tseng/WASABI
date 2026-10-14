"""Fitting the StateSpaceModel parameters (A, C, d, Q, b by default) with B held fixed.

Each step is an optax update on the masked EKF marginal likelihood followed by
`StateSpaceModel.project` (e.g. orthonormal C, clipped A), the same "gradient step,
then project" pattern as decomposed-dynamics' `regularize_operators`.

Refit fields are dotted paths into the model, e.g. "dynamics.A" or "observation.C".
"""

from functools import reduce

from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from wasabi.inference.ekf import NUM_ITERS, filter_latents
from wasabi.inference.joint import BPosterior, filter_trials
from wasabi.model import StateSpaceModel


class TrialBuffer(eqx.Module):
    """Fixed-capacity store of observed trials.

    The capacity is fixed so jitted refits compile once, no matter how many trials
    have been collected. Unused slots have valid == False and contribute nothing.
    """

    ys: Array  # (capacity, T, N) spike counts, NaN where blanked
    ss: Array  # (capacity, T, N) stim patterns
    keep: Array  # (capacity, T) False where blanked
    valid: Array  # (capacity,)
    size: Array

    @classmethod
    def empty(cls, capacity: int, num_timesteps: int, num_channels: int) -> Self:
        return cls(
            ys=jnp.zeros((capacity, num_timesteps, num_channels)),
            ss=jnp.zeros((capacity, num_timesteps, num_channels)),
            keep=jnp.zeros((capacity, num_timesteps), dtype=bool),
            valid=jnp.zeros(capacity, dtype=bool),
            size=jnp.array(0),
        )

    @eqx.filter_jit
    def add(self, ys: Array, ss: Array, keep: Array) -> Self:
        # out-of-bounds .at[].set is silently dropped in JAX, so fail loudly instead
        i = eqx.error_if(self.size, self.size >= self.valid.shape[0], "TrialBuffer is full")
        return TrialBuffer(
            ys=self.ys.at[i].set(ys),
            ss=self.ss.at[i].set(ss),
            keep=self.keep.at[i].set(keep),
            valid=self.valid.at[i].set(True),
            size=i + 1,
        )

    def extend(self, Ys: Array, Ss: Array, keep: Array) -> Self:
        keep = jnp.broadcast_to(keep, Ys.shape[:2])
        buffer = self
        for ys, ss, k in zip(Ys, Ss, keep):
            buffer = buffer.add(ys, ss, k)
        return buffer


REFIT_FIELDS = (
    "dynamics.A",
    "dynamics.Q_chol",
    "dynamics.b",
    "observation.C",
    "observation.d",
)


def _get_field(model: eqx.Module, path: str):
    return reduce(getattr, path.split("."), model)


def trainable_spec(
    model: StateSpaceModel, fields: tuple[str, ...] = REFIT_FIELDS
) -> StateSpaceModel:
    """Filter spec selecting the refit parameters; everything else stays fixed.

    Caveat on b: the B posterior is expressed in units of the current stim input
    u(s; b), so refitting b between trials rescales it without the posterior being
    rescaled (with V = I, b only acts as a gain on u). The drift per trial is small and
    the filter absorbs it; in scripts/joint_ekf_simulation.py refitting b still gives a
    lower stim-effect error than freezing it.
    """
    spec = jax.tree.map(lambda _: False, model)
    return eqx.tree_at(
        lambda m: tuple(_get_field(m, f) for f in fields), spec, replace=(True,) * len(fields)
    )


# The marginal likelihood uses the plain EKF update: iterating it doubles the cost of
# every gradient step without improving the fits.
LIKELIHOOD_EKF_ITERS = 1


def negative_loglik(
    model: StateSpaceModel,
    B: Array,
    buffer: TrialBuffer,
    ekf_iters: int = LIKELIHOOD_EKF_ITERS,
) -> Array:
    """Masked EKF marginal negative log likelihood per kept bin, over valid trials."""
    loglik, *_ = jax.vmap(filter_latents, in_axes=(None, None, 0, 0, 0, None))(
        model, B, buffer.ys, buffer.ss, buffer.keep, ekf_iters
    )
    num_bins = jnp.maximum((buffer.keep & buffer.valid[:, None]).sum(), 1)
    return -jnp.where(buffer.valid, loglik, 0.0).sum() / num_bins


@eqx.filter_jit
def _loss(diff_model, static_model, B, buffer, ekf_iters):
    return negative_loglik(eqx.combine(diff_model, static_model), B, buffer, ekf_iters)


@eqx.filter_jit
def _step(
    model: StateSpaceModel,
    opt_state: optax.OptState,
    B: Array,
    buffer: TrialBuffer,
    optimizer: optax.GradientTransformation,
    fields: tuple[str, ...],
    ekf_iters: int,
):
    diff_model, static_model = eqx.partition(model, trainable_spec(model, fields))
    loss, grads = eqx.filter_value_and_grad(_loss)(
        diff_model, static_model, B, buffer, ekf_iters
    )
    updates, opt_state = optimizer.update(grads, opt_state, diff_model)
    model = eqx.apply_updates(model, updates).project()

    return model, opt_state, loss


def init_optimizer(
    model: StateSpaceModel, learning_rate: float = 5e-3, fields: tuple[str, ...] = REFIT_FIELDS
) -> tuple[optax.GradientTransformation, optax.OptState]:
    """Adam over `fields`. Pass the same `fields` to `refit`."""
    optimizer = optax.adam(learning_rate)
    return optimizer, optimizer.init(eqx.filter(model, trainable_spec(model, fields)))


def refit(
    model: StateSpaceModel,
    opt_state: optax.OptState,
    B: Array,
    buffer: TrialBuffer,
    optimizer: optax.GradientTransformation,
    num_steps: int,
    fields: tuple[str, ...] = REFIT_FIELDS,
    ekf_iters: int = LIKELIHOOD_EKF_ITERS,
) -> tuple[StateSpaceModel, optax.OptState, Array]:
    """Warm-started refit of `fields` on all trials in `buffer`, with B fixed."""
    losses = []
    for _ in range(num_steps):
        model, opt_state, loss = _step(
            model, opt_state, B, buffer, optimizer, fields, ekf_iters
        )
        losses.append(loss)

    return model, opt_state, jnp.array(losses)


def fit_initial(
    model: StateSpaceModel,
    buffer: TrialBuffer,
    num_steps: int = 500,
    learning_rate: float = 5e-3,
) -> tuple[StateSpaceModel, Array]:
    """Fit on control trials before any stim. Stim input is zero there, so B drops out."""
    optimizer, opt_state = init_optimizer(model, learning_rate)
    B = jnp.zeros((model.state_dim, model.num_channels))
    model, _, losses = refit(model, opt_state, B, buffer, optimizer, num_steps)

    return model, losses


def fit_batch(
    model: StateSpaceModel,
    prior: BPosterior,
    buffer: TrialBuffer,
    num_iters: int = 10,
    refit_steps: int = 50,
    learning_rate: float = 5e-3,
    fields: tuple[str, ...] = REFIT_FIELDS,
    filter_iters: int = NUM_ITERS,
) -> tuple[StateSpaceModel, BPosterior, Array]:
    """EM over a fixed set of recorded trials.

    E-step: the joint EKF over every trial in `buffer`, carrying the B posterior from
    `prior`. M-step: `refit_steps` optimizer steps on `fields` with B at its posterior
    mean. Every E-step restarts from `prior`, so a zero-mean isotropic prior needs no
    re-alignment when the latent basis rotates, and the b caveat of `trainable_spec` does
    not apply since the B posterior is rebuilt under the current b. `filter_iters` > 1 uses the
    iterated EKF in the E-step. Returns the model, the final posterior over B, and the
    refit losses of each iteration.
    """
    valid = int(buffer.size)
    Ys, Ss, keep = buffer.ys[:valid], buffer.ss[:valid], buffer.keep[:valid]
    optimizer, opt_state = init_optimizer(model, learning_rate, fields)

    losses = []
    for _ in range(num_iters):
        _, posterior = filter_trials(model, prior, Ys, Ss, keep, filter_iters)
        model, opt_state, iter_losses = refit(
            model, opt_state, posterior.B, buffer, optimizer, refit_steps, fields
        )
        losses.append(iter_losses)

    _, posterior = filter_trials(model, prior, Ys, Ss, keep, filter_iters)
    return model, posterior, jnp.stack(losses)

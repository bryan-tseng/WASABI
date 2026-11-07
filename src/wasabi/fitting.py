"""Fitting the StateSpaceModel parameters (A, C, d, Q, b and the stim-response variance by default) with B held fixed.

Each step is an optax update on the masked EKF marginal likelihood followed by
`StateSpaceModel.project` for your constraints
For example: (orthonormal C, clipped A)

Fields in the model are dotted paths, e.g. "dynamics.A, observation.C"
"""

from functools import partial, reduce

from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from wasabi.inference.ekf import filter_latents
from wasabi.inference.joint import BPosterior, filter_trials
from wasabi.inference.relinearize import RELINEARIZE_ITERS
from wasabi.model import StateSpaceModel

REFIT_FIELDS = (
    "dynamics.A",
    "dynamics.Q_chol",
    "dynamics.b",
    "dynamics.log_stim_var",
    "observation.C",
    "observation.d",
)

class TrialBuffer(eqx.Module):
    """Fixed-capacity store of observed trials.

    The capacity is fixed so jitted refits compile once, no matter how many trials
    have been collected. Unused slots have valid == False and contribute nothing.
    """

    ys: Array  # (capacity, T, N) spike counts, NaN where blanked
    ss: Array  # (capacity, T, S) stim patterns, S stim channels (S = N unless given)
    keep: Array  # (capacity, T) mask which binds are kept
    valid: Array  # (capacity,) True as trials are added, False for empty slots
    size: Array

    @classmethod
    def empty(
        cls, capacity: int, num_timesteps: int, num_channels: int, num_stim_channels: int | None = None
    ) -> Self:
        # Initialize with zeros before filling trials
        num_stim_channels = num_channels if num_stim_channels is None else num_stim_channels
        return cls(
            ys=jnp.zeros((capacity, num_timesteps, num_channels)),
            ss=jnp.zeros((capacity, num_timesteps, num_stim_channels)),
            keep=jnp.zeros((capacity, num_timesteps), dtype=bool),
            valid=jnp.zeros(capacity, dtype=bool),
            size=jnp.array(0),
        )

    @eqx.filter_jit
    def add(self, ys: Array, ss: Array, keep: Array) -> Self:
        # Report error if our of bounds
        i = eqx.error_if(self.size, self.size >= self.valid.shape[0], "Index is out of bounds")
        return TrialBuffer(
            ys=self.ys.at[i].set(ys),
            ss=self.ss.at[i].set(ss),
            keep=self.keep.at[i].set(keep),
            valid=self.valid.at[i].set(True),
            size=i + 1,
        )

    def extend(self, Ys: Array, Ss: Array, keep: Array) -> Self:
        """Adding many trials at once, such as fitting control trials"""
        keep = jnp.broadcast_to(keep, Ys.shape[:2])
        buffer = self
        for ys, ss, k in zip(Ys, Ss, keep):
            buffer = buffer.add(ys, ss, k)
        return buffer

def _get_field(model: eqx.Module, path: str):
    return reduce(getattr, path.split("."), model)

def trainable_spec(
    model: StateSpaceModel, fields: tuple[str, ...] = REFIT_FIELDS
) -> StateSpaceModel:
    """Filter spec selecting the refit parameters; everything else stays fixed (i.e. False for non-refit params).

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

def negative_loglik(
    model: StateSpaceModel,
    B: Array,
    buffer: TrialBuffer,
    B_cov: Array | None = None,
    relinearize_iters: int = RELINEARIZE_ITERS,
) -> Array:
    """
    Masked EKF marginal negative log likelihood per kept bin. This is our loss.
    in_axes=(None, None, 0, 0, 0) flags model and B as static and broadcast across trials
    y, s, and keep, which are (T, N) per trial
    Hence the negll is averaged over trials
    B_cov, the posterior covariance of vec(B), is added to the noise of stim bins so that
    B's own uncertainty is not mistaken for trial-to-trial stim variability.
    relinearize_iters: see `filter_latents`; gradients flow through the final pass only.
    """
    filter_one = partial(filter_latents, relinearize_iters=relinearize_iters)
    loglik, *_ = jax.vmap(filter_one, in_axes=(None, None, 0, 0, 0, None))(
        model, B, buffer.ys, buffer.ss, buffer.keep, B_cov
    )
    num_bins = jnp.maximum((buffer.keep & buffer.valid[:, None]).sum(), 1)
    return -jnp.where(buffer.valid, loglik, 0.0).sum() / num_bins


@eqx.filter_jit
def _loss(diff_model, static_model, B, buffer, B_cov, relinearize_iters):
    return negative_loglik(
        eqx.combine(diff_model, static_model), B, buffer, B_cov, relinearize_iters
    )


@eqx.filter_jit
def _step(
    model: StateSpaceModel,
    opt_state: optax.OptState,
    B: Array,
    buffer: TrialBuffer,
    optimizer: optax.GradientTransformation,
    fields: tuple[str, ...],
    B_cov: Array | None,
    relinearize_iters: int,
):
    diff_model, static_model = eqx.partition(model, trainable_spec(model, fields))
    # Split into trainable vs static for jit
    loss, grads = eqx.filter_value_and_grad(_loss)(
        diff_model, static_model, B, buffer, B_cov, relinearize_iters
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
    B_cov: Array | None = None,
    relinearize_iters: int = RELINEARIZE_ITERS,
) -> tuple[StateSpaceModel, optax.OptState, Array]:
    """Warm-started refit of `fields` on all trials in `buffer`, with B fixed (see
    `negative_loglik` for `B_cov` and `relinearize_iters`)."""
    losses = []
    for _ in range(num_steps):
        model, opt_state, loss = _step(
            model, opt_state, B, buffer, optimizer, fields, B_cov, relinearize_iters
        )
        losses.append(loss)

    return model, opt_state, jnp.array(losses)


def fit_initial(
    model: StateSpaceModel,
    buffer: TrialBuffer,
    num_steps: int = 500,
    learning_rate: float = 5e-3,
    relinearize_iters: int = RELINEARIZE_ITERS,
) -> tuple[StateSpaceModel, Array]:
    """Fit on control trials before any stim. Stim input is zero there, so B drops out."""
    optimizer, opt_state = init_optimizer(model, learning_rate)
    B = jnp.zeros((model.state_dim, model.num_inputs))
    model, _, losses = refit(
        model, opt_state, B, buffer, optimizer, num_steps, relinearize_iters=relinearize_iters
    )

    return model, losses


def fit_batch(
    model: StateSpaceModel,
    prior: BPosterior,
    buffer: TrialBuffer,
    num_iters: int = 10,
    refit_steps: int = 50,
    learning_rate: float = 5e-3,
    fields: tuple[str, ...] = REFIT_FIELDS,
    relinearize_iters: int = RELINEARIZE_ITERS,
) -> tuple[StateSpaceModel, BPosterior, Array]:
    """EM over a fixed set of recorded trials.

    E-step:
        1. Joint EKF over every trial in `buffer`
        2. Carrying the B posterior from `prior`
    M-step:
        1. `refit_steps` optimizer steps on `fields` with B at its posterior mean.
        2. Every E-step restarts from `prior`, so a zero-mean isotropic prior needs no
        re-alignment when the latent basis rotates, and the b caveat of `trainable_spec` does not apply since the B posterior is rebuilt under the current b.
    Returns:
        The model, the final posterior over B, and the refit losses of each iteration.
    """
    valid = int(buffer.size)
    Ys, Ss, keep = buffer.ys[:valid], buffer.ss[:valid], buffer.keep[:valid]
    optimizer, opt_state = init_optimizer(model, learning_rate, fields)

    losses = []
    for _ in range(num_iters):
        _, posterior = filter_trials(model, prior, Ys, Ss, keep, relinearize_iters)
        model, opt_state, iter_losses = refit(
            model, opt_state, posterior.B, buffer, optimizer, refit_steps, fields, posterior.cov,
            relinearize_iters,
        )
        losses.append(iter_losses)

    _, posterior = filter_trials(model, prior, Ys, Ss, keep, relinearize_iters)
    return model, posterior, jnp.stack(losses)

"""Blanking-aware EKF for a `StateSpaceModel`.

The measurement update is shared between the latent-only filter (state x) here and the joint filter (state [x; vec(B)]) in joint.py.

Time convention (matches simulations.simulate_trial): s[t] drives
the transition into x_t, then y_t is observed.

The prior (mu0, Sigma0) is on x_{-1}.

With `relinearize_iters > 0` the observation is re-linearized around the smoothed trajectory
and the filter is run again (the damped iterated smoother of relinearize.py); 0 is the plain EKF.
"""

from functools import partial


import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

from wasabi.inference.relinearize import RELINEARIZE_ITERS, damped_gauss_newton
from wasabi.model import StateSpaceModel
from wasabi.utils import symmetrize


class Posterior(eqx.Module):
    marginal_loglik: Array
    filtered_means: Array
    filtered_covs: Array
    smoothed_means: Array
    smoothed_covs: Array


def ekf_update(
    model: StateSpaceModel,
    mean: Array,
    cov: Array,
    y: Array,
    keep: Array,
    lin_x: Array | None = None,
) -> tuple[Array, Array, Array]:
    """One EKF measurement update. Returns (loglik, mean, cov).

    The rate is linearized at `lin_x`, by default the predicted latents mean[:K]; the
    iterated smoother passes the smoothed latents of its previous pass instead.

    keep == False means the bin was blanked: the update is the identity and the bin
    adds nothing to the log likelihood. The log likelihood is the plug-in
    likelihood at the predicted mean, as in the EKF marginal likelihood approximation.
    """
    K = model.state_dim
    filter_mean_x, P_xx = mean[:K], cov[:K, :K]
    lin_x = filter_mean_x if lin_x is None else lin_x
    y = jnp.where(keep, y, 0.0)  # blanked bins hold NaN so switch to 0

    # Jacobian of the rate at the linearization point gives H
    rate = model.rate(lin_x)
    H = jax.jacfwd(model.rate)(lin_x)
    HtRinv = H.T / model.noise_var(rate)

    HtSinv = jnp.linalg.solve(jnp.eye(K) + HtRinv @ H @ P_xx, HtRinv)
    cross_cov = cov[:, :K]
    kalman_gain = cross_cov @ HtSinv
    # Innovation against the linearized rate, rate(lin) + H (x - lin), at the predicted mean
    innovation = y - rate - H @ (filter_mean_x - lin_x)
    mean_updated = mean + kalman_gain @ innovation
    cov_updated = symmetrize(cov - cross_cov @ (HtSinv @ H) @ cov[:K, :])
    loglik = model.log_prob(y, model.rate(filter_mean_x))

    return (
        jnp.where(keep, loglik, 0.0),
        jnp.where(keep, mean_updated, mean),
        jnp.where(keep, cov_updated, cov),
    )


def stim_effect_cov(B_cov: Array, u: Array) -> Array:
    """Covariance of B u when vec(B) has covariance B_cov: (uᵀ ⊗ I_K) B_cov (u ⊗ I_K)."""
    N = u.shape[0]
    K = B_cov.shape[0] // N
    return jnp.einsum("n, nimj, m -> ij", u, B_cov.reshape(N, K, N, K), u)


def process_noises(model: StateSpaceModel, ss: Array, B_cov: Array | None = None) -> Array:
    """(T, K, K) process noise of every bin, computed once per trial and reused by every pass.

    With `B_cov`, B's uncertainty is added to the noise of stim bins, treating it as
    independent from bin to bin.
    """

    def noise(s):
        Q = model.process_noise(s)
        if B_cov is not None:
            Q = Q + stim_effect_cov(B_cov, model.stim_input(s))
        return Q

    return jax.vmap(noise)(ss)


def predict(
    model: StateSpaceModel, mean: Array, cov: Array, B: Array, s: Array, Q: Array | None = None
) -> tuple[Array, Array]:
    """Predictive mean and covariance; Q defaults to the model's process noise for s."""
    A = model.jacobian(mean)
    Q = model.process_noise(s) if Q is None else Q
    return model.transition(mean, B, s), symmetrize(A @ cov @ A.T + Q)


def trajectory_log_density(
    model: StateSpaceModel, xs: Array, B: Array, ys: Array, ss: Array, keep: Array, Qs: Array
) -> Array:
    """log p(x_{0:T-1}, y | B) up to a constant: the objective the iterated smoother climbs.

    x_0 is Gaussian around the transition of mu0 (the prior is on x_{-1}), each later x_t
    around the transition of x_{t-1}, and every kept bin adds its Poisson log likelihood.
    """
    A0 = model.jacobian(model.mu0)
    previous = jnp.concatenate([model.mu0[None], xs[:-1]])
    means = jax.vmap(model.transition, in_axes=(0, None, 0))(previous, B, ss)
    covs = Qs.at[0].add(A0 @ model.Sigma0 @ A0.T)
    dynamics = jax.vmap(jax.scipy.stats.multivariate_normal.logpdf)(xs, means, covs).sum()
    counts = jax.vmap(lambda x, y: model.log_prob(y, model.rate(x)))(
        xs, jnp.where(keep[:, None], ys, 0.0)
    )
    return dynamics + jnp.where(keep, counts, 0.0).sum()


def _filter(model, B, ys, ss, keep, Qs, lin_xs):
    """One filter pass, the observation linearized at `lin_xs` (None: the predicted means)."""

    def step(carry, inputs):
        loglik, mean, cov = carry
        y, s, k, Q, lin = inputs
        pred_mean, pred_cov = predict(model, mean, cov, B, s, Q)
        ll, mean, cov = ekf_update(model, pred_mean, pred_cov, y, k, lin)
        return (loglik + ll, mean, cov), (mean, cov, pred_mean, pred_cov)

    (loglik, _, _), (filtered_means, filtered_covs, predicted_means, predicted_covs) = lax.scan(
        step, (jnp.array(0.0), model.mu0, model.Sigma0), (ys, ss, keep, Qs, lin_xs)
    )
    return loglik, filtered_means, filtered_covs, predicted_means, predicted_covs


def _rts(model, filtered_means, filtered_covs, predicted_means, predicted_covs):
    """RTS backward pass; the last bin's smoothed state is its filtered state."""

    def backward(carry, t):
        smoothed_means_next, smoothed_covs_next = carry
        G = jnp.linalg.solve(predicted_covs[t + 1], model.jacobian(filtered_means[t]) @ filtered_covs[t]).T
        smoothed_means = filtered_means[t] + G @ (smoothed_means_next - predicted_means[t + 1])
        smoothed_covs = symmetrize(filtered_covs[t] + G @ (smoothed_covs_next - predicted_covs[t + 1]) @ G.T)
        return (smoothed_means, smoothed_covs), (smoothed_means, smoothed_covs)

    T = filtered_means.shape[0]
    _, (smoothed_means, smoothed_covs) = lax.scan(
        backward, (filtered_means[-1], filtered_covs[-1]), jnp.arange(T - 1), reverse=True
    )
    return (
        jnp.concatenate([smoothed_means, filtered_means[-1:]]),
        jnp.concatenate([smoothed_covs, filtered_covs[-1:]]),
    )


def smoothed_path(
    model: StateSpaceModel, B: Array, ys: Array, ss: Array, keep: Array, Qs: Array, lin_xs: Array | None
) -> Array:
    """(T, K) smoothed latent means of one filter + RTS pass, linearized at `lin_xs`."""
    return _rts(model, *_filter(model, B, ys, ss, keep, Qs, lin_xs)[1:])[0]


def _linearization(model, B, ys, ss, keep, Qs, relinearize_iters):
    """Where to linearize the observation: None for the plain EKF, else the damped
    Gauss-Newton refinement of the smoothed path. No gradient flows through it."""
    if relinearize_iters == 0:
        return None

    def smooth(lin_xs):
        return smoothed_path(model, B, ys, ss, keep, Qs, lin_xs)

    xs = damped_gauss_newton(
        smooth,
        lambda xs: trajectory_log_density(model, xs, B, ys, ss, keep, Qs),
        smooth(None),
        relinearize_iters - 1,
    )
    return lax.stop_gradient(xs)


@eqx.filter_jit
def filter_latents(
    model: StateSpaceModel,
    B: Array,
    ys: Array,
    ss: Array,
    keep: Array,
    B_cov: Array | None = None,
    relinearize_iters: int = RELINEARIZE_ITERS,
) -> tuple[Array, Array, Array, Array, Array]:
    """Latent-only filter with B held fixed, at its posterior mean if `B_cov` is given.

    Returns (loglik, filtered_means, filtered_covs, predicted_means, predicted_covs) of the
    final pass.
    """
    Qs = process_noises(model, ss, B_cov)
    lin_xs = _linearization(model, B, ys, ss, keep, Qs, relinearize_iters)
    return _filter(model, B, ys, ss, keep, Qs, lin_xs)


@eqx.filter_jit
def smooth_latents(
    model: StateSpaceModel,
    B: Array,
    ys: Array,
    ss: Array,
    keep: Array,
    relinearize_iters: int = RELINEARIZE_ITERS,
) -> Posterior:
    """Latent-only filter plus RTS backward pass, with B held fixed."""
    Qs = process_noises(model, ss)
    lin_xs = _linearization(model, B, ys, ss, keep, Qs, relinearize_iters)
    loglik, filtered_means, filtered_covs, predicted_means, predicted_covs = _filter(
        model, B, ys, ss, keep, Qs, lin_xs
    )
    smoothed_means, smoothed_covs = _rts(
        model, filtered_means, filtered_covs, predicted_means, predicted_covs
    )
    return Posterior(loglik, filtered_means, filtered_covs, smoothed_means, smoothed_covs)


@eqx.filter_jit
def batch_smooth_latents(
    model: StateSpaceModel,
    B: Array,
    Ys: Array,
    Ss: Array,
    keep: Array,
    relinearize_iters: int = RELINEARIZE_ITERS,
) -> Posterior:
    """vmap of `smooth_latents` over trials
    keep is (T,) shared or (n_trials, T)
    """
    keep = jnp.broadcast_to(keep, Ys.shape[:2])
    smooth = partial(smooth_latents, relinearize_iters=relinearize_iters)
    return jax.vmap(smooth, in_axes=(None, None, 0, 0, 0))(model, B, Ys, Ss, keep)

"""Blanking-aware EKF for a `StateSpaceModel`.

The measurement update is shared between the latent-only filter (state x) here and the joint filter (state [x; vec(B)]) in joint.py.

Time convention (matches simulations.simulate_trial): s[t] drives
the transition into x_t, then y_t is observed.

The prior (mu0, Sigma0) is on x_{-1}.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

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
) -> tuple[Array, Array, Array]:
    """One EKF measurement update. Returns (loglik, mean, cov).

    keep == False means the bin was blanked: the update is the identity and the bin
    adds nothing to the log likelihood. The log likelihood is the plug-in
    likelihood at the predicted mean, as in the EKF marginal likelihood approximation.
    """
    K = model.state_dim
    filter_mean_x, P_xx = mean[:K], cov[:K, :K]
    y = jnp.where(keep, y, 0.0)  # blanked bins hold NaN so switch to 0

    # Jacobian of the rate at the predicted latents gives H
    rate = model.rate(filter_mean_x)
    H = jax.jacfwd(model.rate)(filter_mean_x)
    HtRinv = H.T / model.noise_var(rate)

    HtSinv = jnp.linalg.solve(jnp.eye(K) + HtRinv @ H @ P_xx, HtRinv)
    cross_cov = cov[:, :K]
    kalman_gain = cross_cov @ HtSinv
    mean_updated = mean + kalman_gain @ (y - rate)
    cov_updated = symmetrize(cov - cross_cov @ (HtSinv @ H) @ cov[:K, :])
    loglik = model.log_prob(y, rate)

    return (
        jnp.where(keep, loglik, 0.0),
        jnp.where(keep, mean_updated, mean),
        jnp.where(keep, cov_updated, cov),
    )


def predict(
    model: StateSpaceModel, mean: Array, cov: Array, B: Array, s: Array
) -> tuple[Array, Array]:
    # Predictive mean and covariance
    A = model.jacobian(mean)
    return model.transition(mean, B, s), symmetrize(A @ cov @ A.T + model.Q)


@eqx.filter_jit
def filter_latents(
    model: StateSpaceModel,
    B: Array,
    ys: Array,
    ss: Array,
    keep: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Latent-only filter with B held fixed.

    Returns (loglik, filtered_means, filtered_covs, predicted_means, predicted_covs).
    """

    def step(carry, inputs):
        loglik, mean, cov = carry
        y, s, k = inputs
        
        # Predict step
        pred_mean, pred_cov = predict(model, mean, cov, B, s)
        ll, mean, cov = ekf_update(model, pred_mean, pred_cov, y, k)
        return (loglik + ll, mean, cov), (mean, cov, pred_mean, pred_cov)

    (loglik, _, _), (filtered_means, filtered_covs, predicted_means, predicted_covs) = lax.scan(
        step, (jnp.array(0.0), model.mu0, model.Sigma0), (ys, ss, keep)
    )
    return loglik, filtered_means, filtered_covs, predicted_means, predicted_covs


@eqx.filter_jit
def smooth_latents(model: StateSpaceModel, B: Array, ys: Array, ss: Array, keep: Array) -> Posterior:
    """Latent-only filter plus RTS backward pass, with B held fixed."""
    # Unpack filtered latents
    loglik, filtered_means, filtered_covs, predicted_means, predicted_covs = filter_latents(model, B, ys, ss, keep)

    def backward(carry, t):
        smoothed_means_next, smoothed_covs_next = carry
        G = jnp.linalg.solve(predicted_covs[t + 1], model.jacobian(filtered_means[t]) @ filtered_covs[t]).T
        smoothed_means = filtered_means[t] + G @ (smoothed_means_next - predicted_means[t + 1])
        smoothed_covs = symmetrize(filtered_covs[t] + G @ (smoothed_covs_next - predicted_covs[t + 1]) @ G.T)
        return (smoothed_means, smoothed_covs), (smoothed_means, smoothed_covs)

    T = ys.shape[0]
    _, (smoothed_means, smoothed_covs) = lax.scan(backward, (filtered_means[-1], filtered_covs[-1]), jnp.arange(T - 1), reverse=True)
    
    # Last bin is the same
    smoothed_means = jnp.concatenate([smoothed_means, filtered_means[-1:]])
    smoothed_covs = jnp.concatenate([smoothed_covs, filtered_covs[-1:]])

    return Posterior(loglik, filtered_means, filtered_covs, smoothed_means, smoothed_covs)


@eqx.filter_jit
def batch_smooth_latents(
    model: StateSpaceModel, B: Array, Ys: Array, Ss: Array, keep: Array
) -> Posterior:
    """vmap of `smooth_latents` over trials
    keep is (T,) shared or (n_trials, T)
    """
    keep = jnp.broadcast_to(keep, Ys.shape[:2])
    return jax.vmap(smooth_latents, in_axes=(None, None, 0, 0, 0))(model, B, Ys, Ss, keep)

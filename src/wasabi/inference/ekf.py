"""Blanking-aware EKF for a `StateSpaceModel`, with an optional iterated update.

The measurement update is shared by the latent-only filter (state x) and the joint
filter (state [x; vec(B)]). In both, the emission depends only on the first K state
entries, so the update is done in x space and carried to the rest of the state through
the cross covariance:

1. Move x toward the mode x* of N(x; m_x, P_xx) * p(y | rate(x)) with Gauss-Newton
   steps. One step from m_x is the standard EKF update (the default); more steps give
   the iterated EKF. The steps are not damped, so iterating can overshoot and oscillate
   around x* when the counts are far from the predicted rate.
2. Linearize at the last linearization point: H = d rate / dx, R = diag(noise_var), and with
   the push-through identity
       H^T S^-1 = (I_K + H^T R^-1 H P_xx)^-1 H^T R^-1,
   update mean += P_:x P_xx^-1 (x* - m_x) and cov -= P_:x H^T S^-1 H P_x:.

The plain EKF identifies B best in the moderate regime of scripts/joint_ekf_simulation.py.
Iterating pays off when stim drives the latents far from where the rate was predicted
(strong stim, weak loadings), where a single linearization overshoots and the filter
can diverge. Only K x K systems are solved; the full covariance is never inverted, which
matters for the joint state, whose B block has no process noise and can be close to
singular.

Time convention (matches simulations.simulate_trial): s[t] drives
the transition into x_t, then y_t is observed. The prior (mu0, Sigma0) is on x_{-1}.
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


NUM_ITERS = 1  # plain EKF; > 1 for the iterated EKF


def ekf_update(
    model: StateSpaceModel,
    mean: Array,
    cov: Array,
    y: Array,
    keep: Array,
    num_iters: int = NUM_ITERS,
) -> tuple[Array, Array, Array]:
    """One EKF measurement update, iterated `num_iters` times. Returns (loglik, mean, cov).

    keep == False means the bin was blanked: the update is the identity and the bin
    adds nothing to the log likelihood. The log likelihood is the plug-in
    likelihood at the predicted mean, as in the EKF marginal likelihood approximation.
    """
    K = model.state_dim
    m_x, P_xx = mean[:K], cov[:K, :K]
    y = jnp.where(keep, y, 0.0)  # blanked bins hold NaN

    def linearize(x):
        # Jacobian of the rate at x gives H
        rate = model.rate(x)
        H = jax.jacfwd(model.rate)(x)
        return rate, H, H.T / model.noise_var(rate)

    def gauss_newton_step(_, x):
        rate, H, HtRinv = linearize(x)
        return x + jnp.linalg.solve(
            jnp.eye(K) + P_xx @ HtRinv @ H, P_xx @ (HtRinv @ (y - rate)) - (x - m_x)
        )

    # the covariance uses the last linearization point, so num_iters=1 is the plain EKF
    x_lin = lax.fori_loop(0, num_iters - 1, gauss_newton_step, m_x)
    x_post = gauss_newton_step(None, x_lin)

    _, H, HtRinv = linearize(x_lin)
    HtSinv = jnp.linalg.solve(jnp.eye(K) + HtRinv @ H @ P_xx, HtRinv)
    cross_cov = cov[:, :K]
    mean_upd = mean + cross_cov @ jnp.linalg.solve(P_xx, x_post - m_x)
    cov_upd = symmetrize(cov - cross_cov @ (HtSinv @ H) @ cov[:K, :])
    loglik = model.log_prob(y, model.rate(m_x))

    return (
        jnp.where(keep, loglik, 0.0),
        jnp.where(keep, mean_upd, mean),
        jnp.where(keep, cov_upd, cov),
    )


def predict(
    model: StateSpaceModel, mean: Array, cov: Array, B: Array, s: Array
) -> tuple[Array, Array]:
    # Predictive mean and covariance
    F = model.jacobian(mean)
    return model.transition(mean, B, s), symmetrize(F @ cov @ F.T + model.Q)


@eqx.filter_jit
def filter_latents(
    model: StateSpaceModel,
    B: Array,
    ys: Array,
    ss: Array,
    keep: Array,
    ekf_iters: int = NUM_ITERS,
) -> tuple[Array, Array, Array, Array, Array]:
    """Latent-only filter with B held fixed.

    Returns (loglik, filtered_means, filtered_covs, predicted_means, predicted_covs).
    """

    def step(carry, inputs):
        loglik, mean, cov = carry
        y, s, k = inputs
        pred_mean, pred_cov = predict(model, mean, cov, B, s)
        ll, mean, cov = ekf_update(model, pred_mean, pred_cov, y, k, ekf_iters)
        return (loglik + ll, mean, cov), (mean, cov, pred_mean, pred_cov)

    (loglik, _, _), (fm, fP, pm, pP) = lax.scan(
        step, (jnp.array(0.0), model.mu0, model.Sigma0), (ys, ss, keep)
    )
    return loglik, fm, fP, pm, pP


@eqx.filter_jit
def smooth_latents(model: StateSpaceModel, B: Array, ys: Array, ss: Array, keep: Array) -> Posterior:
    """Latent-only filter plus RTS backward pass, with B held fixed."""
    loglik, fm, fP, pm, pP = filter_latents(model, B, ys, ss, keep)

    def step(carry, t):
        sm_next, sP_next = carry
        G = jnp.linalg.solve(pP[t + 1], model.jacobian(fm[t]) @ fP[t]).T
        sm = fm[t] + G @ (sm_next - pm[t + 1])
        sP = symmetrize(fP[t] + G @ (sP_next - pP[t + 1]) @ G.T)
        return (sm, sP), (sm, sP)

    T = ys.shape[0]
    _, (sm, sP) = lax.scan(step, (fm[-1], fP[-1]), jnp.arange(T - 1), reverse=True)
    sm = jnp.concatenate([sm, fm[-1:]])
    sP = jnp.concatenate([sP, fP[-1:]])

    return Posterior(loglik, fm, fP, sm, sP)


@eqx.filter_jit
def batch_smooth_latents(
    model: StateSpaceModel, B: Array, Ys: Array, Ss: Array, keep: Array
) -> Posterior:
    """vmap of `smooth_latents` over trials. keep is (T,) shared or (n_trials, T)."""
    keep = jnp.broadcast_to(keep, Ys.shape[:2])
    return jax.vmap(smooth_latents, in_axes=(None, None, 0, 0, 0))(model, B, Ys, Ss, keep)

"""Blanking-aware EKF filter and RTS smoother for the Poisson LDS.

Two departures from DYNAMAX's conditional_moments_gaussian_filter:

1. A per-timestep observation mask. keep[t] == False means the emission at t was
   discarded (stim artifact blanking), so the update is the identity and the bin
   contributes nothing to the marginal log likelihood.

2. The gain is computed in information form via Woodbury. R is diagonal and
   H P H^T has rank state_dim, so the (n_ch, n_ch) Cholesky in DYNAMAX's
   psd_solve is replaced by two (state_dim, state_dim) inverses.

Input convention matches DYNAMAX: inputs[t] drives the transition t -> t+1.
"""

import jax
import jax.numpy as jnp
from jax import jacfwd, vmap
from tensorflow_probability.substrates.jax.distributions import Poisson

def _condition_on_masked(x, P, h, r_diag, u, y, keep):
    """One EKF update. Returns (loglik, filtered_mean, filtered_cov)."""
    y = jnp.where(keep, y, 0.0) # Discard nan 
    yhat = h(x, u)                      
    H = jacfwd(lambda z: h(z, u))(x)    # Get Jacobian
    r = r_diag(x, u)                    # Process noise
    Hr = H / r[:, None]
    J = jnp.linalg.inv(P) + H.T @ Hr
    P_upd = jnp.linalg.inv(J)
    x_upd = x + P_upd @ Hr.T @ (y - yhat)
    ll = Poisson(rate=yhat).log_prob(y).sum()
    return (jnp.where(keep, ll, 0.0), # Update LL at masked 
            jnp.where(keep, x_upd, x),
            jnp.where(keep, P_upd, P))


def filter_masked(h, r_diag, f, Q, x0, P0, y, u, keep):
    """Forward pass. Returns (loglik, filtered_means, filtered_covs, pred_means, pred_covs)."""
    def step(carry, t):
        ll, x_pred, P_pred = carry
        ll_t, x_filt, P_filt = _condition_on_masked(x_pred, P_pred, h, r_diag, u[t], y[t], keep[t])
        F = jacfwd(lambda z: f(z, u[t]))(x_filt)
        x_next = f(x_filt, u[t])
        P_next = F @ P_filt @ F.T + Q
        return (ll + ll_t, x_next, P_next), (x_filt, P_filt, x_next, P_next)

    (ll, _, _), (xf, Pf, mp, Pp) = jax.lax.scan(
        step, (0.0, x0, P0), jnp.arange(y.shape[0]))
    return ll, xf, Pf, mp, Pp


def smooth_masked(h, r_diag, f, Q, x0, P0, y, u, keep):
    """Forward filter plus RTS backward pass. Returns (loglik, filtered, smoothed) means/covs."""
    ll, xf, Pf, xp, Pp = filter_masked(h, r_diag, f, Q, x0, P0, y, u, keep)

    def step(carry, t):
        xs_next, Ps_next = carry
        F = jacfwd(lambda z: f(z, u[t]))(xf[t])
        G = jnp.linalg.solve(Pp[t], F @ Pf[t]).T
        xs = xf[t] + G @ (xs_next - xp[t])
        Ps = Pf[t] + G @ (Ps_next - Pp[t]) @ G.T
        return (xs, Ps), (xs, Ps)

    T = y.shape[0]
    _, (xs, Ps) = jax.lax.scan(
        step, (xf[-1], Pf[-1]), jnp.arange(T - 1), reverse=True)
    xs = jnp.vstack([xs, xf[-1][None]])
    Ps = jnp.vstack([Ps, Pf[-1][None]])
    return ll, xf, Pf, xs, Ps


def batch_smooth(h, r_diag, f, Q, m0, P0, Y, U, keep):
    """vmap over trials. keep is (T,) shared or (n_trials, T)."""
    keep_broadcasted = jnp.broadcast_to(keep, Y.shape[:2]) if keep.ndim == 1 else keep
    return vmap(lambda y, u, k: smooth_masked(h, r_diag, f, Q, m0, P0, y, u, k))(Y, U, keep_broadcasted)
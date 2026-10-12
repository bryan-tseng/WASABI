from collections import namedtuple

import numpy as np
import jax.numpy as jnp
from jax import vmap

from .masked_plds import batch_smooth, filter_masked
from .utils import softplus_jnp


Posterior = namedtuple(
    "Posterior",
    "marginal_loglik filtered_means filtered_covariances smoothed_means smoothed_covariances",
)


def model_fns(C, d, A, B, b, V, B_SCALE, BETA_STIM, BETA_EMIT):
    """Bind parameters into the (z, u) closures the masked filter needs.

    Args:
        C: (n_ch, state_dim) loading matrix.
        d: (n_ch,) per-channel baseline, pre-softplus.
        A: (state_dim, state_dim) dynamics operator.
        B: (state_dim, n_ch) stim-to-latent map.
        b: scalar recruitment threshold.
        V: (n_ch, n_ch) spatial spread matrix.
        B_SCALE: scale factor applied to B.
        BETA_STIM: softplus sharpness for the stim path.
        BETA_EMIT: softplus sharpness for the observation link.

    Returns:
        h: observation mean, softplus(Cz + d).
        r: observation variance as a diagonal vector, not a matrix.
        f: dynamics, Az + B_SCALE * B @ [softplus(Vu + b) - softplus(b)].

    Note:
        inputs[t] drives the transition t -> t+1 (dynamax convention).
    """
    f = lambda z, u: A @ z + B_SCALE * (B @ (softplus_jnp(V @ u + b, BETA_STIM) - softplus_jnp(b, BETA_STIM)))
    h = lambda z, u: softplus_jnp(C @ z + d, BETA_EMIT) + 1e-4 # jitter for zero rate mean
    r = lambda z, u: softplus_jnp(C @ z + d, BETA_EMIT) + 1e-3 # jitter for near zero variance
    return h, r, f


def loss_fn(theta, observations, inputs, keep, Q_dyn, V, B_SCALE, BETA_STIM, BETA_EMIT):
    """
    Compute the negative log-likelihood using EKF like integrals. This is the Conditional Marginal Gaussian Smoother (CMGS) objective adapted from DYNAMAX, which is an approximation to the true marginal log-likelihood.

    Args:
        theta: (C, d, A, B, b) model parameters.
        observations: (n_trials, T, n_ch) observations.
        inputs: (n_trials, T, n_ch) stim inputs, shifted for the dynamax convention.
        keep: (T,) or (n_trials, T) mask, False where the emission is blanked.
        Q_dyn: (state_dim, state_dim) process noise covariance.
        V, B_SCALE, BETA_STIM, BETA_EMIT: passed through to model_fns.

    Returns:
        Loss: Negative log-likelihood + penalty for stability
    """
    # Unpack
    C, d, A, B, b = theta
    state_dim = A.shape[0]
    
    # Penalty for A
    penalty_A = 1e2 * jnp.maximum(0.0, jnp.linalg.norm(A, ord=2) - 1.0) ** 2

    # Get callable functions
    h, r, f = model_fns(C, d, A, B, b, V, B_SCALE, BETA_STIM, BETA_EMIT)

    keep_broadcasted = jnp.broadcast_to(keep, observations.shape[:2]) # broadcast to match dimensions of observations

    # Get ll from mask-aware filter
    ll, *_ = vmap(lambda y, u, k: filter_masked(
        h, r, f, Q_dyn, jnp.zeros(state_dim), jnp.eye(state_dim), y, u, k
    ))(observations, inputs, keep_broadcasted)
    return -jnp.sum(ll) + penalty_A


def run_smoother(theta, observations, inputs, keep, Q_dyn, V, B_SCALE, BETA_STIM, BETA_EMIT):
    """Run the masked batch smoother for a parameter set, wrapped in a Posterior object.
    This gives you the smoothed latents for plotting.

    Args:
        theta (_type_): _description_
        observations (_type_): _description_
        inputs (_type_): _description_
        keep (_type_): _description_
        Q_dyn (_type_): _description_
        
        V: (n_ch, n_ch) spatial spread matrix.
        B_SCALE: scale factor applied to B.
        BETA_STIM: softplus sharpness for the stim path.
        BETA_EMIT: softplus sharpness for the observation link.
        
    Returns:
        Posterior (Posterior object): Posterior object that includes:
            marginal likelihood
            filtered_means
            filtered_covariances
            smoothed_means
            smoothed_covariances
    """
    h, r, f = model_fns(*theta, V, B_SCALE, BETA_STIM, BETA_EMIT)
    state_dim = theta[2].shape[0]
    return Posterior(*batch_smooth(h, r, f, Q_dyn, jnp.zeros(state_dim),
                                    jnp.eye(state_dim), observations, inputs, keep))


def alignment(C_target, C_source):
    """Change-of-basis matrices aligning C_source into C_target's basis.
    """
    C_target = np.asarray(C_target)
    C_source = np.asarray(C_source)
    rot_inv = np.linalg.pinv(C_source) @ C_target
    rot = np.linalg.pinv(rot_inv)
    return rot, rot_inv

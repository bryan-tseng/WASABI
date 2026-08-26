import numpy as np
import jax.numpy as jnp

def softplus_np(z, beta=5.0):
    return np.logaddexp(0.0, beta * z) / beta

def softplus_jnp(z, beta=5.0):
    return jnp.logaddexp(0.0, beta * z) / beta

def softplus_inv_np(y, beta=5.0):
    return y + np.log(-np.expm1(-beta * y)) / beta

def make_toeplitz_V(N, w):
    """
    Row-normalized Gaussian spatial spread matrix over N channels.
    Args:
        N (int): number of channels
        w (float): spatial spread radius
    Returns:
        V (np.ndarray): row-normalized Gaussian spatial spread matrix
    """
    idx = np.arange(N)
    V_unnorm = np.exp(-((idx[:, None] - idx[None, :]) ** 2) / (2 * w ** 2))
    return V_unnorm / V_unnorm.sum(axis=1, keepdims=True)

def effective_stim_input(s, V, b, beta_stim=5.0):
    """u = softplus(V s + b) - softplus(b), matching dynamics_function in the fit."""
    s = np.asarray(s, dtype=float)
    V = np.asarray(V)
    was_1d = s.ndim == 1
    s2 = np.atleast_2d(s)
    u = softplus_np(s2 @ V.T + b, beta_stim) - softplus_np(b, beta_stim)
    return u[0] if was_1d else u
import numpy as np


def softplus_np(z, beta=5.0):
    return np.logaddexp(0.0, beta * z) / beta


def softplus_inv_np(y, beta=5.0):
    return y + np.log(-np.expm1(-beta * y)) / beta


def make_toeplitz_V(N, w, identity=False):
    """
    Row-normalized Gaussian spatial spread matrix over N channels.
    Args:
        N (int): number of channels
        w (float): spatial spread radius
        identity (bool): whether to return an identity matrix
    Returns:
        V (np.ndarray): row-normalized Gaussian spatial spread matrix
    """
    if identity:
        return np.eye(N)
    idx = np.arange(N)
    V_unnorm = np.exp(-((idx[:, None] - idx[None, :]) ** 2) / (2 * w ** 2))
    return V_unnorm / V_unnorm.sum(axis=1, keepdims=True)


def effective_stim_input(s, V, b, beta_stim=5.0):
    """u = softplus(V s + b) - softplus(b), matching wasabi.utils.stim_nonlinearity applied to V s."""
    s = np.asarray(s, dtype=float)
    return softplus_np(s @ np.asarray(V).T + b, beta_stim) - softplus_np(b, beta_stim)


def stim_schedule(pattern, T, t_stim, n_stim_bins):
    """(T, N) stim matrix holding `pattern` on for bins t_stim .. t_stim + n_stim_bins - 1."""
    s = np.zeros((T, len(pattern)))
    s[t_stim:t_stim + n_stim_bins] = pattern
    return s

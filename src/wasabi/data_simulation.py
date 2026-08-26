import numpy as np
from scipy.linalg import expm

from .utils import make_toeplitz_V, softplus_np, effective_stim_input


def make_ground_truth_params(K=4, N=128, spectral_radius=0.9, sigma2_Q=0.2, sigma2_C=1.0, w=3.0, b_input=0.8, B_scale=5.0, seed=0):
    """
    Construct ground-truth PLDS parameters.

    Args:
        K (int): latent dimensionality
        N (int): number of channels
        spectral_radius (float): Eigenvalues of A radius, eig_A < 1 for stability
        sigma2_Q (float): process noise
        sigma2_C (float): loading matrix covariance
        w (float): spatial spread radius
        b_input (float): perturbation threshold
    Returns:
        params: dictionary of params
    """
    rng = np.random.default_rng(seed)

    # Skew symmetric S for rotational dynamics
    S_raw = rng.standard_normal((K, K))
    S = (S_raw - S_raw.T) / 2  # make skew-symmetric
    # In a skew-symmetric matrix its eigenvalues are purely imaginary (jθ), so rescale theta for to be <= pi/6
    eigvals_S = np.linalg.eigvals(S)
    max_theta = np.max(np.abs(eigvals_S.imag))
    max_angle = np.pi / 6
    if max_theta > 0:
        S = S * (max_angle / max_theta)
    A = spectral_radius * expm(S) # Diagonals of S is zero, treat as continuous dynamics operator and get discrete A
    Q = sigma2_Q * np.eye(K) # Latent process noise covariance (diagonal)

    # Stim params
    B = B_scale * rng.standard_normal((K, N)) # Stim-to-latent mapping.     / np.sqrt(N) 
    b = b_input # Perturbation threshold
    
    # Measurement params
    C = rng.standard_normal((N, K)) * np.sqrt(sigma2_C / K) # Loading matrix, scaled to have sigma2_C variance per row (per channel)
    d = rng.normal(loc=0.1, scale=0.5, size=N) # Per-channel mean log-firing rate set to about 10Hz

    # Initial state
    mu0 = np.zeros(K)
    Sigma0 = np.eye(K)

    # Toeplitz matrix
    V = make_toeplitz_V(N, w)

    return dict(A=A, B=B, b=b, C=C, d=d, Q=Q, mu0=mu0, Sigma0=Sigma0, V=V)


def simulate_trial(params, T, stim_channels, t_stim, n_stim_bins, rng, softplus_beta_stim=5.0, softplus_beta_emit=1.0):
    """Simulate one trial of the PLDS, stimulation happens at one time-step, then samples Poisson spike counts through a softplus link.

    Args:
        params (dict): Ground-truth parameters: A, B, b, C, d, Q, mu0, Sigma0, and V
        T (int): Number of timesteps
        stim_channels (np.ndarray): Channel indices for stim.
        t_stim (int): Timestep at which stimulation is delivered.
        n_stim_bins (int): Number of bins over which stimulation is delivered.
        rng (np.random.Generator):
        softplus_beta (float, optional): Sharpness of the softplus link; larger values approach a ReLU

    Returns:
        tuple: A tuple `(x, y, s)` where:
            - x (np.ndarray): latent trajectory, shape (T, K)
            - y (np.ndarray): spike counts, shape (T, N)
            - s (np.ndarray): Stim indicator over time (nonzero only at t_stim), shape (T, N)
    """
    # Unpack params
    A, B, b, C, d, Q, mu0, Sigma0, V = (
        params["A"], params["B"], params["b"], params["C"], params["d"],
        params["Q"], params["mu0"], params["Sigma0"], params["V"],
    )

    # Dimensions
    K = A.shape[0]
    N = C.shape[0]

    # Preallocate
    s = np.zeros((T, N))
    x = np.zeros((T, K))
    y = np.zeros((T, N), dtype=int)

    # Stim
    s[t_stim : t_stim + n_stim_bins, stim_channels] = 1.0

    x_prev = rng.multivariate_normal(mu0, Sigma0)
    L_Q = np.linalg.cholesky(Q) # If Q is diagonal, its L_Q

    for t in range(T):
        # Check when stim happens
        stim_input = B @ effective_stim_input(s[t], V, b, softplus_beta_stim)

        eps = L_Q @ rng.standard_normal(K) # Sample noise using Cholesky factor
        x_t = A @ x_prev + stim_input + eps # state equation

        # Measurement
        rate = softplus_np(C @ x_t + d, softplus_beta_emit)
        y_t = rng.poisson(rate) # TODO can use Ogada sampling process?

        # Save results and index
        x[t] = x_t
        y[t] = y_t
        x_prev = x_t
    return x, y, s


def blank_stim_artifact(Y, t_stim, n_blank):
    """Artifact blanking: what the recording pipeline actually hands us."""
    Y_obs = np.asarray(Y, dtype=float).copy()
    Y_obs[:, t_stim:t_stim + n_blank, :] = np.nan
    return Y_obs
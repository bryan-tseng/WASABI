import numpy as np
from scipy.linalg import expm

from .utils import make_toeplitz_V, softplus_np, effective_stim_input


def make_ground_truth_params(K=4, N=128, spectral_radius=0.9, sigma2_Q=0.2, sigma2_C=1.0, w=3.0, b_input=0.8, B_scale=5.0, seed=0, identity_V=True):
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
        identity_V (bool): use V = I instead of the Gaussian spread, so stim hits only
            the channels selected. Note this makes u a constant multiple of s, which
            flattens the softplus threshold b into a plain gain.
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
    V = make_toeplitz_V(N, w, identity=identity_V)

    return dict(A=A, B=B, b=b, C=C, d=d, Q=Q, mu0=mu0, Sigma0=Sigma0, V=V)


def make_structured_B(K, N, rng, n_sites=8, site_w=4.0, span=None, amp_sigma=0.8,
                      bg_frac=0.35, smooth_w=3.0, B_scale=1.0):
    """Spatially structured stim-to-latent map, in the shape a fitted B actually takes.

    An iid Gaussian B is white across channels, so almost all of its energy sits in
    the null space of the stim inputs (V smears any pattern to a low spatial
    frequency) and every channel is as informative as every other. A B fitted to
    real stim data instead looks like a handful of localized "responsive sites":
    smooth blobs at shared channel positions across latents, with a few strong ones
    dominating the rest.

    This builds that: `n_sites` Gaussian bumps with heavy-tailed strengths, each
    living on a single randomly assigned latent (spread as evenly over the K latents
    as n_sites allows), plus a weak smooth background.

    Args:
        K (int): latent dimensionality
        N (int): number of channels
        rng (np.random.Generator)
        n_sites (int): number of localized responsive sites
        site_w (float): median site radius in channels; keep >= the spread width w,
            since anything sharper is unreachable through V anyway
        span (tuple, optional): (lo, hi) stimulable channel range. Outside it B is
            tapered to zero, mimicking an array whose stim sites cover part of the
            probe. None spans all N channels.
        amp_sigma (float): lognormal spread of site strengths; larger gives a more
            dominant hotspot
        bg_frac (float): smooth background amplitude, as a fraction of the site std
        smooth_w (float): spatial scale of the background and of the span taper
        B_scale (float): target std of B over the stimulable span

    Returns:
        B (np.ndarray): (K, N) stim-to-latent mapping
    """
    idx = np.arange(N)
    lo, hi = (0, N) if span is None else span # Just in case we limit the number of stim sites we can stim at

    centers = rng.uniform(lo, hi, n_sites) # Place a few localized bumps
    widths = site_w * np.exp(rng.normal(0.0, 0.35, n_sites)) # Randomize the widths of the bumps
    Phi = np.exp(-0.5 * ((idx[None, :] - centers[:, None]) / widths[:, None]) ** 2) # Make Gaussian bumps around the centers
    site_gain = np.exp(rng.normal(0.0, amp_sigma, n_sites)) # Lognormal strengths, so a few sites dominate the rest

    # Assign each site to one latent, dealing latents out in random order so the
    # sites spread over all K before any latent gets a second one
    reps = int(np.ceil(n_sites / K))
    site_latent = np.concatenate([rng.permutation(K) for _ in range(reps)])[:n_sites]
    W = np.zeros((K, n_sites))
    W[site_latent, np.arange(n_sites)] = rng.standard_normal(n_sites) * site_gain # Random signs and heavy-tailed strengths
    B = W @ Phi # apply the bumps to the latent strengths

    # Smooth background at the spread scale
    bg = rng.standard_normal((K, N)) @ make_toeplitz_V(N, smooth_w).T # Make a smooth background pattern
    B = B + bg_frac * (B.std() / bg.std()) * bg # Add smooth background with the bumps

    # Taper to zero outside the stimulable span, so unstimulable channels carry no drive
    left = np.exp(-0.5 * ((idx - lo) / smooth_w) ** 2)
    right = np.exp(-0.5 * ((idx - (hi - 1)) / smooth_w) ** 2)
    B = B * np.where(idx < lo, left, np.where(idx > hi - 1, right, 1.0))

    return B * (B_scale / B[:, lo:hi].std()) # Set the target std for each latent over the stimulation span


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
        stim_input = B @ effective_stim_input(s[t], V, b, softplus_beta_stim, identity=True) # Get stim input

        eps = L_Q @ rng.standard_normal(K) # Sample noise using Cholesky factor
        x_t = A @ x_prev + stim_input + eps # state equation

        rate = softplus_np(C @ x_t + d, softplus_beta_emit) # measurement equation
        y_t = rng.poisson(rate) # TODO can use Ogada sampling process?

        # Save results and index
        x[t] = x_t
        y[t] = y_t
        x_prev = x_t
    return x, y, s


def make_timevec(t_start_ms=-800.0, t_stop_ms=600.0, bin_ms=20.0, prestim_ms=0.0, poststim_ms=100.0):
    """Make dictionary that puts together time vectors, index, bin_size, stim timings etc.

    Args:
        t_start_ms (float): left edge of the first bin, relative to stim onset
        t_stop_ms (float): right edge of the last bin
        bin_ms (float): bin width
        prestim_ms (float): stim onset, and when the pre-stim state is read
        poststim_ms (float): when the response is read

    Returns:
        dict: prestim_ms, poststim_ms, T, bin_ms, t_ms, t_stim, n_stim_bins,
            horizon, and the indices i_prestim / i_poststim those times map to
    """
    if poststim_ms <= prestim_ms:
        raise ValueError(f"poststim_ms ({poststim_ms}) must be after prestim_ms ({prestim_ms})")

    T = int(round((t_stop_ms - t_start_ms) / bin_ms))
    t_ms = t_start_ms + bin_ms * np.arange(T)

    # prestim is the left edge of stim and poststim is the right edge of stim. so the indices are from i_prestim to i_poststim below
    i_prestim = int(round((prestim_ms - t_start_ms) / bin_ms))
    i_poststim = int(round((poststim_ms - t_start_ms) / bin_ms)) - 1
    horizon = i_poststim - i_prestim

    t_stim = i_prestim + 1              # first stimulated bin starts at prestim_ms
    n_stim_bins = horizon               # stim held across the readout window

    if i_prestim < 0:
        raise ValueError(f"prestim_ms={prestim_ms} needs at least one bin before it")
    if i_poststim >= T:
        raise ValueError(f"poststim_ms={poststim_ms} is past the end of the trial")

    return dict(prestim_ms=prestim_ms, poststim_ms=poststim_ms,
                T=T, bin_ms=bin_ms, t_ms=t_ms, t_stim=t_stim,
                n_stim_bins=n_stim_bins, horizon=horizon,
                i_prestim=i_prestim, i_poststim=i_poststim)


def simulate_perturbation(params, time_vec, stim_channels, rng,
                          softplus_beta_stim=5.0, softplus_beta_emit=1.0):
    """Run one full-dynamics trial and read the perturbation off the latents.

    Unlike the analytic z = M B u + eps shortcut, this drives the actual PLDS:
    the state rolls forward from mu0 through the whole trial with process noise,
    and spikes are emitted throughout, so the pre-stim state is whatever the
    dynamics happened to leave there.

    Args:
        params (dict): ground-truth PLDS parameters
        time_vec (dict): from make_trial_timebase
        stim_channels (np.ndarray): channel indices to stimulate
        rng (np.random.Generator)

    Returns:
        tuple: (z, x, y, s) where z is the perturbation between the trial's
            prestim and poststim times, x(poststim) - A^H x(prestim), and
            x, y, s are the full trial.
    """
    x, y, s = simulate_trial(
        params, time_vec["T"], stim_channels, time_vec["t_stim"], time_vec["n_stim_bins"], rng,
        softplus_beta_stim=softplus_beta_stim, softplus_beta_emit=softplus_beta_emit,
    )
    A_H = np.linalg.matrix_power(params["A"], time_vec["horizon"])
    z = x[time_vec["i_poststim"]] - A_H @ x[time_vec["i_prestim"]]  # free evolution of the pre-stim state removed
    return z, x, y, s


def blank_stim_artifact(Y, t_stim, n_blank):
    """Artifact blanking: what the recording pipeline actually hands us."""
    Y_obs = np.asarray(Y, dtype=float).copy()
    Y_obs[:, t_stim:t_stim + n_blank, :] = np.nan
    return Y_obs
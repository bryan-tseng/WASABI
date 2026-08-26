"""Recursive least squares tracking of the posterior over B.

Each row w_k = B[k, :] carries its own N x N covariance. "Cross latent"
covariance is dropped, but the full K x K noise covariance R_i is used in
the update.
"""

import numpy as np

from .utils import effective_stim_input


def get_perturbation_agg(smoothed_means, smoothed_covs, A, Q, t0, t1):
    """Extract perturbation term (z_i) and its covariance (R_i) at the stim time bin from smoothed posteriors.

    Args:
        smoothed_means: (n_trials, T, K)
        smoothed_covs: (n_trials, T, K, K)
        A, Q: (K, K) fitted dynamics and process noise
        t_stim: index of the stim time bin
        cross_covs: optional (n_trials, K, K) lag-one smoothed Cov(x_t, x_{t-1} | y).
            If given, R_i is corrected; if None, R_i is the conservative version.
    Returns:
        z: (n_trials, K)
        R: (n_trials, K, K) Noise introduced in that one step
    """
    x, P = np.asarray(smoothed_means), np.asarray(smoothed_covs)
    A = np.asarray(A); k = t1 - t0
    Ak = np.linalg.matrix_power(A, k)
    z = x[:, t1] - x[:, t0] @ Ak.T
    Qacc = sum(np.linalg.matrix_power(A, j) @ Q @ np.linalg.matrix_power(A, j).T
               for j in range(k))
    R = P[:, t1] + Ak @ P[:, t0] @ Ak.T + Qacc
    return z, 0.5 * (R + np.swapaxes(R, 1, 2))


class RowRLS:
    """Row-wise Gaussian posterior over B, each Row has its own N x N covariance."""

    def __init__(self, K, N, prior_var=25.0, mean_init=None, full_noise_cov=True, jitter=1e-10):
        """Initializations

        Args:
            K (int): Number of latents
            N (int): Number of stim channels
            prior_var (float, optional): Prior variance of B. Defaults to 25 times identity.
            mean_init (_type_, optional): Initial mean. Defaults to None.
            full_noise_cov (bool, optional): Whether to use full noise covariance. Defaults to True.
            jitter (_type_, optional): Jitter for numerical stability. Defaults to 1e-10.
        """
        self.K, self.N = K, N
        # Initialization of the rows of B (W) and their covariances (Sigma)
        self.W = np.zeros((K, N)) if mean_init is None else np.array(mean_init, dtype=float)
        self.Sigma = np.stack([prior_var * np.eye(N) for _ in range(K)]) # Initialize for K rows, this is P_{n|n-1}

        self.full_noise_cov = full_noise_cov
        self.jitter = jitter
        self.history = [] # Keep track per update

    def _predictive_cov(self, u, R):
        """Prior predictive covariance of z given input u and noise covariance R.

        Args:
            u: (N,) effective stim input.
            R: (K, K) noise covariance for this trial.

        Returns:
            Su: (K, N) prior covariance of each row projected onto u (Sigma^(k) u).
            sig: (K,) prior variance of the scalar prediction w_k^T u (u^T Sigma^(k) u).
            S: (K, K) prior predictive covariance of z: parameter uncertainty plus noise.
        """
        Su = self.Sigma @ u         # (K, N), row k is Sigma^(k) u.  or P_{n|n-1} projected onto u
        sig = Su @ u                # (K,), prior variance of w_k^T u
        # full_noise_cov=False drops cross-latent noise correlations
        Reff = np.asarray(R) if self.full_noise_cov else np.diag(np.diag(np.asarray(R)))
        S = np.diag(sig) + Reff
        return Su, sig, S

    def predict(self, u, R):
        """Predicted mean of z and its prior predictive covariance."""
        _, _, S = self._predictive_cov(u, R)
        return self.W @ u, S

    def info_gain(self, u, R):
        """Expected reduction in sum_k logdet Sigma^(k) from observing the next u.

        The covariance update is a rank-1 update per row, so by the matrix
        determinant lemma:

        det(Sigma_new^(k)) = det(Sigma^(k)) * (1 - c_k * sig_k)

        Info gain = log det(Sigma^(k)) - log det(Sigma_new^(k)) = -log(1 - c_k * sig_k)

        Therefore,
        Summing -log(1 - c_k * sig_k) over rows (K) gives the log-volume shrinkage.
        Depends only on u and R, so candidates can be scored before delivering stim.

        Args:
            u: (N,) effective stim input.
            R: (K, K) noise covariance.

        Returns:
            Expected log-det reduction in nats, >= 0.
        """
        _, sig, S = self._predictive_cov(u, R) # Predictive cov at this direction

        S_inv = np.linalg.inv(S + self.jitter * np.eye(self.K))
        c = np.diag(S_inv)  # (K,), c_k is the k-th diagonal of S^-1

        # Fraction of row k's variance along u that this observation removes
        frac = np.clip(c * sig, 0.0, 1.0 - 1e-12)
        return float(-np.sum(np.log1p(-frac)))

    def update(self, u, z, R):
        """RLS update of posterior given new observation

        Args:
            u (_type_): input vector (N,)
            z (_type_): observed perturbation (K,)
            R (_type_): noise covariance

        Returns:
            _type_: _description_
        """
        Su, sig, S = self._predictive_cov(u, R)
        S_inv = np.linalg.inv(S + self.jitter * np.eye(self.K))

        nu = np.asarray(z) - self.W @ u # Innovation

        # Mean update uses the full k-th row of S^-1; covariance uses only [S^-1]_kk
        self.W = self.W + Su * (S_inv @ nu)[:, None]
        c = np.clip(np.diag(S_inv), 0.0, None)
        self.Sigma = self.Sigma - c[:, None, None] * (Su[:, :, None] * Su[:, None, :])
        self.Sigma = 0.5 * (self.Sigma + np.swapaxes(self.Sigma, 1, 2))

        nis = float(nu @ S_inv @ nu) # Innovation squared scaled by covariance
        self.history.append(dict(nis=nis, sig=sig.copy(), logdet_S=float(np.linalg.slogdet(S)[1]))) # Save logdet of S for diagnostics
        return nis

    def logdet(self):
        return float(sum(np.linalg.slogdet(Sk)[1] for Sk in self.Sigma)) # summing the log determinants across each K

    def trace(self):
        return float(np.trace(self.Sigma, axis1=1, axis2=2).sum()) # gives you the sum of the traces of each row's covariance matrix


# Adaptive selection
def select_stim_from_set(RLS_tracker, candidate_s, V, b, R_hat, beta_stim=5.0):
    """Pick the pattern from a fixed candidate set with the largest expected info gain.

    Args:
        RLS_tracker:
        candidate_s (numpy.ndarray): Matrix of candidate stim patterns (M patterns, N channels)
        V (numpy.ndarray): toeplitz matrix mapping stim channels to effective input
        b (float): stim threshold
        R_hat (numpy.ndarray): Noise covariance of perturbation z
        beta_stim (float, optional): softplus param. Defaults to 5.0.

    Returns:
        j (int): Index of the selected stimulus pattern
        s_selected (numpy.ndarray): The selected stimulus pattern
        gains (numpy.ndarray): Info gains for all candidate patterns
    """
    U = effective_stim_input(np.asarray(candidate_s), V, b, beta_stim)
    gains = np.array([RLS_tracker.info_gain(u, R_hat) for u in U])
    best_ind = int(np.argmax(gains))
    chosen = candidate_s[best_ind]
    return best_ind, chosen, gains


def select_stim_greedy(RLS_tracker, V, b, R_hat, n_channels, beta_stim=5.0, candidates=None):
    """Greedily add channels one at a time to maximize expected info gain.

    Each channel is chosen conditional on those already selected, so redundant
    neighbours (which V smears into nearly identical inputs) are avoided.

    Args:
        RLS_tracker: RowRLS instance supplying the current posterior.
        V: (N, N) spatial spread matrix.
        b: threshold in the softplus stim nonlinearity.
        R_hat: (K, K) assumed noise covariance for scoring.
        n_channels: how many channels to switch on.
        beta_stim: softplus sharpness for the stim path.
        pool: optional subset of channel indices to choose from.
    Returns:
        ind: (N,) binary stim indicator.
        chosen: channel indices in the order they were added.
        total_gain: expected info gain of the full pattern, in nats.
    """
    N = RLS_tracker.N
    candidates = list(range(N)) if candidates is None else list(candidates)
    ind = np.zeros(N)
    chosen = []
    for _ in range(n_channels):
        best_cand, best_gain = None, -np.inf
        for cand in candidates:
            if cand in set(chosen):
                continue
            curr_stim = ind.copy()
            curr_stim[cand] = 1.0
            curr_gain = RLS_tracker.info_gain(effective_stim_input(curr_stim, V, b, beta_stim), R_hat)
            if curr_gain > best_gain:
                best_gain, best_cand = curr_gain, cand
        chosen.append(best_cand)
        ind[best_cand] = 1.0

    total_gain = RLS_tracker.info_gain(effective_stim_input(ind, V, b, beta_stim), R_hat)
    return ind, chosen, total_gain


# Random stim selection
def run_sequential(RLS_tracker, U, Z, R):
    """Feed trials in order, returning diagnostics per trial.
    This is for the random stim selection condition

    Args:
        RLS_tracker:
        U: (n_trials, N) effective stim inputs
        Z: (n_trials, K) observed perturbations
        R: (n_trials, K, K) noise covariances
    Returns:
        nis: (n_trials,) innovation squared per trial
        logdets: (n_trials,) log determinant of the predictive covariance per trial
    """
    nis, logdets = [], []
    for i in range(len(U)):
        nis.append(RLS_tracker.update(U[i], Z[i], R[i]))
        logdets.append(RLS_tracker.logdet())
    return np.array(nis), np.array(logdets)


# Restricting the error to the row space actually probed by U
def projected_error(W_est, W_true, U):
    """Relative error of B restricted to the row space actually probed by U."""
    Pu = np.linalg.pinv(U) @ U  # projector onto span of stim inputs
    err = (W_est - W_true) @ Pu
    return float(np.linalg.norm(err) / np.linalg.norm(W_true @ Pu))


def make_multistep_M(A, horizon=None):
    """
    M_H = I + A + ... + A^(Tau-1).
    horizon(Tau) =None gives M_inf = (I-A)^(-1).
    
    """
    K = A.shape[0]

    if horizon is None:
        return np.linalg.solve(np.eye(K) - A, np.eye(K))

    M = np.zeros_like(A)
    A_power = np.eye(K)
    for _ in range(horizon):
        M += A_power
        A_power = A_power @ A
    return M


def truncated_projector(U, energy=0.99):
    """Projector onto the directions U actually probes, dropping numerically empty ones.

    Args:
        U (): _description_
        energy (float, optional): _description_. Defaults to 0.99.

    Returns:
        _type_: _description_
    """
    _, sv, Vt = np.linalg.svd(np.asarray(U), full_matrices=False)
    r = int(np.searchsorted(np.cumsum(sv ** 2) / np.sum(sv ** 2), energy) + 1)
    return Vt[:r].T @ Vt[:r], r


def run_selection(mode, n_trials, cand, U_cand, B_true, chol, R_fixed,
                   K, N, seed, Pu, M, V, b, BETA_STIM):
    
    rng_arm = np.random.default_rng(seed)
    tracker = RowRLS(K, N, prior_var=25.0)

    R_rls = np.linalg.solve(M, R_fixed)
    R_rls = np.linalg.solve(M, R_rls.T).T

    denom = np.linalg.norm(B_true @ Pu)
    errs, logdets, selected_j = [], [], []

    for _ in range(n_trials):
        if mode == "D_opt":
            j, _, _ = select_stim_from_set(
                tracker, cand, V, b, R_rls, BETA_STIM
            )
        elif mode == "A_opt":
            j, _, _ = select_stim_from_set(
                tracker, cand, V, b, R_rls, BETA_STIM
            )
        elif mode == "tile":
            j, _, _ = select_stim_from_set(
                tracker, cand, V, b, R_rls, BETA_STIM
            )
        else:
            j = int(rng_arm.integers(0, len(cand)))

        u = U_cand[j]

        z = M @ B_true @ u + chol @ rng_arm.standard_normal(K)
        z_rls = np.linalg.solve(M, z)

        tracker.update(u, z_rls, R_rls)

        errs.append(
            np.linalg.norm((tracker.W - B_true) @ Pu) / denom
        )
        logdets.append(tracker.logdet())
        selected_j.append(j)

    return np.array(errs), np.array(logdets), np.array(selected_j), tracker

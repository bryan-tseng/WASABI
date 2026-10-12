"""Recursive least squares tracking of the posterior over B.

Each row w_k = B[k, :] carries its own N x N covariance. "Cross latent"
covariance is dropped, but the full K x K noise covariance R_i is used in
the update.
"""

import numpy as np
from .utils import effective_stim_input, make_multistep_M

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
            mean_init (_type_, optional): Initial mean of w vectors. Defaults to None.
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
        
        # full_noise_cov = False drops cross-latent noise correlations (i.e. diagonalize)
        Reff = np.asarray(R) if self.full_noise_cov else np.diag(np.diag(np.asarray(R)))
        
        S = np.diag(sig) + Reff     # Total predictive cov of z
        return Su, sig, S

    def predict(self, u, R):
        """Predicted mean of z and its predictive covariance."""
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


def accumulated_process_noise(A, Q, horizon):
    """Cov of the perturbation z under the full dynamics: sum_i A^i Q A^i'.

    With z = x[i_post] - A^H x[i_pre], the stim term is deterministic and every
    remaining term is process noise injected during the window, propagated by
    however many steps are left. This is the R to hand the tracker when z comes
    from a simulated trial rather than the analytic shortcut.
    """
    K = A.shape[0]
    R = np.zeros((K, K))
    for i in range(horizon):
        Ai = np.linalg.matrix_power(A, i)
        R += Ai @ Q @ Ai.T
    return 0.5 * (R + R.T)


def truncated_projector(U, energy=0.99):
    """"Projector onto the directions U actually probes, + the rank to reach a given energy
        because U can only explore certain number of rank after smearing by the toeplitz kernel

    Args:
        U (candidates): candidates of control vector (stim vector through the toeplitz and softplus)
        energy (float, optional): energy cutoff to define the reachable directions. Defaults to 0.99.

    Returns:
        Pu (): projector to the spanned subpsace of the control vectors
        r (int): rank of the reachable directions of the control vectors to reach 99% energy
    """
    _, sv, V_trans = np.linalg.svd(np.asarray(U), full_matrices=False)
    r = int(np.searchsorted(np.cumsum(sv ** 2) / np.sum(sv ** 2), energy) + 1)
    return V_trans[:r].T @ V_trans[:r], r 

def make_tiling_schedule(n_cand, n_stim_ch):
    """Deterministic scan over contiguous blocks.
        off gives a list of offsets, j gives a list per offset
        offsets are determined via the size of the contiguous blocks
    """
    if n_stim_ch & (n_stim_ch - 1) == 0: # Edge case
        width = n_stim_ch.bit_length() - 1
        offsets = sorted(range(n_stim_ch),
                         key=lambda o: int(format(o, f"0{width}b")[::-1], 2))
    else:
        offsets = list(range(n_stim_ch))
    return np.array([j for off in offsets
                     for j in range(off, n_cand, n_stim_ch)], dtype=int)

# Adaptive selection
def select_stim_from_set(RLS_tracker, candidate_s, V, b, R_hat, beta_stim=5.0, U=None):
    """Pick the pattern with the largest expected info gain.

    Pass precomputed `U` to skip re-deriving the effective inputs each call.
    
    Args:
        RLS_tracker:
        candidate_s (numpy.ndarray): Matrix of candidate stim patterns (M patterns, N channels)
        V (numpy.ndarray): toeplitz matrix mapping stim channels to effective input
        b (float): stim threshold
        R_hat (numpy.ndarray): Noise covariance of perturbation z
        beta_stim (float, optional): softplus param. Defaults to 5.0.
        U (optional): computed effective candidates

    Returns:
        best_ind (int): Index of the selected stimulus pattern
        s_selected (numpy.ndarray): The selected stimulus pattern
        gains (numpy.ndarray): Info gains for all candidate patterns
    """
    if U is None:
        U = effective_stim_input(np.asarray(candidate_s), V, b, beta_stim)
    gains = np.array([RLS_tracker.info_gain(u, R_hat) for u in U])
    best_ind = int(np.argmax(gains))
    return best_ind, candidate_s[best_ind], gains

def predict_R(u, x_pre, A, B_hat, C, d, R_floor=None, jitter=1e-6, lam_max=1e3):
    """Predicted latent-space noise covariance for stimulating with u.

    Pushes the current state estimate through the dynamics and the candidate's
    stim effect, evaluates the predicted rates, and inverts the Fisher
    information C' diag(lam) C. Higher rates give a tighter readout.
    """
    K = len(x_pre)
    x_post = A @ x_pre + B_hat @ u
    lam = np.clip(np.exp(C @ x_post + d), 1e-8, lam_max)
    J = C.T @ (lam[:, None] * C)                    # diag(lam) @ C, no K x K temporary
    R = np.linalg.inv(J + jitter * np.eye(K))
    if R_floor is not None:
        R = R + R_floor                             # process noise the readout can't beat
    return 0.5 * (R + R.T)

def run_selection(mode, n_trials, s_cand, B_true, R_fixed, K, N, Pu, M, V, b,
                  BETA_STIM, n_stim_ch=None, seed=1, prior_var=25.0,
                  full_noise_cov=True, snap_at=(),
                  A=None, C=None, d=None, R_floor=None, Q_state=None,
                  trial_sim=None):
    """Run one simulated stim-selection experiment and track the posterior over B.
    
    Each trial picks a stim pattern from the candidate set according to `mode`,
    simulates the multi-step perturbation z = M * B * u + eps, maps it back to 
    M^-1 z, and updates the RowRLS tracker. Error is projected measured to the subspace 
    the candidate set can excite (via `Pu`), since the complement is in the null space of possible stims.

    In "D_opt_R" the noise model is rate-dependent and estimation is fully
    separated from generation (most faithful): the simulator draws noise using B_true and the
    true state x, while scoring and updating use tracker.W and the propagated
    estimate x_hat. A, C, d are shared (which you get from fitted PLDS).
    
    Args:
        mode (str): Selection rule. "D_opt, tile or random"
        n_trials (int): Number of stim trials to run.
        s_cand (np.ndarray): (n_cand, N) binary candidate stim patterns.
        B_true (np.ndarray): (K, N) ground-truth stim-to-latent mapping.
        R_fixed (np.ndarray): (K, K) latent noise covariance on the multi-step observation z.
        K (int): Number of latents.
        N (int): Number of stim channels.
        Pu (np.ndarray): (N, N) orthogonal projector onto the directions the
            candidate set actually probes.
        M (np.ndarray): (K, K) multi-step propagator I + A + ... + A^(H-1).
        V (np.ndarray): (N, N) Toeplitz spatial spread matrix, used to score
            candidates in "D_opt" mode.
        b (float): Threshold in the softplus stim nonlinearity.
        BETA_STIM (float): Softplus sharpness for the stim path.
        n_stim_ch (int, optional): Active channels per pattern, for the "tile" schedule.
        seed (int): Seed for the arm-selection and noise generator.
        trial_sim (callable, optional): f(s_row) -> z. When given, the perturbation
            comes from this instead of the analytic z = M B u + eps, so the loop
            runs on full simulated trials. Pass R_fixed = accumulated_process_noise(
            A, Q, horizon) so the tracker's noise model matches.

    Raises:
        ValueError: If `mode` is not one of "D_opt", "tile", or "random".

    Returns:
        errs (np.ndarray): (n_trials,) relative error of W against B_true on the
            probed subspace, after each update.
        logdets (np.ndarray): (n_trials,) sum over rows of log det Sigma^(k).
        selected_cand_ind (np.ndarray): (n_trials,) index into `s_cand` per trial.
        tracker (RowRLS): The tracker at its final state.
        snaps: maps each trial count in `snap_at` to a copy of tracker.W at that point.
    """
    if mode not in ("D_opt", "D_opt_R", "tile", "random"):
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "D_opt_R" and any(v is None for v in (A, C, d)):
        raise ValueError("D_opt_R needs A, C, d")
    if mode == "tile" and n_stim_ch is None:
        n_stim_ch = int(round(s_cand[0].sum()))

    rng_arm = np.random.default_rng(seed)
    tracker = RowRLS(K, N, prior_var=prior_var, full_noise_cov=full_noise_cov)

    U_cand = effective_stim_input(s_cand, V, b, BETA_STIM)
    M_inv = np.linalg.inv(M)
    R_multistep = M_inv @ R_fixed @ M_inv.T
    R_multistep = 0.5 * (R_multistep + R_multistep.T)
    chol_fixed = np.linalg.cholesky(R_fixed)
    schedule = make_tiling_schedule(len(s_cand), n_stim_ch) if mode == "tile" else None

    x_true = np.zeros(K)  # generative state, simulator only
    x_hat = np.zeros(K)   # propagated estimate, tracker only
    chol_Q = np.linalg.cholesky(Q_state) if Q_state is not None else None

    denom = np.linalg.norm(B_true @ Pu)
    errs, logdets, selected_cand_ind, snaps = [], [], [], {}

    for i in range(n_trials):
        if mode == "D_opt":
            cand_ind, _, _ = select_stim_from_set(
                tracker, s_cand, V, b, R_multistep, BETA_STIM, U=U_cand)
        elif mode == "D_opt_R":
            gains = np.array([
                tracker.info_gain(
                    u, M_inv @ predict_R(u, x_hat, A, tracker.W, C, d, R_floor) @ M_inv.T)
                for u in U_cand
            ])
            cand_ind = int(gains.argmax())
        elif mode == "tile":
            cand_ind = int(schedule[i % len(schedule)])
        else:
            cand_ind = int(rng_arm.integers(0, len(s_cand)))

        u = U_cand[cand_ind]

        if mode == "D_opt_R":
            # Generation: true mapping, true state, noise on the M-step observation
            R_true = predict_R(u, x_true, A, B_true, C, d, R_floor)
            z = M @ (B_true @ u) + np.linalg.cholesky(R_true) @ rng_arm.standard_normal(K)
            # Inference: estimated mapping and state, same M^-1 transform as the fixed path
            R_hat = predict_R(u, x_hat, A, tracker.W, C, d, R_floor)
            R_hat_ms = M_inv @ R_hat @ M_inv.T
            R_hat_ms = 0.5 * (R_hat_ms + R_hat_ms.T)
            tracker.update(u, M_inv @ z, R_hat_ms)
            x_true = A @ x_true + B_true @ u
            if chol_Q is not None:
                x_true = x_true + chol_Q @ rng_arm.standard_normal(K)
            x_hat = A @ x_hat + tracker.W @ u
        else:
            if trial_sim is not None:
                # Full dynamics: z comes from an actual simulated trial, so the
                # pre-stim state and noise are whatever the PLDS produced.
                z = trial_sim(s_cand[cand_ind])
            else:
                z = M @ (B_true @ u) + chol_fixed @ rng_arm.standard_normal(K)
            tracker.update(u, M_inv @ z, R_multistep)

        errs.append(np.linalg.norm((tracker.W - B_true) @ Pu) / denom)
        logdets.append(tracker.logdet())
        selected_cand_ind.append(cand_ind)
        if (i + 1) in snap_at:
            snaps[i + 1] = tracker.W.copy()

    nis = np.array([h["nis"] for h in tracker.history])
    return (np.array(errs), np.array(logdets),
            np.array(selected_cand_ind), tracker, snaps, nis)
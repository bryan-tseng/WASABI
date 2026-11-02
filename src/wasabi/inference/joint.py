"""Joint EKF over the augmented state [x; vec(B)].

Only the x rows of the augmented transition are non-trivial,

    F_x = [A, u_t^T kron I_K],
    F_B = [0, I_KN],
    Q~ = blockdiag(Q(s_t), 0),

so `predict` only updates the x rows and columns of the covariance; the B block is
unchanged because B is constant and has no process noise.

The measurement update is `ekf.ekf_update`, shared with the latent-only filter. With
`relinearize_iters > 0` the observation is re-linearized around the smoothed latent path and the
filter is run again (the damped iterated smoother of relinearize.py), so the B posterior is the
Laplace approximation at the trial's posterior mode instead of the plain EKF one.
"""

from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

from wasabi.inference.ekf import Posterior, ekf_update, smoothed_path, trajectory_log_density
from wasabi.inference.relinearize import RELINEARIZE_ITERS, damped_gauss_newton
from wasabi.model import StateSpaceModel
from wasabi.utils import symmetrize, unvec, vec


class BPosterior(eqx.Module):
    """Gaussian posterior over vec(B), carried from one trial to the next.
    mean is vec(B), cov is cov(vec(B)) is KN by KN
    num_latents is K used to reshape vec(B)
    """

    mean: Array
    cov: Array
    num_latents: int = eqx.field(static=True)

    @classmethod
    # Initialize from a prior with prior_var * I
    def from_prior(cls, num_latents: int, num_channels: int, prior_var: float) -> Self:
        dim = num_latents * num_channels
        return cls(jnp.zeros(dim), prior_var * jnp.eye(dim), num_latents)

    @property
    def B(self) -> Array:
        # For reshape when printing
        return unvec(self.mean, self.num_latents)

    def logdet(self) -> Array:
        # information criterion~~
        return jnp.linalg.slogdet(self.cov)[1]

    def realign(self, M: Array) -> Self:
        """Express the posterior in a new latent basis, x_new = M x_old, so B_new = M B_old.

        In vectorized form this is vec(B) -> (I_N kron M) vec(B).
        """
        K = self.num_latents
        mean = vec(M @ self.B)
        
        # reshaping cov to (N, K, N, K) to apply kron product and then back to (NK, NK), don't ask me about the einsum notation lol
        cov = self.cov.reshape(-1, K, self.cov.shape[0] // K, K)
        cov = jnp.einsum("ai, nimj, bj -> namb", M, cov, M).reshape(self.cov.shape)
        # return a eqx object with new mean and cov
        return eqx.tree_at(lambda p: (p.mean, p.cov), self, (mean, symmetrize(cov)))


def initial_state(model: StateSpaceModel, prior: BPosterior) -> tuple[Array, Array]:
    """Initialize augmented state.
    For every trial, x is reset to (mu0, Sigma0), uncorrelated with B.
    """
    mean = jnp.concatenate([model.mu0, prior.mean])
    cov = jax.scipy.linalg.block_diag(model.Sigma0, prior.cov)
    return mean, cov


def build_aug_operator(model: StateSpaceModel, x: Array, s: Array) -> Array:
    """x_t = A x_{t-1} + (uᵀ ⊗ I_K) vec(B) so this builds the top x rows
    """
    A = model.jacobian(x)
    u = model.stim_input(s) # represent stim input
    uTkronI = jnp.kron(u[None, :], jnp.eye(model.state_dim))
    return jnp.concatenate([A, uTkronI], axis=1)


def predict(
    model: StateSpaceModel, mean: Array, cov: Array, s: Array, Q: Array | None = None
) -> tuple[Array, Array]:
    # Predictive mean and covariance; Q defaults to the model's process noise for s
    K = model.state_dim
    Q = model.process_noise(s) if Q is None else Q
    A_U = build_aug_operator(model, mean[:K], s)
    x_pred = model.transition(mean[:K], unvec(mean[K:], K), s)
    FP = A_U @ cov  # (K, K + NK)
    cov = cov.at[:K, :].set(FP).at[:, :K].set(FP.T) # Write top K rows and K columns
    cov = cov.at[:K, :K].set(FP @ A_U.T + Q) # Include process noise in top left block
    return mean.at[:K].set(x_pred), symmetrize(cov)


def transition_matrix(model: StateSpaceModel, x: Array, s: Array) -> Array:
    """Dense F~ linearized at x, only used by the smoother."""
    K = model.state_dim
    D = K + model.state_dim * model.num_channels
    return jnp.eye(D).at[:K, :].set(build_aug_operator(model, x, s))


def _filter(model, prior, ys, ss, keep, Qs, lin_xs, store):
    """One joint filter pass, the observation linearized at `lin_xs` (None: the predicted
    means). Returns (loglik, final mean, final cov, per-bin outputs): the filtered latent
    means, or with `store` the full filtered and predicted states the smoother needs."""
    K = model.state_dim

    def step(carry, inputs):
        loglik, mean, cov = carry
        y, s, k, Q, lin = inputs
        pred_mean, pred_cov = predict(model, mean, cov, s, Q)
        ll, mean, cov = ekf_update(model, pred_mean, pred_cov, y, k, lin)
        out = (mean, cov, pred_mean, pred_cov) if store else mean[:K]
        return (loglik + ll, mean, cov), out

    (loglik, mean, cov), out = lax.scan(
        step, (jnp.array(0.0), *initial_state(model, prior)), (ys, ss, keep, Qs, lin_xs)
    )
    return loglik, mean, cov, out


def _rts(model, ss, filtered_means, filtered_covs, predicted_means, predicted_covs):
    """Dense RTS backward pass over the augmented state."""

    def backward(carry, t):
        smoothed_means_next, smoothed_covs_next = carry
        F = transition_matrix(model, filtered_means[t, : model.state_dim], ss[t + 1])
        G = jnp.linalg.solve(predicted_covs[t + 1], F @ filtered_covs[t]).T
        smoothed_means = filtered_means[t] + G @ (smoothed_means_next - predicted_means[t + 1])
        smoothed_covs = symmetrize(filtered_covs[t] + G @ (smoothed_covs_next - predicted_covs[t + 1]) @ G.T)
        return (smoothed_means, smoothed_covs), (smoothed_means, smoothed_covs)

    T = ss.shape[0]
    _, (smoothed_means, smoothed_covs) = lax.scan(
        backward, (filtered_means[-1], filtered_covs[-1]), jnp.arange(T - 1), reverse=True
    )
    return (
        jnp.concatenate([smoothed_means, filtered_means[-1:]]),
        jnp.concatenate([smoothed_covs, filtered_covs[-1:]]),
    )


def _linearization(model, prior, ys, ss, keep, Qs, relinearize_iters):
    """Where to linearize the observation: None for the plain EKF, else the damped
    Gauss-Newton refinement of the smoothed latent path. B is part of the trajectory the
    log posterior is evaluated on, with the trial's prior on it. No gradient flows through it.

    Once linearized, the model is linear-Gaussian, so E[x_t | y] = E[x_t | y, B = E[B | y]]:
    the smoothed latent path is the cheap latent-only smoother at the filtered B mean, with no
    need for the dense smoother over the augmented state.
    """
    if relinearize_iters == 0:
        return None
    K = model.state_dim
    prior_chol = jnp.linalg.cholesky(prior.cov)

    def smooth(trajectory):
        lin_xs = trajectory[0]
        _, mean, _, _ = _filter(model, prior, ys, ss, keep, Qs, lin_xs, store=False)
        b = mean[K:]  # B is constant, so its smoothed mean is the last filtered one
        return smoothed_path(model, unvec(b, K), ys, ss, keep, Qs, lin_xs), b

    def log_density(trajectory):
        xs, b = trajectory
        r = jax.scipy.linalg.solve_triangular(prior_chol, b - prior.mean, lower=True)
        return trajectory_log_density(model, xs, unvec(b, K), ys, ss, keep, Qs) - 0.5 * r @ r

    trajectory = damped_gauss_newton(smooth, log_density, smooth((None,)), relinearize_iters - 1)
    return lax.stop_gradient(trajectory[0])


@eqx.filter_jit
def filter_trial(
    model: StateSpaceModel,
    prior: BPosterior,
    ys: Array,
    ss: Array,
    keep: Array,
    relinearize_iters: int = RELINEARIZE_ITERS,
) -> tuple[Array, BPosterior, Array]:
    """Run the joint filter over one trial.

    Returns (loglik, the posterior over B at the end of the trial, filtered latent means),
    from the final pass. The B block of the final filtered state is also the smoothed one,
    so it is the next trial's prior.
    """
    K = model.state_dim
    Qs = jax.vmap(model.process_noise)(ss)
    lin_xs = _linearization(model, prior, ys, ss, keep, Qs, relinearize_iters)
    loglik, mean, cov, filtered_means = _filter(model, prior, ys, ss, keep, Qs, lin_xs, store=False)
    Bposterior = eqx.tree_at(
        lambda p: (p.mean, p.cov), prior, (mean[K:], symmetrize(cov[K:, K:]))
    )
    return loglik, Bposterior, filtered_means


@eqx.filter_jit
def filter_trials(
    model: StateSpaceModel,
    prior: BPosterior,
    Ys: Array,
    Ss: Array,
    keep: Array,
    relinearize_iters: int = RELINEARIZE_ITERS,
) -> tuple[Array, BPosterior]:
    """`filter_trial` over trials in order, carrying the B posterior from one to the next.

    With the dynamics fixed, this is the (Gaussian-approximate) posterior over B given all trials. keep is (T,) shared or (n_trials, T).
    
    Returns (total loglik, posterior)
    """
    keep = jnp.broadcast_to(keep, Ys.shape[:2])

    def step(carry, inputs):
        loglik, prior = carry
        ll, prior, _ = filter_trial(model, prior, *inputs, relinearize_iters=relinearize_iters)
        return (loglik + ll, prior), None

    (loglik, Bposterior), _ = lax.scan(step, (jnp.array(0.0), prior), (Ys, Ss, keep))
    return loglik, Bposterior


@eqx.filter_jit
def smooth_trial(
    model: StateSpaceModel,
    prior: BPosterior,
    ys: Array,
    ss: Array,
    keep: Array,
    relinearize_iters: int = RELINEARIZE_ITERS,
) -> Posterior:
    """Joint filter plus RTS backward pass over the augmented state.

    Stores (T, NK, NK) covariances, so use it for inspecting one trial, not inside the loop.
    """
    Qs = jax.vmap(model.process_noise)(ss)
    lin_xs = _linearization(model, prior, ys, ss, keep, Qs, relinearize_iters)
    loglik, _, _, states = _filter(model, prior, ys, ss, keep, Qs, lin_xs, store=True)
    smoothed_means, smoothed_covs = _rts(model, ss, *states)
    filtered_means, filtered_covs, _, _ = states
    return Posterior(loglik, filtered_means, filtered_covs, smoothed_means, smoothed_covs)

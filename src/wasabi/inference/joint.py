"""Joint EKF over the augmented state [x; vec(B)].

Only the x rows of the augmented transition are non-trivial,

    F_x = [A, u_t^T kron I_K],
    F_B = [0, I_KN],
    Q~ = blockdiag(Q(s_t), 0),

so `predict` only updates the x rows and columns of the covariance; the B block is
unchanged because B is constant and has no process noise.

The measurement update is `ekf.ekf_update`, shared with the latent-only filter.
"""

from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

from wasabi.inference.ekf import Posterior, ekf_update
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


def predict(model: StateSpaceModel, mean: Array, cov: Array, s: Array) -> tuple[Array, Array]:
    # Predictive mean and covariance
    K = model.state_dim
    A_U = build_aug_operator(model, mean[:K], s)
    x_pred = model.transition(mean[:K], unvec(mean[K:], K), s)
    FP = A_U @ cov  # (K, K + NK)
    cov = cov.at[:K, :].set(FP).at[:, :K].set(FP.T) # Write top K rows and K columns
    cov = cov.at[:K, :K].set(FP @ A_U.T + model.process_noise(s)) # Include process noise in top left block
    return mean.at[:K].set(x_pred), symmetrize(cov)


def transition_matrix(model: StateSpaceModel, x: Array, s: Array) -> Array:
    """Dense F~ linearized at x, only used by the smoother."""
    K = model.state_dim
    D = K + model.state_dim * model.num_channels
    return jnp.eye(D).at[:K, :].set(build_aug_operator(model, x, s))


@eqx.filter_jit
def filter_trial(
    model: StateSpaceModel,
    prior: BPosterior,
    ys: Array,
    ss: Array,
    keep: Array,
) -> tuple[Array, BPosterior, Array]:
    """Run the joint filter over one trial.

    Returns (loglik, the posterior over B at the end of the trial, filtered latent means).
    The B block of the final filtered state is also the smoothed one, so it is the
    next trial's prior.
    """
    K = model.state_dim

    def step(carry, inputs):
        loglik, mean, cov = carry
        y, s, k = inputs
        mean, cov = predict(model, mean, cov, s)
        ll, mean, cov = ekf_update(model, mean, cov, y, k)
        return (loglik + ll, mean, cov), mean[:K]

    (loglik, mean, cov), filtered_means = lax.scan(
        step, (jnp.array(0.0), *initial_state(model, prior)), (ys, ss, keep)
    )
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
) -> tuple[Array, BPosterior]:
    """`filter_trial` over trials in order, carrying the B posterior from one to the next.

    With the dynamics fixed, this is the (Gaussian-approximate) posterior over B given all trials. keep is (T,) shared or (n_trials, T).
    
    Returns (total loglik, posterior)
    """
    keep = jnp.broadcast_to(keep, Ys.shape[:2])

    def step(carry, inputs):
        loglik, prior = carry
        ll, prior, _ = filter_trial(model, prior, *inputs)
        return (loglik + ll, prior), None

    (loglik, Bposterior), _ = lax.scan(step, (jnp.array(0.0), prior), (Ys, Ss, keep))
    return loglik, Bposterior


@eqx.filter_jit
def smooth_trial(
    model: StateSpaceModel, prior: BPosterior, ys: Array, ss: Array, keep: Array
) -> Posterior:
    """Joint filter plus RTS backward pass over the augmented state.

    Stores (T, NK, NK) covariances, so use it for inspecting one trial, not inside the loop.
    """

    def step(carry, inputs):
        loglik, mean, cov = carry
        y, s, k = inputs
        pm, pP = predict(model, mean, cov, s)
        ll, mean, cov = ekf_update(model, pm, pP, y, k)
        return (loglik + ll, mean, cov), (mean, cov, pm, pP)

    (loglik, _, _), (filtered_means, filtered_covs, predicted_means, predicted_covs) = lax.scan(
        step, (jnp.array(0.0), *initial_state(model, prior)), (ys, ss, keep)
    )

    def backward(carry, t):
        smoothed_means_next, smoothed_covs_next = carry
        F = transition_matrix(model, filtered_means[t, : model.state_dim], ss[t + 1])
        G = jnp.linalg.solve(predicted_covs[t + 1], F @ filtered_covs[t]).T
        smoothed_means = filtered_means[t] + G @ (smoothed_means_next - predicted_means[t + 1])
        smoothed_covs = symmetrize(filtered_covs[t] + G @ (smoothed_covs_next - predicted_covs[t + 1]) @ G.T)
        return (smoothed_means, smoothed_covs), (smoothed_means, smoothed_covs)

    T = ys.shape[0]
    _, (smoothed_means, smoothed_covs) = lax.scan(backward, (filtered_means[-1], filtered_covs[-1]), jnp.arange(T - 1), reverse=True)

    return Posterior(
        loglik,
        filtered_means,
        filtered_covs,
        jnp.concatenate([smoothed_means, filtered_means[-1:]]),
        jnp.concatenate([smoothed_covs, filtered_covs[-1:]]),
    )

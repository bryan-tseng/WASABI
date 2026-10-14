"""Joint EKF over the augmented state [x; vec(B)].

Only the x rows of the augmented transition are non-trivial,

    F~_x = [df/dx, u_t^T kron I_K],     F~_B = [0, I_KN],     Q~ = blockdiag(Q, 0),

(df/dx = A for linear dynamics), so the prediction updates the x rows and columns of the covariance in O(D^2 K) without
forming F~. The measurement update is `ekf.ekf_update`, shared with the latent-only filter.
"""

from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

from wasabi.inference.ekf import NUM_ITERS, Posterior, ekf_update
from wasabi.model import StateSpaceModel
from wasabi.utils import symmetrize, unvec, vec


class BPosterior(eqx.Module):
    """Gaussian posterior over vec(B), carried from one trial to the next."""

    mean: Array
    cov: Array
    num_latents: int = eqx.field(static=True)

    @classmethod
    def from_prior(cls, num_latents: int, num_channels: int, prior_var: float) -> Self:
        dim = num_latents * num_channels
        return cls(jnp.zeros(dim), prior_var * jnp.eye(dim), num_latents)

    @property
    def B(self) -> Array:
        return unvec(self.mean, self.num_latents)

    def logdet(self) -> Array:
        return jnp.linalg.slogdet(self.cov)[1]

    def realign(self, M: Array) -> Self:
        """Express the posterior in a new latent basis, x_new = M x_old, so B_new = M B_old.

        In column-major vec form this is vec(B) -> (I_N kron M) vec(B).
        """
        K = self.num_latents
        mean = vec(M @ self.B)
        cov = self.cov.reshape(-1, K, self.cov.shape[0] // K, K)
        cov = jnp.einsum("ai, nimj, bj -> namb", M, cov, M).reshape(self.cov.shape)
        return eqx.tree_at(lambda p: (p.mean, p.cov), self, (mean, symmetrize(cov)))


def initial_state(model: StateSpaceModel, prior: BPosterior) -> tuple[Array, Array]:
    """Trial-start augmented state: x reset to (mu0, Sigma0), uncorrelated with B."""
    mean = jnp.concatenate([model.mu0, prior.mean])
    return mean, jax.scipy.linalg.block_diag(model.Sigma0, prior.cov)


def _x_rows(model: StateSpaceModel, x: Array, s: Array) -> Array:
    """F~_x = [df/dx, u^T kron I_K], the only non-identity rows of the augmented transition."""
    u = model.stim_input(s)
    return jnp.concatenate(
        [model.jacobian(x), jnp.kron(u[None, :], jnp.eye(model.state_dim))], axis=1
    )


def predict(model: StateSpaceModel, mean: Array, cov: Array, s: Array) -> tuple[Array, Array]:
    K = model.state_dim
    Fx = _x_rows(model, mean[:K], s)
    x_pred = model.transition(mean[:K], unvec(mean[K:], K), s)
    FP = Fx @ cov  # (K, D)
    cov = cov.at[:K, :].set(FP).at[:, :K].set(FP.T)
    cov = cov.at[:K, :K].set(FP @ Fx.T + model.Q)
    return mean.at[:K].set(x_pred), symmetrize(cov)


def transition_matrix(model: StateSpaceModel, x: Array, s: Array) -> Array:
    """Dense F~ linearized at x, only used by the smoother."""
    K = model.state_dim
    D = K + model.state_dim * model.num_channels
    return jnp.eye(D).at[:K, :].set(_x_rows(model, x, s))


@eqx.filter_jit
def filter_trial(
    model: StateSpaceModel,
    prior: BPosterior,
    ys: Array,
    ss: Array,
    keep: Array,
    ekf_iters: int = NUM_ITERS,
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
        ll, mean, cov = ekf_update(model, mean, cov, y, k, ekf_iters)
        return (loglik + ll, mean, cov), mean[:K]

    (loglik, mean, cov), xs = lax.scan(
        step, (jnp.array(0.0), *initial_state(model, prior)), (ys, ss, keep)
    )
    posterior = eqx.tree_at(
        lambda p: (p.mean, p.cov), prior, (mean[K:], symmetrize(cov[K:, K:]))
    )
    return loglik, posterior, xs


@eqx.filter_jit
def filter_trials(
    model: StateSpaceModel,
    prior: BPosterior,
    Ys: Array,
    Ss: Array,
    keep: Array,
    ekf_iters: int = NUM_ITERS,
) -> tuple[Array, BPosterior]:
    """`filter_trial` over trials in order, carrying the B posterior from one to the next.

    With the dynamics fixed, this is the (Gaussian-approximate) posterior over B given
    all trials. keep is (T,) shared or (n_trials, T). Returns (total loglik, posterior).
    """
    keep = jnp.broadcast_to(keep, Ys.shape[:2])

    def step(carry, inputs):
        loglik, prior = carry
        ll, prior, _ = filter_trial(model, prior, *inputs, ekf_iters)
        return (loglik + ll, prior), None

    (loglik, posterior), _ = lax.scan(step, (jnp.array(0.0), prior), (Ys, Ss, keep))
    return loglik, posterior


@eqx.filter_jit
def smooth_trial(
    model: StateSpaceModel, prior: BPosterior, ys: Array, ss: Array, keep: Array
) -> Posterior:
    """Joint filter plus RTS backward pass over the augmented state.

    Stores (T, D, D) covariances, so use it for inspecting one trial, not inside the loop.
    """

    def step(carry, inputs):
        loglik, mean, cov = carry
        y, s, k = inputs
        pm, pP = predict(model, mean, cov, s)
        ll, mean, cov = ekf_update(model, pm, pP, y, k)
        return (loglik + ll, mean, cov), (mean, cov, pm, pP)

    (loglik, _, _), (fm, fP, pm, pP) = lax.scan(
        step, (jnp.array(0.0), *initial_state(model, prior)), (ys, ss, keep)
    )

    def backward(carry, t):
        sm_next, sP_next = carry
        F = transition_matrix(model, fm[t, : model.state_dim], ss[t + 1])
        G = jnp.linalg.solve(pP[t + 1], F @ fP[t]).T
        sm = fm[t] + G @ (sm_next - pm[t + 1])
        sP = symmetrize(fP[t] + G @ (sP_next - pP[t + 1]) @ G.T)
        return (sm, sP), (sm, sP)

    T = ys.shape[0]
    _, (sm, sP) = lax.scan(backward, (fm[-1], fP[-1]), jnp.arange(T - 1), reverse=True)

    return Posterior(
        loglik,
        fm,
        fP,
        jnp.concatenate([sm, fm[-1:]]),
        jnp.concatenate([sP, fP[-1:]]),
    )

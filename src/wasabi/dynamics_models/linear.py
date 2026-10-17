from typing import Self, override

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from wasabi.dynamics_models.base import DynamicsModel


class LinearDynamics(DynamicsModel):
    """f(x) = A x, with the spectral norm of A clipped to `rho_max` by `project`.

    The stim kernel is a fixed matrix V.

    Trial-to-trial variability of the stim response: the kick is (B + eta) u with
    eta[:, n] ~ N(0, stim_var[n] I_K), which adds sum_n u_n^2 stim_var[n] I_K to the process
    noise of stim bins. It is isotropic in the latents, so it is unchanged by a change of
    latent basis. eta is drawn per bin here, an approximation when it is fixed per trial.
    """

    A: Array
    Q_chol: Array
    V: Array
    b: Array
    log_stim_var: Array  # (num input channels,) log of stim_var
    mu0: Array
    Sigma0: Array
    beta_stim: float = eqx.field(static=True, default=5.0)
    rho_max: float = eqx.field(static=True, default=0.999)

    @classmethod
    def initialize(
        cls,
        state_dim: int,
        V: Array,
        q_init: float = 0.1,
        stim_var_init: float = 1e-3,
        **init_kwargs,
    ) -> Self:
        """A = 0.9 I, Q = q_init I, b = 0, near-zero stim variability, standard normal prior on x_{-1}."""
        V = jnp.asarray(V)
        return cls(
            A=0.9 * jnp.eye(state_dim),
            Q_chol=jnp.sqrt(q_init) * jnp.eye(state_dim),
            V=V,
            b=jnp.array(0.0),
            log_stim_var=jnp.full(V.shape[0], jnp.log(stim_var_init)),
            mu0=jnp.zeros(state_dim),
            Sigma0=jnp.eye(state_dim),
            **init_kwargs,
        )

    @property
    @override
    def Q(self) -> Array:
        L = jnp.tril(self.Q_chol)
        return L @ L.T + 1e-6 * jnp.eye(self.state_dim)

    @property
    def stim_var(self) -> Array:
        return jnp.exp(self.log_stim_var)

    @override
    def process_noise(self, s: Array) -> Array:
        u = self.stim_input(s)
        return self.Q + jnp.sum(u**2 * self.stim_var) * jnp.eye(self.state_dim)

    @override
    def propagate(self, x: Array) -> Array:
        return self.A @ x

    @override
    def jacobian(self, x: Array) -> Array:
        return self.A

    @override
    def apply_stim_kernel(self, s: Array) -> Array:
        return self.V @ s

    @override
    def project(self) -> Self:
        # Clip spectral norm of A to rho_max to stablize dynamics
        U, sv, Vt = jnp.linalg.svd(self.A)
        A = U @ jnp.diag(jnp.minimum(sv, self.rho_max)) @ Vt
        return eqx.tree_at(lambda m: m.A, self, A)

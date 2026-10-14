from typing import Self, override

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from wasabi.dynamics_models.base import DynamicsModel


class LinearDynamics(DynamicsModel):
    """f(x) = A x, with the spectral norm of A clipped to `rho_max` by `project`.

    The stim kernel is a fixed matrix V.
    """

    A: Array
    Q_chol: Array
    V: Array
    b: Array
    mu0: Array
    Sigma0: Array
    beta_stim: float = eqx.field(static=True, default=5.0)
    rho_max: float = eqx.field(static=True, default=0.999)

    @classmethod
    def initialize(
        cls, state_dim: int, V: Array, q_init: float = 0.1, **init_kwargs
    ) -> Self:
        """A = 0.9 I, Q = q_init I, b = 0, standard normal prior on x_{-1}."""
        return cls(
            A=0.9 * jnp.eye(state_dim),
            Q_chol=jnp.sqrt(q_init) * jnp.eye(state_dim),
            V=jnp.asarray(V),
            b=jnp.array(0.0),
            mu0=jnp.zeros(state_dim),
            Sigma0=jnp.eye(state_dim),
            **init_kwargs,
        )

    @property
    @override
    def Q(self) -> Array:
        L = jnp.tril(self.Q_chol)
        return L @ L.T + 1e-6 * jnp.eye(self.state_dim)

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

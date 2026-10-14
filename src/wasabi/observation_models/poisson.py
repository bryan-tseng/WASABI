from typing import Self, override

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.scipy.stats import poisson

from wasabi.observation_models.base import ObservationModel
from wasabi.utils import inv_softplus, polar, softplus


class PoissonObservations(ObservationModel):
    """y_t ~ Poisson(softplus(C x_t + d)).

    C has orthonormal columns (enforced by `project`), which pins the latent scale and
    leaves only an orthogonal ambiguity in the latent basis.
    """

    C: Array
    d: Array
    beta_emit: float = eqx.field(static=True, default=1.0)

    @classmethod
    def initialize(
        cls,
        num_channels: int,
        state_dim: int,
        key: Array,
        mean_rate: Array | None = None,
        **kwargs,
    ) -> Self:
        """Random orthonormal C, d matched to `mean_rate` when given."""
        beta_emit = kwargs.get("beta_emit", 1.0)
        d = (
            jnp.zeros(num_channels)
            if mean_rate is None
            else inv_softplus(jnp.clip(mean_rate, 1e-3), beta_emit)
        )
        return cls(C=polar(jr.normal(key, (num_channels, state_dim))), d=d, **kwargs)

    @property
    @override
    def num_channels(self) -> int:
        return self.C.shape[0]

    @override
    def rate(self, x: Array) -> Array:
        # floor keeps log(rate) and the diag(rate) noise away from zero
        return softplus(self.C @ x + self.d, self.beta_emit) + 1e-4

    @override
    def noise_var(self, rate: Array) -> Array:
        # Noise equals mean
        return rate

    @override
    def log_prob(self, y: Array, rate: Array) -> Array:
        return poisson.logpmf(y, rate).sum()

    @override
    def basis_change(self, old: Self) -> Array:
        return self.C.T @ old.C

    @override
    def project(self) -> Self:
        return eqx.tree_at(lambda m: m.C, self, polar(self.C))

from typing import Self

import equinox as eqx
from jax import Array

from wasabi.dynamics_models import DynamicsModel, LinearDynamics
from wasabi.observation_models import ObservationModel, PoissonObservations


class StateSpaceModel(eqx.Module):
    """A dynamics model and an observation model sharing one latent space.

    The inference code only talks to this class, so either component can be swapped
    without touching the filters.
    """

    dynamics: DynamicsModel
    observation: ObservationModel

    @property
    def state_dim(self) -> int:
        return self.dynamics.state_dim

    @property
    def num_channels(self) -> int:
        """Recorded channels."""
        return self.observation.num_channels

    @property
    def num_inputs(self) -> int:
        """Input channels u(s), i.e. columns of B."""
        return self.dynamics.num_inputs

    @property
    def num_stim_channels(self) -> int:
        """Length of a stim pattern. Can be fewer than the recorded channels."""
        return self.dynamics.num_stim_channels

    @property
    def Q(self) -> Array:
        return self.dynamics.Q

    def process_noise(self, s: Array) -> Array:
        return self.dynamics.process_noise(s)

    @property
    def mu0(self) -> Array:
        return self.dynamics.mu0

    @property
    def Sigma0(self) -> Array:
        return self.dynamics.Sigma0

    def stim_input(self, s: Array) -> Array:
        return self.dynamics.stim_input(s)

    def transition(self, x: Array, B: Array, s: Array) -> Array:
        return self.dynamics.transition(x, B, s)

    def jacobian(self, x: Array) -> Array:
        return self.dynamics.jacobian(x)

    def rate(self, x: Array) -> Array:
        return self.observation.rate(x)

    def noise_var(self, rate: Array) -> Array:
        return self.observation.noise_var(rate)

    def log_prob(self, y: Array, rate: Array) -> Array:
        return self.observation.log_prob(y, rate)

    def project(self) -> Self:
        return StateSpaceModel(self.dynamics.project(), self.observation.project())

    def basis_change(self, old: Self) -> Array:
        """M such that latents in `old`'s basis map to this model's basis, x_new = M x_old."""
        return self.observation.basis_change(old.observation)


def make_poisson_lds(
    state_dim: int,
    V: Array,
    key: Array,
    mean_rate: Array | None = None,
    beta_stim: float = 5.0,
    beta_emit: float = 1.0,
    num_channels: int | None = None,
) -> StateSpaceModel:
    """Linear dynamics with a softplus stim input, Poisson softplus observations.

    `num_channels` is the number of recorded channels; by default V.shape[0], i.e. every
    stimulated channel is also recorded and nothing else is.
    """
    return StateSpaceModel(
        LinearDynamics.initialize(state_dim, V, beta_stim=beta_stim),
        PoissonObservations.initialize(
            V.shape[0] if num_channels is None else num_channels, state_dim, key, mean_rate,
            beta_emit=beta_emit,
        ),
    )

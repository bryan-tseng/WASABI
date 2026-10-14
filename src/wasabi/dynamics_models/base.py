from abc import abstractmethod
from typing import Self

import equinox as eqx
import jax
from jax import Array

from wasabi.utils import stim_nonlinearity


class DynamicsModel(eqx.Module):
    """Latent dynamics with an additive stim term:

        x_t = f(x_{t-1}) + B u(s_t) + eps_t,     eps_t ~ N(0, Q)
        u(s) = stim_nonlinearity(apply_stim_kernel(s), b)

    B is not a field: it is either tracked by the joint filter or passed in fixed. It
    must enter linearly, since the joint EKF treats vec(B) as part of the state.
    Subclasses define f (`propagate`), Q, the stim kernel and the prior on x_{-1}.
    """

    b: eqx.AbstractVar[Array]
    mu0: eqx.AbstractVar[Array]
    Sigma0: eqx.AbstractVar[Array]
    beta_stim: eqx.AbstractVar[float]

    @property
    def state_dim(self) -> int:
        return self.mu0.shape[0]

    @property
    @abstractmethod
    def Q(self) -> Array:
        raise NotImplementedError

    @abstractmethod
    def propagate(self, x: Array) -> Array:
        """The autonomous part of the transition, f(x)."""
        raise NotImplementedError

    def jacobian(self, x: Array) -> Array:
        """d f / dx, used by the EKF prediction. Override when it is known in closed form."""
        return jax.jacfwd(self.propagate)(x)

    @abstractmethod
    def apply_stim_kernel(self, s: Array) -> Array:
        """Map a stim pattern s (N,) to the drive of each input channel, e.g. V s."""
        raise NotImplementedError

    def stim_input(self, s: Array) -> Array:
        return stim_nonlinearity(self.apply_stim_kernel(s), self.b, self.beta_stim)

    def transition(self, x: Array, B: Array, s: Array) -> Array:
        return self.propagate(x) + B @ self.stim_input(s)

    def project(self) -> Self:
        """Pull parameters back onto their constraint set after a gradient step."""
        return self

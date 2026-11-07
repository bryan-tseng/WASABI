from abc import abstractmethod
from typing import Self

import equinox as eqx
import jax
from jax import Array

from wasabi.utils import stim_nonlinearity


class DynamicsModel(eqx.Module):
    """Latent dynamics with an additive stim term:

        x_t = f(x_{t-1}) + B u(s_t) + eps_t, eps_t ~ N(0, process_noise(s_t))
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
    def num_inputs(self) -> int:
        """Number of input channels u(s), i.e. columns of B."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_stim_channels(self) -> int:
        """Length of a stim pattern s. Can differ from the number of recorded channels."""
        raise NotImplementedError

    @property
    @abstractmethod
    def Q(self) -> Array:
        raise NotImplementedError

    def process_noise(self, s: Array) -> Array:
        """Covariance of eps_t given the stim s_t. Defaults to Q; override to let stim add noise."""
        return self.Q

    @abstractmethod
    def propagate(self, x: Array) -> Array:
        """Propagate one step forward f(x)"""
        raise NotImplementedError

    def jacobian(self, x: Array) -> Array:
        """df / dx, used by the EKF prediction, equivalent to A. Override when it is known in closed form."""
        return jax.jacfwd(self.propagate)(x)

    @abstractmethod
    def apply_stim_kernel(self, s: Array) -> Array:
        """Map a stim pattern s (N,) to the drive of each input channel, e.g. Vs."""
        raise NotImplementedError

    def stim_input(self, s: Array) -> Array:
        """Apply nonlinearity to stim kernel"""
        return stim_nonlinearity(self.apply_stim_kernel(s), self.b, self.beta_stim)

    def transition(self, x: Array, B: Array, s: Array) -> Array:
        """Stim nonlinearity"""
        return self.propagate(x) + B @ self.stim_input(s)

    def project(self) -> Self:
        """Pull parameters back onto their constraint set after a gradient step."""
        return self

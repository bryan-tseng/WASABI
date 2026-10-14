from abc import abstractmethod
from typing import Self

import equinox as eqx
from jax import Array


class ObservationModel(eqx.Module):
    """y_t | x_t with mean rate(x_t) and variance noise_var(rate), independent across channels.

    The EKF linearizes `rate` and uses diag(noise_var) as the measurement covariance;
    `log_prob` gives the marginal likelihood and the iterated update's objective.
    """

    @property
    @abstractmethod
    def num_channels(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def rate(self, x: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def noise_var(self, rate: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, y: Array, rate: Array) -> Array:
        """log p(y | rate), summed over channels."""
        raise NotImplementedError

    @abstractmethod
    def basis_change(self, old: Self) -> Array:
        """M such that latents in `old`'s basis map to this model's basis, x_new = M x_old."""
        raise NotImplementedError

    def project(self) -> Self:
        """Pull parameters back onto their constraint set after a gradient step."""
        return self

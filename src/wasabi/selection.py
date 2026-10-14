"""D-optimal stim selection by rolling the joint EKF covariance through a trial.

The covariance recursion does not depend on the observed counts except through the
linearization point, so the expected end-of-trial covariance of vec(B) is approximated
by running the filter on its own predictions (zero innovation). Candidates are scored
by the log det of that covariance (smaller is better).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self, override

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array, lax

from wasabi.inference.ekf import ekf_update
from wasabi.inference.joint import BPosterior, initial_state, predict
from wasabi.model import StateSpaceModel


class TrialDesign(eqx.Module):
    """Timing shared by every stim trial: when stim is on and which bins are blanked."""

    keep: Array  # (T,) False where the emission is blanked
    t_stim: int = eqx.field(static=True)
    num_stim_bins: int = eqx.field(static=True)

    @classmethod
    def from_timevec(cls, time_vec: dict, num_blank: int | None = None) -> Self:
        """Build from `simulations.make_timevec`, blanking the stim bins by default."""
        num_blank = time_vec["n_stim_bins"] if num_blank is None else num_blank
        t_stim = time_vec["t_stim"]
        keep = jnp.ones(time_vec["T"], dtype=bool).at[t_stim : t_stim + num_blank].set(False)
        return cls(keep, t_stim, time_vec["n_stim_bins"])

    @property
    def num_timesteps(self) -> int:
        return self.keep.shape[0]

    def schedule(self, pattern: Array) -> Array:
        """(T, N) stim matrix with `pattern` held on through the stim window."""
        pattern = jnp.asarray(pattern, dtype=float)
        ss = jnp.zeros((self.num_timesteps, pattern.shape[0]))
        return ss.at[self.t_stim : self.t_stim + self.num_stim_bins].set(pattern)


def _rollout(
    model: StateSpaceModel, mean: Array, cov: Array, ss: Array, keep: Array
) -> tuple[Array, Array]:
    K = model.state_dim

    def step(carry, inputs):
        mean, cov = carry
        s, k = inputs
        mean, cov = predict(model, mean, cov, s)
        expected_y = model.rate(mean[:K])
        # zero innovation keeps x at the mode, so one plain EKF step is exact
        _, mean, cov = ekf_update(model, mean, cov, expected_y, k, num_iters=1)
        return (mean, cov), None

    (mean, cov), _ = lax.scan(step, (mean, cov), (ss, keep))
    return mean, cov


@eqx.filter_jit
def expected_final_logdets(
    model: StateSpaceModel, prior: BPosterior, candidates: Array, design: TrialDesign
) -> Array:
    """Expected log det of the end-of-trial B covariance for each candidate pattern."""
    K = model.state_dim
    t = design.t_stim
    zeros = jnp.zeros((t, model.num_channels))

    # Before stim onset every candidate sees the same (empty) input, so roll that once.
    mean, cov = _rollout(model, *initial_state(model, prior), zeros, design.keep[:t])

    def score(pattern):
        ss = design.schedule(pattern)[t:]
        _, final_cov = _rollout(model, mean, cov, ss, design.keep[t:])
        return jnp.linalg.slogdet(final_cov[K:, K:])[1]

    return jax.vmap(score)(jnp.asarray(candidates, dtype=float))


class SelectionStrategy(ABC):
    @abstractmethod
    def select(
        self,
        key: Array,
        model: StateSpaceModel,
        prior: BPosterior,
        candidates: Array,
        design: TrialDesign,
        trial: int,
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class DOptimal(SelectionStrategy):
    @override
    def select(self, key, model, prior, candidates, design, trial) -> int:
        return int(jnp.argmin(expected_final_logdets(model, prior, candidates, design)))


@dataclass(frozen=True)
class Tile(SelectionStrategy):
    schedule: tuple[int, ...]

    @override
    def select(self, key, model, prior, candidates, design, trial) -> int:
        return self.schedule[trial % len(self.schedule)]


@dataclass(frozen=True)
class Random(SelectionStrategy):
    @override
    def select(self, key, model, prior, candidates, design, trial) -> int:
        return int(jr.randint(key, (), 0, len(candidates)))


def tile_strategy(schedule: np.ndarray) -> Tile:
    return Tile(tuple(int(j) for j in schedule))

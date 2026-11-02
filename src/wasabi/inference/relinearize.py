"""Damped Gauss-Newton over a trial's trajectory: the iterated extended Kalman smoother.

The EKF linearizes the observation at the predicted mean, before seeing the counts. Iterating
instead re-linearizes every bin's observation at the smoothed mean of the previous pass and runs
the filter and smoother again. Each pass is one Gauss-Newton step on the trial's log posterior,
so it converges to the posterior mode, and the covariance of the last pass is the curvature at
the mode (the Laplace approximation). See Bell (1994), "The iterated Kalman smoother as a
Gauss-Newton method", and Paninski et al. (2010), "A new look at state-space models for neural
data".

Only the observation is re-linearized: with linear dynamics the transition is exactly linear in
the state (x, and vec(B) in the joint filter), so the prediction has nothing to re-linearize.

Damping: a full step can overshoot when the rate is very nonlinear, so each step is scaled by
the first of `STEP_SIZES` that increases the log posterior, or skipped if none does.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax

# Passes after the plain EKF pass. 0 (the default) is the plain EKF; set > 0 for the damped
# iterated smoother. In our tests iterating fit worse (higher NLL on real data, higher error in
# the closed-loop simulation), likely because the refit loss is still the plug-in likelihood at the
# predicted mean while the update is linearized at the smoothed path.
RELINEARIZE_ITERS = 0
STEP_SIZES = (1.0, 0.5, 0.25, 0.125)


def damped_gauss_newton(
    smooth: Callable,
    log_density: Callable,
    trajectory,
    num_iters: int,
):
    """Refine a linearization point by `num_iters` damped Gauss-Newton steps.

    smooth(trajectory) -> the smoothed trajectory of a pass linearized at `trajectory`
    log_density(trajectory) -> the trial's log posterior at `trajectory`
    trajectory: a pytree of arrays, e.g. the latent path, or (latent path, vec(B))
    """
    steps = jnp.array(STEP_SIZES)

    def step(current, _):
        proposal = smooth(current)
        candidates = jax.tree.map(
            lambda a, b: a + steps.reshape((-1,) + (1,) * a.ndim) * (b - a), current, proposal
        )
        improves = jax.vmap(log_density)(candidates) > log_density(current)
        # First (largest) step that improves, or keep the current point
        best = jnp.argmax(improves)
        current = jax.tree.map(
            lambda c, a: jnp.where(improves.any(), c[best], a), candidates, current
        )
        return current, None

    trajectory, _ = lax.scan(step, trajectory, None, length=num_iters)
    return trajectory

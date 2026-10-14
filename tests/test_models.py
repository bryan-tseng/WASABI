import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy.stats import ortho_group

from wasabi.fitting import TrialBuffer, negative_loglik
from wasabi.inference.joint import BPosterior, filter_trial
from wasabi.dynamics_models import DynamicsModel, LinearDynamics
from wasabi.model import StateSpaceModel, make_poisson_lds
from wasabi.observation_models import PoissonObservations
from wasabi.selection import TrialDesign, expected_final_logdets

K, N, T = 2, 6, 12


def make_model(key):
    model = make_poisson_lds(K, jnp.eye(N), key)
    dynamics = eqx.tree_at(
        lambda m: (m.A, m.b), model.dynamics, (3.0 * jr.normal(key, (K, K)), jnp.array(-0.3))
    )
    observation = eqx.tree_at(
        lambda m: m.C, model.observation, jr.normal(jr.split(key)[0], (N, K))
    )
    return StateSpaceModel(dynamics, observation)


def test_project_constraints():
    model = make_model(jr.key(0)).project()
    C, dynamics = model.observation.C, model.dynamics
    np.testing.assert_allclose(C.T @ C, jnp.eye(K), atol=1e-10)
    assert jnp.linalg.norm(dynamics.A, ord=2) <= dynamics.rho_max + 1e-10


def test_generic_jacobian_matches_linear():
    dynamics = make_model(jr.key(5)).dynamics
    x = jr.normal(jr.key(6), (K,))
    np.testing.assert_allclose(DynamicsModel.jacobian(dynamics, x), dynamics.A, rtol=1e-12)


def rotate(model, R):
    """Same generative model expressed in the latent basis x' = R x."""
    dyn, obs = model.dynamics, model.observation
    dynamics = LinearDynamics(
        A=R @ dyn.A @ R.T,
        Q_chol=jnp.linalg.cholesky(R @ dyn.Q @ R.T),
        V=dyn.V,
        b=dyn.b,
        mu0=R @ dyn.mu0,
        Sigma0=R @ dyn.Sigma0 @ R.T,
    )
    return StateSpaceModel(dynamics, PoissonObservations(C=obs.C @ R.T, d=obs.d))


def test_rotation_invariance_after_realign():
    key = jr.key(1)
    model = make_model(key).project()
    R = jnp.asarray(ortho_group.rvs(K, random_state=0))
    rotated = rotate(model, R)
    M = rotated.basis_change(model)
    np.testing.assert_allclose(M, R, atol=1e-10)

    prior = BPosterior.from_prior(K, N, prior_var=2.0)
    prior = BPosterior(jr.normal(key, (K * N,)), prior.cov, K)
    design = TrialDesign(jnp.ones(T, dtype=bool).at[4:6].set(False), 4, 2)
    ss = design.schedule(jnp.zeros(N).at[1:3].set(1.0))
    ys = jr.poisson(key, 2.0, (T, N)).astype(float)

    ll, post, _ = filter_trial(model, prior, ys, ss, design.keep)
    ll_rot, post_rot, _ = filter_trial(rotated, prior.realign(M), ys, ss, design.keep)

    # LinearDynamics.Q adds a 1e-6 jitter, which `rotate` counts twice, hence rtol 1e-6
    np.testing.assert_allclose(ll, ll_rot, rtol=1e-6)
    np.testing.assert_allclose(post.realign(M).mean, post_rot.mean, atol=1e-6)
    np.testing.assert_allclose(post.logdet(), post_rot.logdet(), rtol=1e-6)

    candidates = jnp.eye(N)
    np.testing.assert_allclose(
        expected_final_logdets(model, prior, candidates, design),
        expected_final_logdets(rotated, prior.realign(M), candidates, design),
        rtol=1e-6,
    )


def test_observing_stim_trial_shrinks_B_uncertainty():
    key = jr.key(2)
    model = make_model(key).project()
    prior = BPosterior.from_prior(K, N, prior_var=1.0)
    design = TrialDesign(jnp.ones(T, dtype=bool).at[4:6].set(False), 4, 2)
    pattern = jnp.zeros(N).at[0:2].set(1.0)
    ys = jr.poisson(key, 2.0, (T, N)).astype(float)

    _, post, _ = filter_trial(model, prior, ys, design.schedule(pattern), design.keep)
    _, post_none, _ = filter_trial(model, prior, ys, design.schedule(jnp.zeros(N)), design.keep)

    assert post.logdet() < prior.logdet()
    np.testing.assert_allclose(post_none.logdet(), prior.logdet(), rtol=1e-10)


def test_buffer_ignores_unused_slots():
    model = make_model(jr.key(3)).project()
    ys = jr.poisson(jr.key(4), 2.0, (2, T, N)).astype(float)
    ss = jnp.zeros((2, T, N))
    keep = jnp.ones(T, dtype=bool)
    B = jnp.zeros((K, N))

    small = TrialBuffer.empty(2, T, N).extend(ys, ss, keep)
    large = TrialBuffer.empty(10, T, N).extend(ys, ss, keep)

    np.testing.assert_allclose(
        negative_loglik(model, B, small), negative_loglik(model, B, large), rtol=1e-10
    )


def test_full_buffer_raises():
    ys = jnp.zeros((T, N))
    buffer = TrialBuffer.empty(1, T, N).add(ys, ys, jnp.ones(T, dtype=bool))
    with pytest.raises(eqx.EquinoxRuntimeError, match="TrialBuffer is full"):
        buffer.add(ys, ys, jnp.ones(T, dtype=bool))

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from wasabi.inference.ekf import ekf_update, stim_effect_cov
from wasabi.inference.joint import BPosterior, predict, transition_matrix
from wasabi.dynamics_models import LinearDynamics
from wasabi.model import StateSpaceModel, make_poisson_lds
from wasabi.utils import unvec, vec

K, N = 2, 5
D = K + K * N


@pytest.fixture
def model():
    key = jr.key(0)
    model = make_poisson_lds(K, jnp.eye(N), key)
    keys = jr.split(key, 3)
    dynamics = LinearDynamics(
        A=0.5 * jr.normal(keys[0], (K, K)),
        Q_chol=jnp.tril(jr.normal(keys[2], (K, K))) + jnp.eye(K),
        V=jnp.eye(N),
        b=jnp.array(-0.3),
        log_stim_var=jnp.log(0.5 + jnp.arange(N) / N),
        mu0=model.mu0,
        Sigma0=model.Sigma0,
    )
    observation = eqx.tree_at(lambda o: o.d, model.observation, jr.normal(keys[1], (N,)))
    return StateSpaceModel(dynamics, observation)


def random_psd(key, dim):
    L = jr.normal(key, (dim, dim))
    return L @ L.T / dim + 0.1 * jnp.eye(dim)


def test_vec_matches_kron():
    B = jr.normal(jr.key(1), (K, N))
    u = jr.normal(jr.key(2), (N,))
    np.testing.assert_allclose(B @ u, jnp.kron(u[None, :], jnp.eye(K)) @ vec(B))
    np.testing.assert_allclose(unvec(vec(B), K), B)


def test_update_matches_dense_ekf(model):
    mean = jr.normal(jr.key(3), (D,))
    cov = random_psd(jr.key(4), D)
    y = jnp.arange(N, dtype=float)

    _, mean_upd, cov_upd = ekf_update(model, mean, cov, y, jnp.array(True))

    rate = model.rate(mean[:K])
    H = jnp.concatenate([jax.jacfwd(model.rate)(mean[:K]), jnp.zeros((N, D - K))], 1)
    S = H @ cov @ H.T + jnp.diag(rate)
    gain = cov @ H.T @ jnp.linalg.inv(S)
    np.testing.assert_allclose(mean_upd, mean + gain @ (y - rate), rtol=1e-8)
    np.testing.assert_allclose(cov_upd, (jnp.eye(D) - gain @ H) @ cov, atol=1e-10)


def test_linearizing_at_prediction_matches_plain_update(model):
    mean = jr.normal(jr.key(11), (D,))
    cov = random_psd(jr.key(12), D)
    y = jnp.arange(N, dtype=float)
    plain = ekf_update(model, mean, cov, y, jnp.array(True))
    at_prediction = ekf_update(model, mean, cov, y, jnp.array(True), lin_x=mean[:K])
    for a, b in zip(plain, at_prediction):
        np.testing.assert_allclose(a, b, rtol=1e-12)


def test_masked_update_is_identity(model):
    mean = jr.normal(jr.key(5), (D,))
    cov = random_psd(jr.key(6), D)
    y = jnp.full(N, jnp.nan)

    loglik, mean_upd, cov_upd = ekf_update(model, mean, cov, y, jnp.array(False))

    assert loglik == 0.0
    np.testing.assert_array_equal(mean_upd, mean)
    np.testing.assert_array_equal(cov_upd, cov)


def test_structured_predict_matches_jacobian(model):
    mean = jr.normal(jr.key(7), (D,))
    cov = random_psd(jr.key(8), D)
    s = (jnp.arange(N) % 2).astype(float)

    pred_mean, pred_cov = predict(model, mean, cov, s)

    def augmented_transition(z):
        x, B = z[:K], unvec(z[K:], K)
        return jnp.concatenate([model.transition(x, B, s), z[K:]])

    F = jax.jacfwd(augmented_transition)(mean)
    # stim adds sum_n u_n^2 stim_var[n] to every latent's process noise
    u = model.stim_input(s)
    Q_s = model.Q + jnp.sum(u**2 * model.dynamics.stim_var) * jnp.eye(K)
    Q = jnp.zeros((D, D)).at[:K, :K].set(Q_s)
    np.testing.assert_allclose(pred_mean, augmented_transition(mean), rtol=1e-10)
    np.testing.assert_allclose(pred_cov, F @ cov @ F.T + Q, atol=1e-10)
    np.testing.assert_allclose(transition_matrix(model, mean[:K], s), F, atol=1e-12)


def test_stim_effect_cov_matches_kron():
    u = jr.normal(jr.key(9), (N,))
    B_cov = random_psd(jr.key(10), K * N)
    G = jnp.kron(u[None, :], jnp.eye(K))  # B u = (uᵀ ⊗ I) vec(B)
    np.testing.assert_allclose(stim_effect_cov(B_cov, u), G @ B_cov @ G.T, rtol=1e-10)


def test_realign_matches_kron():
    prior = BPosterior(
        jr.normal(jr.key(9), (K * N,)), random_psd(jr.key(10), K * N), K
    )
    M = jr.normal(jr.key(11), (K, K))
    T = jnp.kron(jnp.eye(N), M)

    realigned = prior.realign(M)

    np.testing.assert_allclose(realigned.mean, T @ prior.mean, rtol=1e-10)
    np.testing.assert_allclose(realigned.cov, T @ prior.cov @ T.T, atol=1e-10)
    np.testing.assert_allclose(realigned.B, M @ prior.B, rtol=1e-10)

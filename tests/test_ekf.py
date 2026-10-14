import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from wasabi.inference.ekf import ekf_update
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


def test_iterated_update_finds_posterior_mode(model):
    mean = jr.normal(jr.key(12), (D,))
    cov = random_psd(jr.key(13), D)
    # moderately above the predicted rate: the undamped iteration converges here, but
    # can oscillate for much larger surprises (e.g. 10x these counts)
    y = 2.0 * jnp.arange(N, dtype=float)

    _, mean_upd, cov_upd = ekf_update(model, mean, cov, y, jnp.array(True), 50)

    def log_posterior(x):
        rate = model.rate(x)
        r = x - mean[:K]
        return (y * jnp.log(rate) - rate).sum() - 0.5 * r @ jnp.linalg.solve(cov[:K, :K], r)

    x_post = mean_upd[:K]
    np.testing.assert_allclose(jax.grad(log_posterior)(x_post), 0.0, atol=1e-6)
    _, mean_ekf, _ = ekf_update(model, mean, cov, y, jnp.array(True))
    assert log_posterior(x_post) >= log_posterior(mean_ekf[:K])
    # the rest of the state moves by the Gaussian regression on x
    gain = cov[K:, :K] @ jnp.linalg.inv(cov[:K, :K])
    np.testing.assert_allclose(mean_upd[K:], mean[K:] + gain @ (x_post - mean[:K]), rtol=1e-8)
    assert jnp.all(jnp.linalg.eigvalsh(cov_upd) > 0)


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
    Q = jnp.zeros((D, D)).at[:K, :K].set(model.Q)
    np.testing.assert_allclose(pred_mean, augmented_transition(mean), rtol=1e-10)
    np.testing.assert_allclose(pred_cov, F @ cov @ F.T + Q, atol=1e-10)
    np.testing.assert_allclose(transition_matrix(model, mean[:K], s), F, atol=1e-12)


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

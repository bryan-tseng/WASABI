import jax.numpy as jnp
import numpy as np
from jax import Array

# Helper functions
def softplus(z: Array, beta: float = 5.0) -> Array:
    return jnp.logaddexp(0.0, beta * z) / beta


def inv_softplus(y: Array, beta: float = 5.0) -> Array:
    return y + jnp.log(-jnp.expm1(-beta * y)) / beta


def stim_nonlinearity(drive: Array, b: Array, beta: float = 5.0) -> Array:
    """u = softplus(drive + b) - softplus(b), so zero drive gives zero input."""
    return softplus(drive + b, beta) - softplus(b, beta)


def vec(B: Array) -> Array:
    """Column-major vec, so that `B @ u == kron(u, I_K) @ vec(B)`."""
    return B.T.reshape(-1)


def unvec(v: Array, num_latents: int) -> Array:
    return v.reshape(-1, num_latents).T


def polar(M: Array) -> Array:
    """Closest matrix with orthonormal columns, in Frobenius norm."""
    U, _, Vt = jnp.linalg.svd(M, full_matrices=False)
    return U @ Vt


def symmetrize(P: Array) -> Array:
    return 0.5 * (P + P.swapaxes(-1, -2))

# Alignment of C
def alignment(C_target, C_source) -> tuple[np.ndarray, np.ndarray]:
    """Change-of-basis matrices aligning C_source's latents into C_target's basis.

    Returns (rot, rot_inv) with rot_inv = pinv(C_source) @ C_target.
    """
    C_target, C_source = np.asarray(C_target), np.asarray(C_source)
    rot_inv = np.linalg.pinv(C_source) @ C_target
    return np.linalg.pinv(rot_inv), rot_inv

# Evaluating B
def projected_error(B_est, B_ref, Pu) -> float:
    """Relative error of B restricted to the input directions spanned by the projector `Pu`."""
    B_est, B_ref = np.asarray(B_est), np.asarray(B_ref)
    return float(np.linalg.norm((B_est - B_ref) @ Pu) / np.linalg.norm(B_ref @ Pu))

def truncated_projector(U, energy: float = 0.99) -> tuple[np.ndarray, int]:
    """Projector onto the directions the effective inputs `U` actually probe.

    V smears stim patterns spatially, so the candidate set only reaches a subspace of
    input space. Returns the projector onto the top singular directions holding
    `energy` of U's energy, and the rank of that subspace.
    """
    _, sv, V_trans = np.linalg.svd(np.asarray(U), full_matrices=False)
    r = int(np.searchsorted(np.cumsum(sv**2) / np.sum(sv**2), energy) + 1)
    return V_trans[:r].T @ V_trans[:r], r

# For simulated experiments
def make_tiling_schedule(n_cand: int, n_stim_ch: int) -> np.ndarray:
    """Deterministic scan over contiguous blocks.

    Walks the candidates in strides of n_stim_ch, one offset at a time. For a
    power-of-two n_stim_ch the offsets are visited in bit-reversed order, so early
    trials are spread across the array instead of clustered.
    """
    if n_stim_ch & (n_stim_ch - 1) == 0:
        width = n_stim_ch.bit_length() - 1
        offsets = sorted(
            range(n_stim_ch), key=lambda o: int(format(o, f"0{width}b")[::-1], 2)
        )
    else:
        offsets = list(range(n_stim_ch))
    return np.array(
        [j for off in offsets for j in range(off, n_cand, n_stim_ch)], dtype=int
    )

def contiguous_blocks(N: int, width: int) -> np.ndarray:
    """(N - width + 1, N) binary patterns, each switching on one contiguous block."""
    starts = np.arange(N - width + 1)
    idx = np.arange(N)
    return ((idx >= starts[:, None]) & (idx < starts[:, None] + width)).astype(float)
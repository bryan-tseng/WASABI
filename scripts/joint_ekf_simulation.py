"""Closed-loop identification of B with the joint EKF, on simulated Poisson LDS data.

Fits the dynamics on control trials, then runs DOptimal / Tile / Random stim selection
for several seeds, refitting the dynamics after every trial. Saves the per-trial error
curves to results/ and a summary figure to figures/.

    uv run python scripts/joint_ekf_simulation.py --num-trials 100 --num-seeds 5
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import subspace_angles
from simulations import (
    blank_stim_artifact,
    effective_stim_input,
    make_ground_truth_params,
    make_structured_B,
    make_timevec,
    simulate_trial,
)

from wasabi import (
    BPosterior,
    DOptimal,
    Random,
    TrialBuffer,
    TrialDesign,
    fit_initial,
    init_optimizer,
    make_poisson_lds,
    run_adaptive,
    tile_strategy,
)
from wasabi.fitting import REFIT_FIELDS
from wasabi.utils import (
    contiguous_blocks,
    make_tiling_schedule,
    truncated_projector,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--num-control", type=int, default=100)
    parser.add_argument("--latents", type=int, default=4)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--stim-width", type=int, default=8)
    parser.add_argument("--B-scale", type=float, default=1.0)
    parser.add_argument("--prior-var", type=float, default=1.0)
    parser.add_argument("--refit-steps", type=int, default=20)
    parser.add_argument("--freeze-b", action="store_true", help="do not refit b between trials")
    parser.add_argument("--filter-iters", type=int, default=1, help="> 1 for the iterated EKF in the joint filter")
    parser.add_argument("--strategies", nargs="+", default=["DOptimal", "Tile", "Random"])
    return parser.parse_args()


def make_simulator(params, time_vec, rng, blank=True):
    def simulate(pattern):
        _, y, _ = simulate_trial(
            params,
            time_vec["T"],
            np.flatnonzero(pattern),
            time_vec["t_stim"],
            time_vec["n_stim_bins"],
            rng,
        )
        if not blank:
            return y.astype(float)
        return blank_stim_artifact(y[None], time_vec["t_stim"], time_vec["n_stim_bins"])[0]

    return simulate


def make_evaluator(params, candidates, beta_stim):
    """Relative error of the stim effect B u over the candidate set, in the fitted basis.

    B itself is not compared directly: with V = I and binary stim, u = c(b) s, so b and
    the scale of B trade off and only the product B u is identifiable.
    """
    U_true = effective_stim_input(candidates, params["V"], params["b"], beta_stim)
    C_true, B_true = params["C"], params["B"]

    def evaluate(model, prior):
        # C_fit x_fit = C_true x_true, and C_fit has orthonormal columns
        B_ref = np.asarray(model.observation.C.T) @ C_true @ B_true
        U_fit = np.asarray(jax.vmap(model.stim_input)(jnp.asarray(candidates)))
        true_effect = U_true @ B_ref.T
        pred_effect = U_fit @ np.asarray(prior.B).T
        return {
            "effect_err": np.linalg.norm(pred_effect - true_effect)
            / np.linalg.norm(true_effect)
        }

    return evaluate


def main():
    args = parse_args()
    K, N = args.latents, args.channels
    beta_stim, beta_emit = 5.0, 1.0

    rng = np.random.default_rng(0)
    params = make_ground_truth_params(
        K=K, N=N, spectral_radius=0.9, sigma2_Q=0.2, b_input=-0.5, seed=0, identity_V=True
    )
    params["B"] = make_structured_B(K, N, rng, n_sites=10, site_w=4.0, B_scale=args.B_scale)

    time_vec = make_timevec()
    design = TrialDesign.from_timevec(time_vec)
    candidates = contiguous_blocks(N, args.stim_width)
    U_true = effective_stim_input(candidates, params["V"], params["b"], beta_stim)
    _, rank = truncated_projector(U_true)
    print(f"{len(candidates)} candidates, reachable input rank {rank} of {N}")

    # Control trials: no stim, nothing blanked
    simulate = make_simulator(params, time_vec, np.random.default_rng(1), blank=False)
    Y_control = np.stack([simulate(np.zeros(N)) for _ in range(args.num_control)])
    keep_control = jnp.ones(time_vec["T"], dtype=bool)
    S_control = jnp.zeros(Y_control.shape)

    buffer = TrialBuffer.empty(args.num_control + args.num_trials, time_vec["T"], N)
    buffer = buffer.extend(jnp.asarray(Y_control), S_control, keep_control)

    model = make_poisson_lds(
        K,
        jnp.asarray(params["V"]),
        jr.key(0),
        mean_rate=jnp.asarray(Y_control.mean(axis=(0, 1))),
        beta_stim=beta_stim,
        beta_emit=beta_emit,
    )
    model, losses = fit_initial(model, buffer, num_steps=1000)
    print(f"initial fit NLL per bin: {losses[0]:.4f} -> {losses[-1]:.4f}")
    angles = np.degrees(subspace_angles(np.asarray(model.observation.C), params["C"]))
    print(f"principal angles between fitted and true C (deg): {np.round(angles, 1)}")

    strategies = {
        "DOptimal": DOptimal(),
        "Tile": tile_strategy(make_tiling_schedule(len(candidates), args.stim_width)),
        "Random": Random(),
    }
    evaluate = make_evaluator(params, candidates, beta_stim)
    refit_fields = tuple(f for f in REFIT_FIELDS if not (args.freeze_b and f == "dynamics.b"))

    results = {}
    for name in args.strategies:
        runs = []
        for seed in range(args.num_seeds):
            optimizer, opt_state = init_optimizer(model, fields=refit_fields)
            _, _, _, history = run_adaptive(
                model,
                BPosterior.from_prior(K, N, args.prior_var),
                strategies[name],
                candidates,
                design,
                make_simulator(params, time_vec, np.random.default_rng(100 + seed)),
                buffer,
                optimizer,
                opt_state,
                args.num_trials,
                jr.key(seed),
                refit_steps=args.refit_steps,
                refit_fields=refit_fields,
                filter_iters=args.filter_iters,
                evaluate=evaluate,
            )
            runs.append(history)
            print(
                f"{name} seed {seed}: effect err {history['effect_err'][-1]:.3f}, "
                f"distinct patterns {len(np.unique(history['choice']))}"
            )
        results[name] = {key: np.stack([r[key] for r in runs]) for key in runs[0]}

    (ROOT / "results").mkdir(exist_ok=True)
    np.savez(
        ROOT / "results" / "joint_ekf_simulation.npz",
        **{f"{name}_{key}": v for name, r in results.items() for key, v in r.items()},
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    trials = np.arange(1, args.num_trials + 1)
    for name, r in results.items():
        for ax, key, label in zip(
            axes,
            ("effect_err", "logdet"),
            ("Relative error of B u over candidates", r"$\log\det \Sigma_B$"),
        ):
            lo, med, hi = np.percentile(r[key], [25, 50, 75], axis=0)
            line = ax.plot(trials, med, lw=1.8, label=name)[0]
            ax.fill_between(trials, lo, hi, color=line.get_color(), alpha=0.2, lw=0)
            ax.set_xlabel("Trial")
            ax.set_title(label, fontsize=10)
    axes[0].legend()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "joint_ekf_simulation.png", dpi=150)

    for name, r in results.items():
        print(
            f"{name}: final effect err {r['effect_err'][:, -1].mean():.3f}, "
            f"logdet drop {r['logdet'][:, 0].mean() - r['logdet'][:, -1].mean():.1f}"
        )


if __name__ == "__main__":
    main()

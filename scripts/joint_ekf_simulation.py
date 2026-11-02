"""Closed-loop identification of B with the joint EKF, on simulated Poisson LDS data.

For each of several ground truths (different A, C, d, Q, B), fits the dynamics on control
trials, then runs DOptimal / Tile / Random stim selection for several seeds, refitting the
dynamics after every trial. Saves the per-trial error curves to results/ and a summary
figure to figures/.

    uv run python scripts/joint_ekf_simulation.py --num-trials 100 --num-ground-truths 5 --num-seeds 3
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
    make_toeplitz_V,
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
from wasabi.inference import RELINEARIZE_ITERS
from wasabi.utils import (
    contiguous_blocks,
    make_tiling_schedule,
    truncated_projector,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--num-ground-truths", type=int, default=5)
    parser.add_argument(
        "--first-ground-truth", type=int, default=0,
        help="index of the first ground truth, to split a run across processes",
    )
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-control", type=int, default=100)
    parser.add_argument("--latents", type=int, default=4)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--stim-width", type=int, default=8)
    parser.add_argument("--B-scale", type=float, default=1.0)
    parser.add_argument("--prior-var", type=float, default=1.0)
    parser.add_argument("--refit-steps", type=int, default=20)
    parser.add_argument("--freeze-b", action="store_true", help="do not refit b between trials")
    parser.add_argument(
        "--stim-var", type=float, default=1.0,
        help="trial-to-trial variance of B on the unreliable channels, relative to mean(B**2)",
    )
    parser.add_argument(
        "--stim-gain-sd", type=float, default=0.0,
        help="sd of the log per-channel stim gain (0: every channel equally effective)",
    )
    parser.add_argument(
        "--freeze-stim-var", action="store_true",
        help="do not fit the stim-response variance (the model assumes consistent responses)",
    )
    parser.add_argument("--strategies", nargs="+", default=["DOptimal", "Tile", "Random"])
    parser.add_argument("--tag", default="", help="suffix for the results and figure file names")
    parser.add_argument(
        "--relinearize-iters", type=int, default=RELINEARIZE_ITERS,
        help="iterated-smoother passes per trial (0, the default: plain EKF)",
    )
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


def make_stim_var(B, rng, scale, num_bands=3, band_width=16):
    """Per-channel trial-to-trial variance of B: `num_bands` contiguous bands of unreliable channels."""
    N = B.shape[1]
    stim_var = np.zeros(N)
    for start in rng.choice(N - band_width, num_bands, replace=False):
        stim_var[start : start + band_width] = scale * np.mean(B**2)
    return stim_var


def make_stim_gain(N, rng, sd, smooth_w=4.0):
    """Per-channel stim gain exp(sd * z), z a smooth unit-variance profile across channels.

    Applied as V <- diag(gain) V, so both the simulator and the model (which is given the
    true V) know how effective each channel is. Low-gain channels fall below the softplus
    threshold and barely drive the latents.
    """
    z = make_toeplitz_V(N, smooth_w) @ rng.standard_normal(N)
    return np.exp(sd * z / z.std())


def make_evaluator(params, candidates, beta_stim):
    """Relative error of the stim effect B u over the candidate set, in the fitted basis.

    B itself is not compared directly: with V = I and binary stim, u = c(b) s, so b and
    the scale of B trade off and only the product B u is identifiable.
    """
    U_true = effective_stim_input(candidates, params["V"], params["b"], beta_stim)
    C_true, B_true = params["C"], params["B"]

    stim_var_true = params.get("stim_var", np.zeros(B_true.shape[1]))

    def evaluate(model, prior):
        # C_fit x_fit = C_true x_true, and C_fit has orthonormal columns
        B_ref = np.asarray(model.observation.C.T) @ C_true @ B_true
        U_fit = np.asarray(jax.vmap(model.stim_input)(jnp.asarray(candidates)))
        true_effect = U_true @ B_ref.T
        pred_effect = U_fit @ np.asarray(prior.B).T
        # Extra latent noise variance each candidate adds per stim bin, true vs fitted
        var_true = (U_true**2) @ stim_var_true
        var_fit = (U_fit**2) @ np.asarray(model.dynamics.stim_var)
        return {
            "effect_err": np.linalg.norm(pred_effect - true_effect)
            / np.linalg.norm(true_effect),
            "stim_var_corr": np.corrcoef(var_true, var_fit)[0, 1] if var_true.any() else np.nan,
        }

    return evaluate


def run_ground_truth(gt, args, strategies_for):
    """Fit one ground truth on control trials, then run every strategy for every seed.

    Returns {strategy: {metric: (num_seeds, num_trials)}}.
    """
    K, N = args.latents, args.channels
    beta_stim, beta_emit = 5.0, 1.0

    rng = np.random.default_rng(gt)
    params = make_ground_truth_params(
        K=K, N=N, spectral_radius=0.9, sigma2_Q=0.2, b_input=-0.5, seed=gt, identity_V=True
    )
    params["B"] = make_structured_B(K, N, rng, n_sites=10, site_w=4.0, B_scale=args.B_scale)
    if args.stim_var > 0:
        params["stim_var"] = make_stim_var(params["B"], rng, args.stim_var)
    if args.stim_gain_sd > 0:
        # Own generator, so the other ground-truth draws do not change with this flag
        gain = make_stim_gain(N, np.random.default_rng(10_000 + gt), args.stim_gain_sd)
        params["V"] = gain[:, None] * params["V"]
        single = effective_stim_input(np.eye(N), params["V"], params["b"], beta_stim).diagonal()
        reference = effective_stim_input(np.eye(1), np.eye(1), params["b"], beta_stim).item()
        print(
            f"[gt {gt}] stim gain range {gain.min():.2f}-{gain.max():.2f}; channels with under "
            f"10% of a unit-gain channel's input: {(single < 0.1 * reference).mean():.0%}"
        )

    time_vec = make_timevec()
    design = TrialDesign.from_timevec(time_vec)
    candidates = contiguous_blocks(N, args.stim_width)
    U_true = effective_stim_input(candidates, params["V"], params["b"], beta_stim)
    _, rank = truncated_projector(U_true)
    print(f"[gt {gt}] {len(candidates)} candidates, reachable input rank {rank} of {N}")

    # Control trials: no stim, nothing blanked
    simulate = make_simulator(params, time_vec, np.random.default_rng(1000 * gt + 1), blank=False)
    Y_control = np.stack([simulate(np.zeros(N)) for _ in range(args.num_control)])
    keep_control = jnp.ones(time_vec["T"], dtype=bool)
    S_control = jnp.zeros(Y_control.shape)

    buffer = TrialBuffer.empty(args.num_control + args.num_trials, time_vec["T"], N)
    buffer = buffer.extend(jnp.asarray(Y_control), S_control, keep_control)

    model = make_poisson_lds(
        K,
        jnp.asarray(params["V"]),
        jr.key(gt),
        mean_rate=jnp.asarray(Y_control.mean(axis=(0, 1))),
        beta_stim=beta_stim,
        beta_emit=beta_emit,
    )
    model, losses = fit_initial(
        model, buffer, num_steps=1000, relinearize_iters=args.relinearize_iters
    )
    angles = np.degrees(subspace_angles(np.asarray(model.observation.C), params["C"]))
    print(
        f"[gt {gt}] initial fit NLL per bin {losses[0]:.4f} -> {losses[-1]:.4f}, "
        f"angles to true C (deg) {np.round(angles, 1)}"
    )

    strategies = strategies_for(len(candidates))
    evaluate = make_evaluator(params, candidates, beta_stim)
    frozen = {"dynamics.b"} if args.freeze_b else set()
    if args.freeze_stim_var:
        frozen.add("dynamics.log_stim_var")
    refit_fields = tuple(f for f in REFIT_FIELDS if f not in frozen)

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
                make_simulator(params, time_vec, np.random.default_rng(1000 * gt + 100 + seed)),
                buffer,
                optimizer,
                opt_state,
                args.num_trials,
                jr.key(seed),
                refit_steps=args.refit_steps,
                refit_fields=refit_fields,
                evaluate=evaluate,
                relinearize_iters=args.relinearize_iters,
            )
            runs.append(history)
            print(
                f"[gt {gt}] {name} seed {seed}: effect err {history['effect_err'][-1]:.3f}, "
                f"stim var corr {history['stim_var_corr'][-1]:.2f}, "
                f"distinct patterns {len(np.unique(history['choice']))}"
            )
        results[name] = {key: np.stack([r[key] for r in runs]) for key in runs[0]}

    return results


def plot_results(results, num_trials):
    """Median and IQR over every (ground truth, seed) run, plus final error per ground truth."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    trials = np.arange(1, num_trials + 1)
    for name, r in results.items():
        for ax, key, label in zip(
            axes[:2],
            ("effect_err", "logdet"),
            ("Relative error of B u over candidates", r"$\log\det \Sigma_B$"),
        ):
            data = r[key].reshape(-1, num_trials)
            lo, med, hi = np.percentile(data, [25, 50, 75], axis=0)
            line = ax.plot(trials, med, lw=1.8, label=name)[0]
            ax.fill_between(trials, lo, hi, color=line.get_color(), alpha=0.2, lw=0)
            ax.set_xlabel("Trial")
            ax.set_title(label, fontsize=10)

    # Final error per ground truth, one marker per seed, strategies side by side
    num_gt = next(iter(results.values()))["effect_err"].shape[0]
    width = 0.8 / len(results)
    for i, (name, r) in enumerate(results.items()):
        final = r["effect_err"][:, :, -1]  # (num_gt, num_seeds)
        x = np.arange(num_gt) + (i - (len(results) - 1) / 2) * width
        axes[2].scatter(np.repeat(x, final.shape[1]), final.ravel(), s=12, alpha=0.6)
        axes[2].scatter(x, final.mean(axis=1), marker="_", s=200, color="k", lw=2)
    axes[2].set_xticks(np.arange(num_gt))
    axes[2].set_xlabel("Ground truth")
    axes[2].set_title("Final effect error (bar = mean over seeds)", fontsize=10)

    axes[0].legend()
    return fig


def plot_selections(results, candidates, gt=0, seed=0):
    """Stim channels chosen on each trial by each strategy, for one ground truth and seed."""
    fig, axes = plt.subplots(
        1, len(results), figsize=(3 * len(results), 3), constrained_layout=True, sharey=True
    )
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
        choices = r["choice"][gt, seed].astype(int)
        ax.imshow(
            candidates[choices].T, aspect="auto", interpolation="nearest", vmin=0, vmax=1,
            cmap="Greys",
        )
        ax.set_title(f"{name} ({len(np.unique(choices))} distinct)", fontsize=10)
        ax.set_xlabel("Trial")
    np.atleast_1d(axes)[0].set_ylabel("Channel")
    fig.suptitle(f"Stim channels selected, ground truth {gt}, seed {seed}", fontsize=10)
    return fig


def main():
    args = parse_args()

    def strategies_for(num_candidates):
        return {
            "DOptimal": DOptimal(),
            "Tile": tile_strategy(make_tiling_schedule(num_candidates, args.stim_width)),
            "Random": Random(),
        }

    # {strategy: {metric: (num_ground_truths, num_seeds, num_trials)}}
    gts = range(args.first_ground_truth, args.first_ground_truth + args.num_ground_truths)
    per_gt = [run_ground_truth(gt, args, strategies_for) for gt in gts]
    results = {
        name: {key: np.stack([r[name][key] for r in per_gt]) for key in per_gt[0][name]}
        for name in args.strategies
    }

    (ROOT / "results").mkdir(exist_ok=True)
    np.savez(
        ROOT / "results" / f"joint_ekf_simulation{args.tag}.npz",
        **{f"{name}_{key}": v for name, r in results.items() for key, v in r.items()},
    )
    (ROOT / "figures").mkdir(exist_ok=True)
    plot_results(results, args.num_trials).savefig(
        ROOT / "figures" / f"joint_ekf_simulation{args.tag}.png", dpi=150
    )
    plot_selections(results, contiguous_blocks(args.channels, args.stim_width)).savefig(
        ROOT / "figures" / f"joint_ekf_simulation{args.tag}_selection.png", dpi=150
    )

    print("\nFinal stim-effect error, mean over seeds:")
    print("  gt  " + "".join(f"{name:>10}" for name in results))
    for i, gt in enumerate(gts):
        row = "".join(f"{r['effect_err'][i, :, -1].mean():10.3f}" for r in results.values())
        print(f"  {gt:<4}{row}")
    for name, r in results.items():
        final = r["effect_err"][:, :, -1]
        print(
            f"{name}: final effect err {final.mean():.3f} +/- {final.std():.3f}, "
            f"logdet drop {(r['logdet'][..., 0] - r['logdet'][..., -1]).mean():.1f}"
        )
    if "DOptimal" in results:
        dopt = results["DOptimal"]["effect_err"][:, :, -1].mean(axis=1)
        for name, r in results.items():
            if name != "DOptimal":
                wins = (dopt < r["effect_err"][:, :, -1].mean(axis=1)).sum()
                print(f"DOptimal beats {name} on {wins} of {len(dopt)} ground truths")


if __name__ == "__main__":
    main()

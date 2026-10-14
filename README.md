# WASABI
Weighted Adaptive Stimulation After Batch-testing Identification (WASABI)

# Overview
This repository contains the simulation code for the identification of stimulation parameters.  

Folder structure:
```
.
├── docs                       # Problem formulation (joint_ekf.tex)
├── src
│   └── wasabi
│       ├── dynamics_models    # DynamicsModel base; LinearDynamics (A, Q, V, b)
│       ├── observation_models # ObservationModel base; PoissonObservations (C, d)
│       ├── model.py           # StateSpaceModel(dynamics, observation), make_poisson_lds
│       ├── inference/ekf.py   # blanking-aware (optionally iterated) EKF; latent filter/smoother
│       ├── inference/joint.py # joint EKF over [x; vec(B)], BPosterior
│       ├── fitting.py         # refit / fit_initial / fit_batch (EM), TrialBuffer
│       ├── selection.py       # expected end-of-trial log det, DOptimal / Tile / Random
│       └── experiment.py      # observe_trial, run_adaptive, run_batch
├── simulations                # uv workspace package: ground-truth Poisson LDS simulator
├── scripts                    # notebooks and joint_ekf_simulation.py
├── tests                      # pytest
├── LICENSE
└── README.md
```
Outputs:
```
.
├── figures                      # Output figures
├── results                      # Output files
```

---
# Installing the pipeline
Requires **Python >= 3.12** (`jax >= 0.11`).

The project is managed with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/bryan-tseng/WASABI
cd WASABI
uv sync --extra cuda
```

`uv sync` creates `.venv/` and installs the GPU build.
Run scripts with `uv run python ...`, or activate the environment with
`source .venv/bin/activate` (`.venv\Scripts\activate` on Windows).

### Without uv
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt   # pinned transitive deps
pip install -e . -e simulations   # or: pip install -e ".[cuda]" -e simulations
```

---
# Details of the pipeline
Model (see `docs/problem_formulation/joint_ekf.tex`):

    x_t = A x_{t-1} + B u(s_t) + eps_t,   u(s) = softplus(V s + b) - softplus(b)
    y_t ~ Poisson(softplus(C x_t + d))

1. **Control fit** (`fit_initial`): fit A, C, d, Q on control trials, where the stim
   input is zero and B drops out. C is kept orthonormal, which pins the latent scale.
2. **Per stim trial** (`observe_trial`):
   1. the joint EKF tracks the augmented state [x; vec(B)] through the trial, with the
      Poisson noise diag(rate) and blanked bins handled in the update;
   2. A, C, d, Q, b are refit on all trials so far, with B at its posterior mean;
   3. the posterior over B is rotated into the refit latent basis and carried to the
      next trial as its prior.
3. **Stim selection** (`selection.py`): each candidate pattern is scored by the expected
   log det of the B covariance at the end of the trial, by rolling the joint EKF
   covariance forward. `DOptimal` picks the smallest; `Tile` and `Random` are baselines.
4. **Recorded data** (`fit_batch`): EM over a fixed set of trials, alternating a joint EKF
   pass over every trial with a refit of A, C, d, Q, b.

The filters use the plain EKF update by default. When stim drives the latents far from
baseline (strong B, weak loadings), pass `filter_iters > 1` for the iterated EKF, which
stays stable there and identifies B better (see `scripts/dynamics_simulation.ipynb`).

Run the tests with `uv run pytest`, and the closed-loop simulation with
`uv run python scripts/joint_ekf_simulation.py`.

## Outputs
1. `results/joint_ekf_simulation.npz`: per-trial stim-effect error and log det of the
   B covariance, for each strategy and seed.
2. `figures/joint_ekf_simulation.png`: the same, as median and interquartile range.


## TODOs

- 

---

## Evaluation

## Contact
For questions, contact Bryan Tseng btseng2@jh.edu.

## Acknowledgements
Adam S. Charles, Sai Koukuntla.

## License
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
│       ├── inference/ekf.py   # blanking-aware EKF; latent filter/smoother
│       ├── inference/joint.py # joint EKF over [x; vec(B)], BPosterior
│       ├── inference/relinearize.py # damped iterated smoother (re-linearization)
│       ├── fitting.py         # refit / fit_initial / fit_batch (EM), TrialBuffer
│       ├── selection.py       # expected end-of-trial log det, DOptimal / Tile / Random
│       └── experiment.py      # observe_trial, run_adaptive, run_batch
├── experiment                 # closed-loop WASABI during a recording (see "Running the experiment")
│   ├── config
│   │   ├── online.yaml        # session, stim, trial, spike detection and model settings
│   │   ├── machines.yaml      # data_root_prefix of each machine, by hostname
│   │   ├── probes             # channel maps (positions for the local reference)
│   │   └── waveform_templates # spike waveforms for the matched filter
│   └── src
│       ├── wasabi_online.py   # the experiment: fit, choose a stim, update after each trial
│       ├── control_fitting.py # cut a control recording into control trials
│       ├── intan_binning.py   # spike detection and binning from RHX amplifier.dat
│       ├── config_loading.py  # reads online.yaml
│       ├── offline_data.py    # recorded sessions replayed by --source offline
│       └── tcp                # live RHX data over TCP
├── simulations                # uv workspace package: ground-truth Poisson LDS simulator
├── scripts                    # notebooks (e.g. inspect_amplifier.ipynb) and joint_ekf_simulation.py
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

The filters can optionally re-linearize the Poisson observation around the smoothed trajectory
and run again (a damped iterated extended Kalman smoother, `inference/relinearize.py`), so the
posterior is the Laplace approximation at the trial's posterior mode. `relinearize_iters`
(`--relinearize-iters` in the script) sets the number of passes; the default 0 is the plain EKF,
which fit better in our tests.

Run the tests with `uv run pytest`, and the closed-loop simulation with
`uv run python scripts/joint_ekf_simulation.py`.

## Outputs
1. `results/joint_ekf_simulation.npz`: per-trial stim-effect error and log det of the
   B covariance, for each strategy and seed.
2. `figures/joint_ekf_simulation.png`: the same, as median and interquartile range.

---
# Running the experiment
`experiment/src/wasabi_online.py` runs WASABI closed-loop during an Intan RHX recording.
It fits the dynamics on control trials, prints the stim for the next trial, and after
each trial reads it from the recording, updates the model and prints the next stim.
Settings are in `experiment/config/online.yaml`.

### Setup
1. In `experiment/config/machines.yaml`, set `data_root_prefix` for this machine's
   hostname (`hostname` prints it). Data paths in `online.yaml` are relative to it.
2. In `online.yaml`, under `paths`:
   - `wasabi_data_dir`: the RHX recording of the session (a new timestamped folder
     each time RHX starts recording);
   - `control_data_dir`: a no-stim reaching recording, for the control fit.

   Paths are relative to `data_root_prefix` (no leading `/`), or absolute.
3. Check `recording.num_channels` (enabled amplifier channels), `trial.ir_channel` and
   `trial.ir_edge` (the IR beam input), and the `spikes` detection settings.

In RHX, save as **"One File Per Signal Type"**: the code reads `amplifier.dat`,
`digitalin.dat`, `time.dat` and `settings.xml` directly.

### Steps
```bash
# 1. once per control recording: detect spikes, cut trials at the IR crossings and save
#    them as control_trials.npy in control_data_dir
uv run python experiment/src/control_fitting.py

# 2. start the RHX recording, then
uv run python experiment/src/wasabi_online.py
```
After the control fit, the channels to stimulate are printed and written to
`results/online/next_stim.json`. Deliver that stim on the next IR trial; once the trial
has been recorded press any key, and the next stim is printed (`q` quits). The history of
the session is saved to `results/online/history_<strategy>_<date>_<time>.npz` after
every trial.

- Spikes are detected from `amplifier.dat` as it is written: bandpass, local median
  reference, matched filter with the template, and a threshold on its noise SD from the
  first `noise_s` seconds of the recording, so keep those free of stimulation.
- Trials are taken in the order of the IR crossings in the recording, from its start.
  Break the beam only once the first stim is printed.
- `scripts/inspect_amplifier.ipynb` steps through the spike detection on a window of a
  recording, to check the settings.

### Other sources
```bash
uv run python experiment/src/wasabi_online.py --source sim      # dry run on simulated data
uv run python experiment/src/wasabi_online.py --source offline  # replay a recorded session (offline: in online.yaml)
```
`--strategy` (DOptimal, Tile, Random, Sequential), `--wasabi-data-dir`, `--control-data-dir`
and `--config` override the config.


## TODOs

- 

---

## Evaluation

## Contact
For questions, contact Bryan Tseng btseng2@jh.edu.

## Acknowledgements
Adam S. Charles, Sai Koukuntla.

## License
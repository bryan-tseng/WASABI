"""Run WASABI closed-loop during an experiment.

The script stays running. After the initial fit it prints the stim channels for the
first trial. The next trial is then read on a key press (session.trigger: keypress), on
its own once a trial on the IR input has been recorded after the stim was decided
(trigger: DIGITAL-IN-02, the trial.ir_channel), or straight away (auto: sim, offline). The
trial is read from a `TrialSource`, folded into the model with `observe_trial`, and the
channels for the next trial are printed and written to `paths.output` as JSON, in
results/online, offline or simulation by source.

Settings come from experiment/config/online.yaml (see config_loading.py). The dynamics are
fit on all recorded channels (recording.num_channels, 128); stim is on a subset of them.
By default 80 channels can be stimulated, in 20 contiguous groups of 4, and each candidate
switches on one group or two groups (20 choose 1 + 20 choose 2 = 210 candidates). Offline,
the candidates are the stims of the recorded session.

Keys (trigger: keypress): any key = trial done, q = quit. Ctrl-C stops the other triggers.

    # dry run against simulated data
    uv run python experiment/src/wasabi_online.py --source sim

    # replay a recorded session (offline section of the config): WASABI chooses among the
    # recorded stims and each choice takes the next unused trial recorded with that stim
    uv run python experiment/src/wasabi_online.py --source offline

    # real data straight from the Intan recording in progress, trials aligned at IR crossings
    uv run python experiment/src/wasabi_online.py --source online --wasabi-data-dir R --control-data-dir C

    # another config file
    uv run python experiment/src/wasabi_online.py --config my_session.yaml
"""

import argparse
import itertools
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from wasabi import (
    BPosterior,
    DOptimal,
    Random,
    TrialBuffer,
    TrialDesign,
    init_optimizer,
    make_poisson_lds,
    observe_trial,
    refit,
)
from wasabi.fitting import REFIT_FIELDS
from wasabi.selection import SelectionStrategy, expected_final_logdets

from config_loading import DEFAULT_CONFIG, ModelConfig, OnlineConfig, load_config


@dataclass(frozen=True)
class RoundRobin(SelectionStrategy):
    """Tile: step through the candidates in order. Unlike `wasabi.Tile` it works when the
    candidate set shrinks (offline, as recorded stims run out of trials)."""

    def select(self, key, model, prior, candidates, design, trial) -> int:
        return trial % len(candidates)


@dataclass(frozen=True)
class Sequential(SelectionStrategy):
    """The first available candidate. Offline this replays the session in recording order:
    every trial of the first condition, then the next, as the experiment was run."""

    def select(self, key, model, prior, candidates, design, trial) -> int:
        return 0


# Candidate set
def group_candidates(
    num_channels: int, group_size: int, max_groups: int
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Every combination of 1..`max_groups` contiguous channel groups.

    Returns (groups, patterns): groups[i] is the tuple of group indices of candidate i and
    patterns[i] its (num_channels,) binary stim pattern.
    """
    if num_channels % group_size:
        raise ValueError(f"{num_channels} channels do not split into groups of {group_size}")
    num_groups = num_channels // group_size
    groups = [
        combo
        for k in range(1, max_groups + 1)
        for combo in itertools.combinations(range(num_groups), k)
    ]
    patterns = np.zeros((len(groups), num_channels))
    for i, combo in enumerate(groups):
        for g in combo:
            patterns[i, g * group_size : (g + 1) * group_size] = 1.0
    return groups, patterns


def pattern_groups(pattern: np.ndarray, group_size: int) -> tuple[int, ...]:
    """Indices of the channel groups a stim pattern switches on."""
    return tuple(int(g) for g in np.flatnonzero(pattern.reshape(-1, group_size).any(axis=1)))


# Trial sources
class TrialSource(Protocol):
    """Where trial data comes from. Counts are (T, N), NaN where blanked."""

    def control_trials(self) -> np.ndarray:
        """(n_trials, T, N) unstimulated trials for the initial fit."""
        ...

    def next_trial(self, pattern: np.ndarray) -> np.ndarray | None:
        """Counts of the trial just delivered with `pattern`, or None if not available yet."""
        ...

    # Preprocessing of the data last returned that ran before the call, counted in its load time:
    # for control_trials, the time control_preproc.py took; for next_trial online, the spike
    # detection run in the background since the previous trial
    preprocess_s: float


class SimulatedSource:
    """Ground-truth Poisson LDS from the `simulations` package, for dry runs.

    The ground truth has `num_channels` recorded channels; stim patterns cover the first
    channels of them.
    """

    preprocess_s = 0.0  # trials are simulated when asked for

    def __init__(
        self, design: TrialDesign, num_channels: int, num_control: int, latents: int = 4, seed: int = 0
    ):
        from simulations import make_ground_truth_params, make_structured_B, simulate_trial

        self._simulate = simulate_trial
        self.rng = np.random.default_rng(seed)
        self.params = make_ground_truth_params(
            K=latents, N=num_channels, spectral_radius=0.9, sigma2_Q=0.2, b_input=-0.5,
            seed=seed, identity_V=True,
        )
        self.params["B"] = make_structured_B(latents, num_channels, self.rng, n_sites=10)
        self.design = design
        self.num_channels = num_channels
        self.num_control = num_control

    def _run(self, pattern):
        _, y, _ = self._simulate(
            self.params, self.design.num_timesteps, np.flatnonzero(pattern),
            self.design.t_stim, self.design.num_stim_bins, self.rng,
        )
        return y.astype(float)

    def control_trials(self) -> np.ndarray:
        return np.stack([self._run(np.zeros(self.num_channels)) for _ in range(self.num_control)])

    def next_trial(self, pattern: np.ndarray) -> np.ndarray | None:
        return self._run(pattern)

    def evaluator(self, candidates: np.ndarray) -> Callable[[object, BPosterior], dict]:
        """Per-trial error against the ground truth, as `effect_err` in joint_ekf_simulation.py.

        effect_err: relative error of the stim effect B u over the candidate set,
        ||U_fit B_fitᵀ - U_true B_refᵀ|| / ||U_true B_refᵀ||. B is compared through B u because
        b and the scale of B trade off, and in the fitted latent basis, B_ref = C_fitᵀ C_true B_true
        (C_fit has orthonormal columns, so C_fit x_fit = C_true x_true).
        """
        from simulations import effective_stim_input

        p = self.params
        # Stim patterns cover the first channels of the ground truth
        padded = np.zeros((len(candidates), p["B"].shape[1]))
        padded[:, : candidates.shape[1]] = candidates
        U_true = effective_stim_input(padded, p["V"], p["b"])
        C_true, B_true = np.asarray(p["C"]), np.asarray(p["B"])
        candidates = jnp.asarray(candidates)

        def evaluate(model, prior) -> dict:
            B_ref = np.asarray(model.observation.C).T @ C_true @ B_true
            U_fit = np.asarray(jax.vmap(model.stim_input)(candidates))
            true_effect = U_true @ B_ref.T
            pred_effect = U_fit @ np.asarray(prior.B).T
            return {"effect_err": float(np.linalg.norm(pred_effect - true_effect) / np.linalg.norm(true_effect))}

        return evaluate


class OfflineSource:
    """Replays a recorded stim session (`offline_data.RecordedSession`).

    The candidates are the session's stims, one per recorded condition. Choosing one takes
    the next unused trial recorded with it, in recording order; a stim whose trials are all
    used drops out of `available`.
    """

    preprocess_s = 0.0  # spikes were detected beforehand, in RCP_analysis

    def __init__(self, session, num_stim_channels: int):
        self.session = session
        self.candidates = np.zeros((len(session.conditions), num_stim_channels))
        for i, condition in enumerate(session.conditions):
            self.candidates[i, condition.channels] = 1.0
        self.used = np.zeros(len(session.conditions), dtype=int)
        self.num_trials = np.array([len(c.trials) for c in session.conditions])

    def control_trials(self) -> np.ndarray:
        return self.session.controls

    def available(self) -> np.ndarray:
        return self.used < self.num_trials

    def next_trial(self, pattern: np.ndarray) -> np.ndarray | None:
        i = int(np.flatnonzero((self.candidates == pattern).all(axis=1))[0])
        if self.used[i] >= self.num_trials[i]:
            return None
        condition = self.session.conditions[i]
        trial = condition.trials[self.used[i]]
        print(
            f"  recorded trial {self.used[i] + 1}/{self.num_trials[i]} of Intan file "
            f"{condition.intan_idx}, IR at {condition.onsets_ms[self.used[i]] / 1e3:.1f} s"
        )
        self.used[i] += 1
        return trial


def load_control_trials(path: Path, num_channels: int) -> tuple[np.ndarray, float]:
    """(n_trials, T, N) control trials from a .npy, or from the control_trials.npy in a recording folder."""
    from intan_rt_proc import TRIALS_FILE

    if path.is_dir():
        if not (path / TRIALS_FILE).exists():
            raise FileNotFoundError(
                f"no {TRIALS_FILE} in {path}; make it with\n"
                f'    uv run python experiment/src/control_preproc.py "{path}"'
            )
        path = path / TRIALS_FILE
    Y = np.load(path).astype(float)
    if Y.ndim != 3 or Y.shape[2] != num_channels:
        raise ValueError(f"control trials must be (n_trials, T, {num_channels}), got {Y.shape}")
    info = path.with_suffix(".json")  # written by control_preproc.py
    if not info.exists():
        print(f"  no {info.name} beside {path.name}: control load time leaves out the preprocessing")
        return Y, 0.0
    return Y, json.loads(info.read_text())["preprocess_s"]


class IntanSource:
    """Reads trials from the Intan RHX recording in progress, cut at IR crossings.

    Spikes are detected from amplifier.dat in a background thread as it is written (see
    `intan_rt_proc.IntanStream`). Until the first stim is decided only the first noise_s
    seconds are processed, for the thresholds; `start_trials` then jumps to just before the
    present, so the trials start there. On each key press the thread is left to process
    what is on disk, and the next complete trial not yet used is taken.

    With an IR trigger, `arm` is called once the stim is decided, and only IR crossings
    written after it count: earlier ones were delivered with the previous stim, or none.
    The background thread keeps up on its own, so nothing is waited for.
    """

    WARMUP_S = 0.2  # processed before the earliest trial window after the jump, for the filter to settle

    def __init__(
        self, recording_dir: Path, control_file: Path, num_channels: int, ir_trigger: bool = False,
        **stream_kwargs,
    ):
        from intan_rt_proc import IntanStream

        self.control_file = Path(control_file)
        self.num_channels = num_channels
        self.ir_trigger = ir_trigger
        self.stream = IntanStream(recording_dir, **stream_kwargs)
        if len(self.stream.channels) != num_channels:
            raise ValueError(
                f"{recording_dir} has {len(self.stream.channels)} amplifier channels, "
                f"recording.num_channels is {num_channels}"
            )
        # Only the noise estimate while the control fit runs; trials start at `start_trials`
        self.stream.hold()
        self.stream.start() # Creates background thread
        self.first_onset = 0  # earliest IR crossing sample that can be the next trial
        self.preprocess_s = 0.0
        self._process_s_counted = 0.0  # stream.process_s already counted in a trial's load time

    def control_trials(self) -> np.ndarray:
        Y, self.preprocess_s = load_control_trials(self.control_file, self.num_channels)
        return Y

    # First stim decided: skip to just before now, as no earlier trial can be used
    def start_trials(self) -> None:
        now = self.stream.num_written()
        margin = round(self.WARMUP_S * self.stream.sample_rate) - self.stream.offset_samples  # 0.2 + 0.8 s
        self.stream.resume(now - margin)
        self.first_onset = max(self.first_onset, now)

    # IR trigger: the stim is decided now, so the next trial is the first IR crossing after this
    def arm(self) -> None:
        self.first_onset = max(self.first_onset, self.stream.num_written())

    def next_trial(self, pattern: np.ndarray) -> np.ndarray | None:
        self.stream.check()
        background_s = self.stream.process_s - self._process_s_counted
        if not self.ir_trigger:
            self.stream.catch_up()  # wait for the thread to process what is on disk at the key press
        onsets = self.stream.onsets()  # IR crossings whose trial is complete
        onsets = onsets[onsets >= self.first_onset]
        if not len(onsets):
            return None
        # The catch-up above is timed by the caller, so count only what ran in the background
        self.preprocess_s = background_s
        self._process_s_counted = self.stream.process_s
        onset = onsets[0]
        if len(onsets) > 1:
            print(
                f"  warning: {len(onsets)} new IR trials since the last one used; "
                f"using the earliest, at {self.stream.onset_time(onset):.3f} s"
            )
        else:
            print(f"  IR trial at {self.stream.onset_time(onset):.3f} s")
        self.first_onset = onset + 1
        return self.stream.trial(onset)  # (T, N) spike counts


# Control fit
def fit_until_converged(
    model,
    buffer: TrialBuffer,
    tol: float = 1e-5,
    max_steps: int = 5000,
    window: int = 100,
    chunk: int = 50,
    learning_rate: float = 5e-3,
) -> tuple[object, np.ndarray, bool]:
    """`fit_initial` until the loss stops improving, instead of for a fixed number of steps.

    Runs `refit` in chunks of `chunk` steps, carrying the optimizer state, and stops once the
    loss has improved by less than `tol` (relative) over the last `window` steps, or after
    `max_steps`. The control loss is full-batch and deterministic, so it decreases smoothly.

    Returns (model, losses per step, whether it converged before `max_steps`).
    """
    optimizer, opt_state = init_optimizer(model, learning_rate)
    B = jnp.zeros((model.state_dim, model.num_inputs))  # stim input is zero on control trials
    losses = []
    while len(losses) < max_steps:
        model, opt_state, chunk_losses = refit(
            model, opt_state, B, buffer, optimizer, min(chunk, max_steps - len(losses))
        )
        losses.extend(np.asarray(chunk_losses).tolist())
        if len(losses) > window:
            before, now = losses[-window - 1], losses[-1]
            if (before - now) / abs(now) < tol:
                return model, np.array(losses), True
    return model, np.array(losses), False


# Online session
@dataclass
class OnlineSession:
    """Model state carried between trials, plus the stim choice for the upcoming trial."""

    model: object
    prior: BPosterior
    buffer: TrialBuffer
    optimizer: object
    opt_state: object
    design: TrialDesign
    groups: list[tuple[int, ...]]
    candidates: np.ndarray
    strategy: SelectionStrategy
    key: jnp.ndarray
    refit_steps: int = 20
    refit_fields: tuple[str, ...] = REFIT_FIELDS  # parameters refit after every trial
    available: Callable[[], np.ndarray] | None = None  # which candidates can still be chosen
    # evaluate(model, prior) -> dict of per-trial diagnostics added to the history, e.g. the
    # error against the ground truth in simulation
    evaluate: Callable[[object, BPosterior], dict] | None = None
    trial: int = 0
    choice: int = -1
    # Per trial: choice, logdet, refit_loss, load_s (preprocessing and reading the trial),
    # wait_s (the part of load_s after the key press, or in the poll that found the trial), fitting_s (filter
    # and refit), and the fitted B mean (K, S), C (N, K) and b after it (B_trial, C_trial,
    # b_trial). Per stim decision, one more than trials (entry 0 is the first stim):
    # acquisition_s (scoring the candidates) and decided_at (Unix time), and for DOptimal the
    # decision variable: scores (n_candidates,) and score_prior_logdet, the prior's log det
    history: dict = field(default_factory=lambda: {
        "choice": [], "logdet": [], "refit_loss": [], "load_s": [], "wait_s": [], "fitting_s": [],
        "acquisition_s": [], "decided_at": [], "B_trial": [], "C_trial": [], "b_trial": [],
    })
    fit_losses: np.ndarray | None = None  # control fit loss per step
    fit_converged: bool = False
    control_load_s: float = float("nan")  # preprocessing and reading the control trials
    control_fit_s: float = float("nan")

    @classmethod
    def start(
        cls,
        Y_control: np.ndarray,
        design: TrialDesign,
        groups: list[tuple[int, ...]],
        candidates: np.ndarray,
        strategy: SelectionStrategy,
        model_cfg: ModelConfig,
        max_trials: int,
        seed: int = 0,
        available: Callable[[], np.ndarray] | None = None,
        evaluate: Callable[[object, BPosterior], dict] | None = None,
    ) -> "OnlineSession":
        """Fit the dynamics on control trials (nothing blanked, no stim) and pick the first stim.

        `model_cfg` (the config's 'model' section) sets the latents, B prior, and control
        fit and per-trial refit settings. The model records the N channels of `Y_control`
        and is stimulated through the S channels of `candidates`. `available`, if given,
        returns which candidates can still be chosen. `evaluate`, if given, adds its
        diagnostics to the history after every trial.
        """
        num_control, T, N = Y_control.shape
        S = candidates.shape[1]
        if T != design.num_timesteps:
            raise ValueError(f"control trials have {T} bins, the trial design has {design.num_timesteps}")

        t0 = time.perf_counter()
        control = (jnp.asarray(Y_control), jnp.zeros((num_control, T, S)), jnp.ones(T, dtype=bool))
        model = make_poisson_lds(
            model_cfg.latents, jnp.eye(S), jr.key(seed),
            mean_rate=jnp.asarray(np.nanmean(Y_control, axis=(0, 1))), num_channels=N,
        )
        # The loss is computed over every buffer slot, so fit on the control trials alone
        # rather than in the buffer sized for the whole session
        model, losses, converged = fit_until_converged(
            model, TrialBuffer.empty(num_control, T, N, S).extend(*control),
            model_cfg.initial_fit_tol, model_cfg.initial_fit_max_steps,
        )
        control_fit_s = time.perf_counter() - t0  # losses come back as numpy, so the fit has finished
        buffer = TrialBuffer.empty(num_control + max_trials, T, N, S).extend(*control)

        # Per-trial refits cover model_cfg.refit_fields (without b when freeze_b)
        optimizer, opt_state = init_optimizer(model, fields=model_cfg.refit_fields)
        session = cls(
            model, BPosterior.from_prior(model_cfg.latents, S, model_cfg.prior_var), buffer,
            optimizer, opt_state, design, groups, candidates, strategy, jr.key(seed + 1),
            model_cfg.refit_steps, model_cfg.refit_fields, available, evaluate,
        )
        session.fit_losses, session.fit_converged, session.control_fit_s = losses, converged, control_fit_s
        session.propose()
        return session

    def propose(self) -> int:
        """Pick the stim for the upcoming trial; -1 if no candidate is available."""
        t0 = time.perf_counter()
        self.key, subkey = jr.split(self.key)
        options = np.arange(len(self.candidates)) if self.available is None else np.flatnonzero(self.available())
        if len(options) == 0:
            self.choice = -1
            return self.choice
        if isinstance(self.strategy, DOptimal):
            # Same choice as DOptimal.select, keeping the decision variable: each candidate's
            # expected end-of-trial log det of cov(vec(B)), NaN for candidates not available
            scores = np.asarray(expected_final_logdets(
                self.model, self.prior, self.candidates[options], self.design
            ))
            pick = int(np.argmin(scores))
            full = np.full(len(self.candidates), np.nan)
            full[options] = scores
            self.history.setdefault("scores", []).append(full)
            self.history.setdefault("score_prior_logdet", []).append(float(self.prior.logdet()))
        else:
            pick = self.strategy.select(
                subkey, self.model, self.prior, self.candidates[options], self.design, self.trial
            )
        self.choice = int(options[pick])
        self.history["acquisition_s"].append(time.perf_counter() - t0)  # np.asarray / select wait for the result
        self.history["decided_at"].append(time.time())
        return self.choice

    def update(self, ys: np.ndarray, load_s: float = float("nan"), wait_s: float = float("nan")) -> int:
        """Fold in the trial just delivered with the current choice, then pick the next stim."""
        ys = np.array(ys, dtype=float)
        if ys.shape != (self.design.num_timesteps, self.model.num_channels):
            raise ValueError(
                f"trial counts must be ({self.design.num_timesteps}, {self.model.num_channels}), "
                f"got {ys.shape}"
            )
        t0 = time.perf_counter()
        ys[~np.asarray(self.design.keep)] = np.nan  # blank the stim artifact bins
        ss = self.design.schedule(self.candidates[self.choice])
        self.model, self.prior, self.buffer, self.opt_state, losses = observe_trial(
            self.model, self.prior, self.buffer, self.optimizer, self.opt_state,
            jnp.asarray(ys), ss, self.design.keep, self.refit_steps, self.refit_fields,
        )
        self.history["choice"].append(self.choice)
        self.history["logdet"].append(float(self.prior.logdet()))
        self.history["refit_loss"].append(float(losses[-1]) if losses.size else float("nan"))
        # float() above waits for the results, so this times the whole filter and refit
        self.history["fitting_s"].append(time.perf_counter() - t0)
        self.history["load_s"].append(load_s)
        self.history["wait_s"].append(wait_s)
        # Fitted stim effect after the trial, so B can be compared across trials (e.g. with the
        # final B offline): C B u is the same in any latent basis, u = u(s; b) of this trial
        self.history["B_trial"].append(np.asarray(self.prior.B))
        self.history["C_trial"].append(np.asarray(self.model.observation.C))
        self.history["b_trial"].append(float(self.model.dynamics.b))
        if self.evaluate is not None:
            for name, value in self.evaluate(self.model, self.prior).items():
                self.history.setdefault(name, []).append(value)
        self.trial += 1
        return self.propose()

    def selection(self, channel_base: int = 1) -> dict:
        """The upcoming stim as group indices and channel numbers."""
        channels = np.flatnonzero(self.candidates[self.choice]) + channel_base
        return {
            "trial": self.trial,
            "candidate": int(self.choice),
            "groups": [int(g) for g in self.groups[self.choice]],
            "channels": [int(c) for c in channels],
            "decided_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.history["decided_at"][-1])),
        }

    def save_history(self, path: Path) -> None:
        np.savez(
            path,
            **{name: np.array(v) for name, v in self.history.items()},
            candidates=self.candidates,
            B_mean=np.asarray(self.prior.B),
            control_load_s=self.control_load_s,
            control_fit_s=self.control_fit_s,
            control_fit_losses=self.fit_losses,
            control_fit_converged=self.fit_converged,
        )


# Key press
def wait_for_key() -> str:
    """Block until one key is pressed. Falls back to a line of input when stdin is not a terminal."""
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        return line[:1] if line else "q"
    if os.name == "nt":
        import msvcrt

        return msvcrt.getwch()
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def write_selection(selection: dict, path: Path) -> None:
    """Atomic write, so a reader on the rig never sees a half-written file.

    The temporary file is named per process, so two sessions writing the same output
    cannot rename each other's temporary file away.
    """
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(selection))
    tmp.replace(path)


# Terminal status
TTY = sys.stdout.isatty()


def _style(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if TTY else text


class Running:
    """Shows "WASABI running: <task>" with a spinner and elapsed time while the block runs.

    The spinner is redrawn from a background thread (JAX releases the GIL while it
    computes). Without a terminal, e.g. when logging to a file, the task is printed once.
    """

    FRAMES = "|/-\\"

    def __init__(self, task: str):
        self.task = task
        self._stop = threading.Event()

    def _line(self, mark: str) -> str:
        return f"{mark} WASABI running: {self.task}  {time.perf_counter() - self.t0:5.1f} s"

    def _spin(self):
        i = 0
        while not self._stop.wait(0.1):
            sys.stdout.write("\r" + _style(self._line(self.FRAMES[i % len(self.FRAMES)]), "33"))
            sys.stdout.flush()
            i += 1

    def __enter__(self):
        self.t0 = time.perf_counter()
        if TTY:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        else:
            print(f"WASABI running: {self.task}...", flush=True)
        return self

    def __exit__(self, exc_type, *_):
        if TTY:
            self._stop.set()
            self._thread.join()
            sys.stdout.write("\r\033[K")  # clear the spinner line
        self.elapsed = time.perf_counter() - self.t0
        if exc_type is None:
            print(f"  done: {self.task} ({self.elapsed:.1f} s)", flush=True)
        return False


def announce(selection: dict, path: Path, trigger: str = "keypress") -> None:
    write_selection(selection, path)
    groups = ", ".join(str(g) for g in selection["groups"])
    bar = "=" * 72
    print(_style(
        f"\n{bar}\n"
        f"  NEXT STIM DECIDED  -  trial {selection['trial']}  ({time.strftime('%H:%M:%S')})\n"
        f"  groups [{groups}] -> channels {selection['channels']}\n"
        f"{bar}",
        "1;32",
    ))
    if trigger == "keypress":
        print("  Waiting: deliver this stim, then press any key when the trial is done (q to quit)", flush=True)
    elif trigger != "auto":
        print(f"  Waiting for the next trial on {trigger}: deliver this stim (Ctrl-C to stop)", flush=True)
    else:
        print("  Starting the next trial (Ctrl-C to stop)", flush=True)


# CLI
HISTORY_NAMES = {"DOptimal": "doptimal", "Tile": "tiled", "Random": "random", "Sequential": "sequential"}  # in history file names


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    # Per-session overrides of the config
    parser.add_argument("--source", choices=["sim", "offline", "online"])
    parser.add_argument("--strategy", choices=["DOptimal", "Tile", "Random", "Sequential"])
    parser.add_argument("--wasabi-data-dir", type=Path, help="Intan RHX recording folder of the session (online)")
    parser.add_argument(
        "--control-data-dir", type=Path,
        help="control trials: a recording folder with control_trials.npy (control_preproc.py), or an (n_trials, T, N) .npy",
    )
    parser.add_argument("--history", type=Path)
    return parser.parse_args()


def make_design(
    num_bins: int, offset_ms: float, stim_ms: float, blank_after_ms: float, bin_ms: float
) -> TrialDesign:
    """Stim starts at the IR crossing, which is -`offset_ms` into the trial window.

    Every bin the stim touches counts as stimulated; every bin from stim onset to
    `blank_after_ms` after stim ends is blanked.
    """
    onset_ms = -offset_ms
    stim_bin = int(np.floor(onset_ms / bin_ms))
    stim_bins = int(np.ceil((onset_ms + stim_ms) / bin_ms)) - stim_bin
    # Artifact removal included
    blank_end = int(np.ceil((onset_ms + stim_ms + blank_after_ms) / bin_ms))
    if not 0 <= stim_bin < num_bins or blank_end > num_bins or stim_bins < 1:
        raise ValueError(
            f"stim at {onset_ms:g} ms for {stim_ms:g} ms, blanked to {blank_end * bin_ms:g} ms, "
            f"does not fit in a {num_bins * bin_ms:g} ms trial"
        )
    keep = jnp.ones(num_bins, dtype=bool).at[stim_bin:blank_end].set(False)
    return TrialDesign(keep, stim_bin, stim_bins)


def make_source(cfg: OnlineConfig, design: TrialDesign, recorded=None) -> TrialSource:
    """The trial source for `cfg.session.source`.

    `recorded` is the `offline_data.RecordedSession` already loaded for source offline.
    Raises SystemExit if a path the source needs is missing or the source is unknown.
    """
    source, n = cfg.session.source, cfg.recording.num_channels
    control_file = cfg.data_path(cfg.paths.control_data_dir)
    if source == "sim":
        return SimulatedSource(design, n, cfg.session.num_control, cfg.model.latents, cfg.session.seed)
    if source == "offline":
        return OfflineSource(recorded, round_up(recorded.num_stim_channels, cfg.stim.group_size))
    if source == "online":
        if cfg.paths.wasabi_data_dir is None or control_file is None:
            raise SystemExit("source online needs paths.wasabi_data_dir and paths.control_data_dir")
        return IntanSource(
            cfg.data_path(cfg.paths.wasabi_data_dir), control_file, n,
            ir_trigger=cfg.session.trigger not in ("keypress", "auto"), **cfg.intan_stream_kwargs(),
        )
    raise SystemExit(f"unknown source {source!r}: sim, offline or online")


def round_up(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


def main():
    args = parse_args()
    cfg = load_config(args.config)
    # JAX device, set before the first JAX computation picks one
    if cfg.model.device not in ("gpu", "cpu"):
        raise SystemExit(f"model.device must be gpu or cpu, got {cfg.model.device!r}")
    if cfg.model.device == "cpu":
        jax.config.update("jax_platforms", "cpu")
    print(f"JAX fitting on {jax.devices()[0]}")
    # Command-line paths are used as given; config paths are relative to the data root
    if args.source is not None:
        cfg.session.source = args.source
    if args.strategy is not None:
        cfg.session.strategy = args.strategy
    for name in ("wasabi_data_dir", "control_data_dir", "history"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg.paths, name, str(value.resolve()))
    trial, stim, session = cfg.trial, cfg.stim, cfg.session

    # What reads the next trial; an IR trigger takes the first IR crossing after the stim is decided
    trigger = session.trigger
    if trigger not in ("keypress", "auto") and session.source != "online":
        print(f"session.trigger {trigger}: source {session.source} has no IR input, so trials run straight away (auto)")
        trigger = "auto"
    keypress = trigger == "keypress"
    ir_trigger = trigger not in ("keypress", "auto")
    if ir_trigger and trigger != trial.ir_channel:
        raise SystemExit(
            f"session.trigger {trigger!r}: keypress, auto, or the IR input trial.ir_channel ({trial.ir_channel})"
        )

    # Offline, the recorded session sets the candidates and the stim duration
    recorded = None
    if session.source == "offline":
        from offline_data import load_recorded_session

        off = cfg.offline
        with Running(f"loading recorded session {Path(off.session_dir).name}"):
            recorded = load_recorded_session(
                cfg.data_path(off.session_dir), off.intan_files, off.control_files, trial.window_ms,
                trial.bin_ms, cfg.recording.num_channels,
            )
        stim.stim_ms = recorded.stim_dur_ms
        print(
            f"  {sum(len(c.trials) for c in recorded.conditions)} stim trials in "
            f"{len(recorded.conditions)} conditions, {len(recorded.controls)} control trials; "
            f"stim duration {stim.stim_ms:.0f} ms (from the recording)"
        )

    # Loading trial design (time bins and when to blank etc.)
    design = make_design(trial.num_bins, trial.offset_ms, stim.stim_ms, stim.blank_after_ms, trial.bin_ms)
    # For example: 70 bins for length = 1400ms, 20ms bins, (70, -800.0, 80.0, 20.0, 20.0)
    blanked = np.flatnonzero(~np.asarray(design.keep))
    print(
        f"Trial design: stim in bins {design.t_stim}-{design.t_stim + design.num_stim_bins - 1}, "
        f"blanked bins {blanked[0]}-{blanked[-1]} "
        f"({blanked[0] * trial.bin_ms + trial.offset_ms:g} to "
        f"{(blanked[-1] + 1) * trial.bin_ms + trial.offset_ms:g} ms from the IR crossing)"
    )

    # Create data source based on config and trial design
    source = make_source(cfg, design, recorded)

    # Candidate stims: the recorded ones offline, otherwise every set of 1..max_groups groups
    available = None
    max_trials = session.max_trials
    if session.source == "offline":
        candidates = source.candidates
        groups = [pattern_groups(c, stim.group_size) for c in candidates]
        available = source.available
        max_trials = min(max_trials, int(source.num_trials.sum()))
        print(f"{len(candidates)} candidates: the recorded stims on {candidates.shape[1]} channels")
    else:
        groups, candidates = group_candidates(stim.num_channels, stim.group_size, stim.max_groups)
        print(
            f"{len(groups)} candidates: {stim.num_channels} channels in groups of {stim.group_size}, "
            f"up to {stim.max_groups} groups"
        )
    strategies = {"DOptimal": DOptimal, "Tile": RoundRobin, "Random": Random, "Sequential": Sequential}
    if session.strategy not in strategies:
        raise SystemExit(f"unknown strategy {session.strategy!r}: {', '.join(strategies)}")
    strategy = strategies[session.strategy]()

    # Results go in results/online, offline or simulation, by source
    output = cfg.results_path(cfg.paths.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history_path = cfg.results_path(
        cfg.paths.history
        or f"history_{HISTORY_NAMES[session.strategy]}_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # Fitting control trials
    t0 = time.perf_counter()
    Y_control = source.control_trials()
    control_load_s = source.preprocess_s + time.perf_counter() - t0  # preprocessing, then reading
    if Y_control.shape[2] != cfg.recording.num_channels:
        raise SystemExit(
            f"control trials have {Y_control.shape[2]} channels, recording.num_channels is "
            f"{cfg.recording.num_channels}"
        )
    with Running(f"fitting dynamics on {len(Y_control)} control trials, choosing the first stim"):
        online = OnlineSession.start(
            Y_control, design, groups, candidates, strategy, cfg.model, max_trials,
            session.seed, available,
            source.evaluator(candidates) if isinstance(source, SimulatedSource) else None,
        )
    online.control_load_s = control_load_s
    losses = online.fit_losses
    status = "converged" if online.fit_converged else "hit the step limit before converging"
    print(
        f"  control fit NLL per bin {losses[0]:.4f} -> {losses[-1]:.4f} in {len(losses)} steps ({status}) | "
        f"load {online.control_load_s:.2f} s, fitting {online.control_fit_s:.2f} s, "
        f"acquisition {online.history['acquisition_s'][0]:.2f} s"
    )

    # Announce initial selection of stimuli
    announce(online.selection(stim.channel_base), output, trigger)
    if isinstance(source, IntanSource):
        source.start_trials()  # spike detection resumes just before now
    if ir_trigger:
        source.arm()

    try:
        while online.trial < max_trials:
            # Quit if user presses 'q'
            if keypress and wait_for_key().lower() == "q":
                break
            t0 = time.perf_counter()
            ys = source.next_trial(online.candidates[online.choice])
            wait_s = time.perf_counter() - t0
            if ys is None:
                if keypress:
                    print("  no new trial data yet; press a key again once it is written", flush=True)
                else:
                    time.sleep(0.05 if ir_trigger else 0.2)  # poll until the trial has been recorded
                continue
            with Running(f"updating the model with trial {online.trial}, choosing the next stim"):
                online.update(ys, source.preprocess_s + wait_s, wait_s)
                online.save_history(history_path)
            h = online.history
            err = f", effect err {h['effect_err'][-1]:.3f}" if "effect_err" in h else ""
            print(
                f"  logdet {h['logdet'][-1]:.1f}, NLL {h['refit_loss'][-1]:.4f}{err} | "
                f"load {h['load_s'][-1]:.2f} s ({h['wait_s'][-1]:.2f} s {'after the key press' if keypress else 'catching up'}), "
                f"fitting {h['fitting_s'][-1]:.2f} s, "
                f"acquisition {h['acquisition_s'][-1]:.2f} s"
            )
            if online.choice < 0:
                print("\nEvery recorded trial has been used.")
                break
            announce(online.selection(stim.channel_base), output, trigger)
            if ir_trigger:
                source.arm()
    except KeyboardInterrupt:
        pass
    finally:
        online.save_history(history_path)
        print(f"\nStopped after {online.trial} trials; history saved to {history_path}")


if __name__ == "__main__":
    main()

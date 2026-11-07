"""Load the online experiment config, experiment/config/online.yaml by default.

Data paths (recording_dir, control_file, offline.session_dir) are relative to `paths.data_root`,
which is prefixed with this machine's data mount from machines.yaml, as in RCP_analysis.
Output files (output, history) go in the results folder of the session's source,
<results_dir>/online, offline or simulation, with results_dir relative to the WASABI repo.
Absolute paths are used as given. Unknown keys raise, so a typo in the YAML does not silently fall back to a default.
"""

import socket
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "experiment" / "config"
DEFAULT_CONFIG = CONFIG_DIR / "online.yaml"
MACHINES = CONFIG_DIR / "machines.yaml"

# Results subfolder of each session source
RESULTS_FOLDERS = {"online": "online", "offline": "offline", "sim": "simulation"}


@dataclass
class PathsConfig:
    data_root: str = ""
    recording_dir: str | None = None
    control_file: str | None = None
    results_dir: str = "results"
    output: str = "next_stim.json"
    history: str | None = None


@dataclass
class SessionConfig:
    source: str = "online"
    strategy: str = "DOptimal"
    max_trials: int = 500
    seed: int = 0
    num_control: int = 100
    wait_for_key: bool = True


@dataclass
class StimConfig:
    num_channels: int = 80
    group_size: int = 4
    max_groups: int = 2
    channel_base: int = 1
    stim_ms: float = 80.0
    blank_after_ms: float = 0.0


@dataclass
class RecordingConfig:
    num_channels: int = 128


@dataclass
class OfflineConfig:
    session_dir: str = "Nike/20251203_NRR_RW011"
    intan_files: list[int] = field(default_factory=lambda: list(range(12, 22)))
    control_files: list[int] = field(default_factory=lambda: [1, 3])


@dataclass
class TrialConfig:
    bin_ms: float = 20.0
    window_ms: tuple[float, float] = (-800.0, 600.0)
    ir_channel: str = "DIGITAL-IN-01"
    ir_edge: str = "falling"
    ir_threshold: float = 1.0
    keep_artifacts: bool = False

    @property
    def offset_ms(self) -> float:
        """Start of the trial window relative to the IR crossing."""
        return float(self.window_ms[0])

    @property
    def num_bins(self) -> int:
        length = self.window_ms[1] - self.window_ms[0]
        if length <= 0 or length % self.bin_ms:
            raise ValueError(f"trial window {self.window_ms} ms is not a whole number of {self.bin_ms:g} ms bins")
        return int(length // self.bin_ms)

    def load_trials_kwargs(self) -> dict:
        """Keyword arguments of `intan_binning.load_trials`."""
        return dict(
            ir_channel=self.ir_channel, edge=self.ir_edge, trial_ms=self.num_bins * self.bin_ms,
            offset_ms=self.offset_ms, bin_ms=self.bin_ms, analog_threshold=self.ir_threshold,
            no_artifacts=not self.keep_artifacts,
        )


@dataclass
class ModelConfig:
    latents: int = 4
    prior_var: float = 1.0
    refit_steps: int = 20
    initial_fit_tol: float = 1e-5
    initial_fit_max_steps: int = 5000
    # Keep the stim-input offset b at its control-fit value (control trials have no stim, so
    # that is its initial value). Refitting b rescales u(s) without rescaling the B posterior.
    freeze_b: bool = True

    @property
    def refit_fields(self) -> tuple[str, ...]:
        from wasabi.fitting import REFIT_FIELDS

        return tuple(f for f in REFIT_FIELDS if not (self.freeze_b and f == "dynamics.b"))


@dataclass
class OnlineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    stim: StimConfig = field(default_factory=StimConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    trial: TrialConfig = field(default_factory=TrialConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def data_path(self, path: str | Path | None) -> Path | None:
        """`path` under this machine's data root, unless it is absolute."""
        if path is None:
            return None
        path = Path(path)
        return path if path.is_absolute() else resolve_data_root(self.paths.data_root) / path

    @property
    def results_folder(self) -> Path:
        """<results_dir>/online, offline or simulation, by the session's source."""
        source = self.session.source
        if source not in RESULTS_FOLDERS:
            raise ValueError(f"unknown source {source!r}: {', '.join(RESULTS_FOLDERS)}")
        results_dir = Path(self.paths.results_dir)
        results_dir = results_dir if results_dir.is_absolute() else ROOT / results_dir
        return results_dir / RESULTS_FOLDERS[source]

    def results_path(self, path: str | Path) -> Path:
        """`path` in the source's results folder, unless it is absolute."""
        path = Path(path)
        return path if path.is_absolute() else self.results_folder / path


def resolve_data_root(data_root: str, machines_path: Path = MACHINES) -> Path:
    """Prepend this machine's data mount (by hostname) to a relative data root."""
    if Path(data_root).is_absolute():
        return Path(data_root)
    machines = (yaml.safe_load(machines_path.read_text()) or {}).get("machines") or {}
    hostname = socket.gethostname()
    if hostname not in machines:
        raise KeyError(f"no entry for this machine ({hostname}) in {machines_path}")
    return Path(machines[hostname]["data_root_prefix"]) / data_root


def _section(cls, raw: dict | None, name: str):
    raw = raw or {}
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown keys in '{name}': {sorted(unknown)} (expected some of {sorted(known)})")
    # PyYAML reads 1e-5 (no decimal point) as a string, so cast numeric fields
    types = {f.name: f.type for f in fields(cls)}
    return cls(**{
        key: types[key](value) if types[key] in (int, float) and value is not None else value
        for key, value in raw.items()
    })


def load_config(path: str | Path = DEFAULT_CONFIG) -> OnlineConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    sections = {f.name: f.default_factory for f in fields(OnlineConfig)}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"unknown sections in {path}: {sorted(unknown)}")
    cfg = OnlineConfig(**{
        name: _section(factory().__class__, raw.get(name), name) for name, factory in sections.items()
    })
    cfg.trial.window_ms = tuple(float(t) for t in cfg.trial.window_ms)
    return cfg

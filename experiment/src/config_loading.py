"""Load the online experiment config, default: experiment/config/online.yaml.

The machine.yaml prefixes and `paths.data_root` are prepended to data_dir paths,
such as wasabi_data_dir, control_data_dir, and offline.session_dir.

Output files (output, history) go in the results folder of the session's source, <results_dir>/online, offline or simulation
<results_dir> is relative to the WASABI repo
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
    wasabi_data_dir: str | None = None  # the RHX recording of the wasabi session, including .dat files
    control_data_dir: str | None = None  # control trials: a folder with control_trials.npy
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
    # Indicator for the next trial:
    #   Online: "keypress" or an IR input such as "DIGITAL-IN-02"
    #   Offline/Sim: "auto" (straight away)
    trigger: str = "keypress"


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
    sample_rate: float = 30000.0


@dataclass
class SpikeConfig:
    """Spike detection on amplifier.dat (see `intan_rt_proc.SpikeDetector`)."""

    band_hz: tuple[float, float] = (300.0, 10000.0)
    filter_order: int = 3
    reference: str = "median"
    threshold: float = 4.0
    refractory_ms: float = 1.0
    noise_s: float = 10.0
    local_radius: tuple[float, float] | None = None  # (inner, outer) um; None: global reference
    positions_file: str | None = None  # Kilosort chanMap .mat or CSV channel,x,y; relative to experiment/config
    template_file: str | None = None  # 1-D spike waveform .npy for the matched filter; None: threshold the signal

    def config_path(self, path: str | None) -> Path | None:
        """`path` under experiment/config, unless it is absolute."""
        if path is None:
            return None
        return Path(path) if Path(path).is_absolute() else CONFIG_DIR / path


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

    def stream_kwargs(self) -> dict:
        """Trial and IR keyword arguments of `intan_rt_proc.IntanStream`."""
        return dict(
            ir_channel=self.ir_channel, edge=self.ir_edge, trial_ms=self.num_bins * self.bin_ms,
            offset_ms=self.offset_ms, bin_ms=self.bin_ms, analog_threshold=self.ir_threshold,
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
    device: str = "gpu"  # where JAX fits: gpu (the default device, CPU if there is no GPU) | cpu

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
    spikes: SpikeConfig = field(default_factory=SpikeConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def intan_stream_kwargs(self) -> dict:
        """Keyword arguments of `intan_rt_proc.IntanStream`: sample rate, trial, IR and spike detection."""
        return dict(
            sample_rate=self.recording.sample_rate, **self.trial.stream_kwargs(),
            **{f.name: getattr(self.spikes, f.name) for f in fields(SpikeConfig)}
            | {
                "positions_file": self.spikes.config_path(self.spikes.positions_file),
                "template_file": self.spikes.config_path(self.spikes.template_file),
            },
        )

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


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a key given twice in one mapping (PyYAML keeps the last silently)."""

    def construct_mapping(self, node, deep=False):
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        repeated = sorted({k for k in keys if keys.count(k) > 1})
        if repeated:
            raise ValueError(f"keys given more than once {node.start_mark}: {repeated}")
        return super().construct_mapping(node, deep=deep)


def load_config(path: str | Path = DEFAULT_CONFIG) -> OnlineConfig:
    raw = yaml.load(Path(path).read_text(), Loader=_UniqueKeyLoader) or {}
    sections = {f.name: f.default_factory for f in fields(OnlineConfig)}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"unknown sections in {path}: {sorted(unknown)}")
    cfg = OnlineConfig(**{
        name: _section(factory().__class__, raw.get(name), name) for name, factory in sections.items()
    })
    cfg.trial.window_ms = tuple(float(t) for t in cfg.trial.window_ms)
    cfg.spikes.band_hz = tuple(float(f) for f in cfg.spikes.band_hz)
    if cfg.spikes.local_radius is not None:
        cfg.spikes.local_radius = tuple(float(r) for r in cfg.spikes.local_radius)
    return cfg

"""Bin Intan RHX spike data into counts for `wasabi_online`.

Expects a recording saved as "one file per signal type": spike.dat (read with
`ReadIntanSpikeFile.readIntanSpikeFile`), time.dat, the int32 sample timestamps of
the whole recording, and for trials the IR beam signal in digitalin.dat or analogin.dat.

Trials: DIGITAL-IN-01 is high while the IR beam is intact and drops to 0 when it is
crossed, so each falling edge marks a trial. The trial window runs from -800 ms to +600 ms
around the crossing by default: it starts `offset_ms` after the crossing and its end is
extrapolated as `trial_ms` later. From the command line these, the bin width and the IR
settings come from the 'trial' section of the online config. Crossings inside a trial's
window are ignored, and a trial whose window runs past the end of the recording is left
out until it has been recorded.

    # bin the whole recording as one (T, N) .npy that `wasabi_online.py --source offline` picks up
    uv run python experiment/src/intan_binning.py RECORDING_DIR --out DATA_DIR/trial_000.npy

    # cut it into (n_trials, T, N) trials at IR crossings, e.g. as control trials
    uv run python experiment/src/intan_binning.py RECORDING_DIR --ir --out control.npy
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ReadIntanSpikeFile import readIntanSpikeFile
from config_loading import DEFAULT_CONFIG, load_config

ANALOG_VOLTS_PER_BIT = 0.0003125  # RHS analog inputs: V = 312.5 uV * (sample - 32768)


def bin_spikes(
    spikes: list, t_start: float, duration: float, bin_ms: float = 20.0
) -> np.ndarray:
    """(num_bins, num_channels) spike counts in `bin_ms` bins from `t_start` (s).

    `spikes` is the per-channel list from `readIntanSpikeFile`, timestamps in seconds.
    Only whole bins are kept; spikes in a trailing partial bin are dropped.
    """
    bin_s = bin_ms / 1000.0
    num_bins = int(round(duration / bin_s, 9))  # round so float error does not drop a whole bin
    edges = t_start + bin_s * np.arange(num_bins + 1)
    return np.stack(
        [np.histogram(channel[2], bins=edges)[0] for channel in spikes], axis=1
    ).astype(float)


def load_binned(
    recording_dir: Path, bin_ms: float = 20.0, no_artifacts: bool = True
) -> tuple[np.ndarray, list[str]]:
    """Bin the whole recording in `recording_dir`, from its first sample to its last whole bin.

    With `no_artifacts`, spikes Spike Scope flagged as artifacts (spike ID 128) are dropped.
    Returns (counts, native channel names of the columns).
    """
    recording_dir = Path(recording_dir)
    option = "noartifacts" if no_artifacts else "artifacts"
    spikes, sample_rate = readIntanSpikeFile(option, str(recording_dir / "spike.dat"))
    time = np.fromfile(recording_dir / "time.dat", dtype="<i4")
    duration = len(time) / sample_rate
    num_bins = int(duration * 1000.0 // bin_ms)
    counts = bin_spikes(spikes, time[0] / sample_rate, num_bins * bin_ms / 1000.0, bin_ms)
    return counts, [channel[0] for channel in spikes]


def _enabled_channels(recording_dir: Path, prefix: str) -> list[str]:
    """Native names of the enabled channels of one signal group, in the order RHX saves them."""
    root = ET.parse(recording_dir / "settings.xml").getroot()
    group = root.find(f".//SignalGroup[@Prefix='{prefix}']")
    if group is None:
        return []
    return [
        ch.get("NativeChannelName") for ch in group.iter("Channel") if ch.get("Enabled") == "True"
    ]


def read_ir_signal(
    recording_dir: Path, ir_channel: str = "DIGITAL-IN-01", analog_threshold: float = 1.0
) -> np.ndarray:
    """(n_samples,) bool, True while the IR line is high.

    DIGITAL-IN-xx is read from digitalin.dat (one uint16 per sample, bit xx-1 is the
    channel). ANALOG-IN-x is read from analogin.dat and thresholded at `analog_threshold` V.
    """
    recording_dir = Path(recording_dir)
    if ir_channel.startswith("DIGITAL-IN-"):
        path = recording_dir / "digitalin.dat"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: enable {ir_channel} in RHX before recording")
        bit = int(ir_channel.rsplit("-", 1)[1]) - 1
        words = np.memmap(path, dtype="<u2", mode="r")
        return (words >> bit) & 1 == 1
    if ir_channel.startswith("ANALOG-IN-"):
        enabled = _enabled_channels(recording_dir, "ANALOG-IN")
        if ir_channel not in enabled:
            raise ValueError(f"{ir_channel} was not enabled in this recording (enabled: {enabled})")
        raw = np.memmap(recording_dir / "analogin.dat", dtype="<u2", mode="r")
        raw = raw[: len(raw) // len(enabled) * len(enabled)].reshape(-1, len(enabled))
        volts = ANALOG_VOLTS_PER_BIT * (raw[:, enabled.index(ir_channel)].astype(float) - 32768)
        return volts > analog_threshold
    raise ValueError(f"IR channel must be DIGITAL-IN-xx or ANALOG-IN-x, got {ir_channel}")


def find_trial_onsets(
    ir_high: np.ndarray, trial_samples: int, edge: str = "falling"
) -> np.ndarray:
    """Sample indices of the IR crossings that start a trial.

    `edge` is "falling" (line goes low: the IR beam is crossed) or "rising" (line goes high). Crossings within `trial_samples` of the previous trial's onset
    are ignored, so a flickering beam does not start overlapping trials.
    """
    change = np.diff(ir_high.astype(np.int8))
    crossings = np.flatnonzero(change == (1 if edge == "rising" else -1)) + 1
    onsets = []
    for c in crossings:
        if not onsets or c >= onsets[-1] + trial_samples:
            onsets.append(c)
    return np.array(onsets, dtype=int)


def load_trials(
    recording_dir: Path,
    ir_channel: str = "DIGITAL-IN-01",
    edge: str = "falling",
    trial_ms: float = 1400.0,
    offset_ms: float = -800.0,
    bin_ms: float = 20.0,
    analog_threshold: float = 1.0,
    no_artifacts: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Cut the recording into trials at IR crossings and bin each one.

    Each trial runs from crossing + `offset_ms` (negative to include time before the
    crossing) to `trial_ms` after that. Returns (counts (n_trials, T, N), onset times in s
    of the IR crossings, native channel names of the columns); only complete trials are kept.
    """
    recording_dir = Path(recording_dir)
    option = "noartifacts" if no_artifacts else "artifacts"
    spikes, sample_rate = readIntanSpikeFile(option, str(recording_dir / "spike.dat"))
    time = np.fromfile(recording_dir / "time.dat", dtype="<i4")
    ir_high = read_ir_signal(recording_dir, ir_channel, analog_threshold)[: len(time)]

    num_bins = int(trial_ms // bin_ms)
    trial_samples = int(round(num_bins * bin_ms / 1000.0 * sample_rate))
    offset_samples = int(round(offset_ms / 1000.0 * sample_rate))
    onsets = find_trial_onsets(ir_high, trial_samples, edge)
    starts = onsets + offset_samples
    complete = (starts >= 0) & (starts + trial_samples <= len(time))
    onsets, starts = onsets[complete], starts[complete]

    duration = num_bins * bin_ms / 1000.0
    counts = np.stack(
        [bin_spikes(spikes, time[s] / sample_rate, duration, bin_ms) for s in starts]
    ) if len(starts) else np.zeros((0, num_bins, len(spikes)))
    return counts, time[onsets] / sample_rate, [channel[0] for channel in spikes]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument("--out", type=Path, help="save the counts here as .npy")
    parser.add_argument("--ir", action="store_true", help="cut into trials at IR crossings")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="bin width, trial window and IR settings come from its 'trial' section",
    )
    args = parser.parse_args()
    trial = load_config(args.config).trial

    if args.ir:
        counts, onsets, channels = load_trials(args.recording_dir, **trial.load_trials_kwargs())
        print(
            f"{len(counts)} trials of {counts.shape[1]} bins of {trial.bin_ms:g} ms x "
            f"{counts.shape[2]} channels; IR crossings at {np.round(onsets, 3)} s"
        )
    else:
        counts, channels = load_binned(args.recording_dir, trial.bin_ms, not trial.keep_artifacts)
        print(
            f"{counts.shape[0]} bins of {trial.bin_ms:g} ms x {counts.shape[1]} channels "
            f"({channels[0]} .. {channels[-1]}), {int(counts.sum())} spikes"
        )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out, counts)
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()

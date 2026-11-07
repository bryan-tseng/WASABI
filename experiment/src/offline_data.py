"""Recorded stim sessions for offline WASABI runs, from the RCP_analysis checkpoints.

Reads, for each Intan file of the session, the aligned checkpoint
<session>/results/checkpoints/Aligned/*/aligned__<intan>__Intan_<idx>__BR_<idx>.npz, which
holds the detected NPRW spike times per channel (nprw_peak_ms), the IR crossings (ir_ms),
the stim block onsets and bounds (stim_ms, stim_block_bounds_ms) and the stim channels
(nprw_meta["stim_channels"], 1-based). All times are in ms on one clock.

Every stim onset is at an IR crossing, so each one is a stim trial for that file's channels.
Other IR crossings in stim files are the beam being crossed again within the same reach, so
control trials come from separate no-stim reaching files (`control_files`), one per reach:
crossings within a trial window of the previous one are skipped. A trial is binned from
`window_ms` around its crossing and is dropped if the window runs off the recording or
(stim trials) contains another stim block.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Condition:
    """All trials recorded with one stim pattern."""

    intan_idx: int
    channels: np.ndarray  # 0-based stim channels
    trials: np.ndarray  # (n_trials, T, N) spike counts
    onsets_ms: np.ndarray  # (n_trials,) IR crossing of each trial
    stim_dur_ms: float


@dataclass
class RecordedSession:
    conditions: list[Condition]
    controls: np.ndarray  # (n_controls, T, N)
    control_sources: list[tuple[int, float]] = field(default_factory=list)  # (intan_idx, onset_ms)

    @property
    def stim_dur_ms(self) -> float:
        return float(np.median([c.stim_dur_ms for c in self.conditions]))

    @property
    def num_stim_channels(self) -> int:
        """Highest stimulated channel + 1, i.e. the channels the candidates span."""
        return int(max(c.channels.max() for c in self.conditions)) + 1


def _aligned_file(aligned_dir: Path, intan_idx: int) -> Path:
    matches = sorted(aligned_dir.glob(f"*/aligned__*__Intan_{intan_idx:03d}__*.npz"))
    if not matches:
        raise FileNotFoundError(f"no aligned checkpoint for Intan file {intan_idx} in {aligned_dir}")
    return matches[0]


def _bin(peaks: dict, num_channels: int, start_ms: float, num_bins: int, bin_ms: float) -> np.ndarray:
    edges = start_ms + bin_ms * np.arange(num_bins + 1)
    counts = np.zeros((num_bins, num_channels))
    for ch in range(num_channels):
        counts[:, ch] = np.histogram(np.asarray(peaks.get(ch, ()), float), bins=edges)[0]
    return counts


def _overlaps(start: float, end: float, intervals: np.ndarray) -> bool:
    return bool(np.any((intervals[:, 0] < end) & (intervals[:, 1] > start))) if len(intervals) else False


def _load(aligned_dir: Path, idx: int, num_channels: int):
    z = np.load(_aligned_file(aligned_dir, idx), allow_pickle=True)
    meta = z["nprw_meta"].item()
    if meta["n_channels"] < num_channels:
        raise ValueError(f"Intan file {idx} has {meta['n_channels']} channels, expected {num_channels}")
    return z, meta, z["nprw_peak_ms"].item()


def load_recorded_session(
    session_dir: Path,
    intan_files: list[int],
    control_files: list[int],
    window_ms: tuple[float, float] = (-800.0, 600.0),
    bin_ms: float = 20.0,
    num_channels: int = 128,
) -> RecordedSession:
    """Stim trials per condition and control trials, binned in `bin_ms` over `window_ms`."""
    aligned_dir = Path(session_dir) / "results" / "checkpoints" / "Aligned"
    num_bins = int(round((window_ms[1] - window_ms[0]) / bin_ms))
    trial_ms = window_ms[1] - window_ms[0]
    conditions, controls, control_sources = [], [], []

    def window(t):
        return t + window_ms[0], t + window_ms[1]

    def on_recording(t, meta):
        start, end = window(t)
        return start >= float(meta["rec_start_ms"]) and end <= float(meta["rec_end_ms"])

    for idx in intan_files:
        z, meta, peaks = _load(aligned_dir, idx, num_channels)
        stim = np.sort(z["stim_ms"].astype(float))
        blocks = z["stim_block_bounds_ms"].astype(float).reshape(-1, 2)

        # Stim trials: no other stim block inside the window
        onsets = []
        for s in stim:
            start, end = window(s)
            others = blocks[np.abs(blocks[:, 0] - s) > 1.0]
            if on_recording(s, meta) and not _overlaps(start, end, others):
                onsets.append(s)
        trials = np.stack([_bin(peaks, num_channels, window(s)[0], num_bins, bin_ms) for s in onsets]) \
            if onsets else np.zeros((0, num_bins, num_channels))
        conditions.append(Condition(
            idx, np.asarray(meta["stim_channels"], int) - 1, trials, np.array(onsets),
            float(np.median(z["stim_dur_measured_ms"])),
        ))

    for idx in control_files:
        z, meta, peaks = _load(aligned_dir, idx, num_channels)
        if len(z["stim_ms"]) if "stim_ms" in z.files else 0:
            raise ValueError(f"control file {idx} has stim blocks")
        last = -np.inf
        for t in np.sort(z["ir_ms"].astype(float)):
            if t < last + trial_ms or not on_recording(t, meta):
                continue  # a re-crossing within the same reach, or off the recording
            controls.append(_bin(peaks, num_channels, window(t)[0], num_bins, bin_ms))
            control_sources.append((idx, t))
            last = t
    if not controls:
        raise ValueError(f"no control trials in Intan files {control_files}")

    return RecordedSession(conditions, np.stack(controls), control_sources)

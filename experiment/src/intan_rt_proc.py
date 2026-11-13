"""Detect and bin spikes from the raw Intan amplifier data for `wasabi_online`.

Reads a recording saved as "one file per signal type" (amplifier.dat, settings.xml, time.dat, and digitalin.dat or analogin.dat for the IR beam). Spikes are detected causally,
chunk by chunk, so the same code works on a recording still being written:
  1. bandpass (Butterworth `band_hz`)
  2. common reference: subtract the median (or mean) of all channels, or with
     `local_radius` of the channels in a ring around each one (`positions_file`)
  3. matched filter with the waveform in `template_file` (none: just flip the sign)
  4. threshold at `threshold` x the noise SD from the first `noise_s` seconds, with
     `refractory_ms` between spikes on a channel

Trials start at each IR crossing (`edge` of `ir_channel`) and span `offset_ms` to
`offset_ms + trial_ms` around it, binned at `bin_ms`.

`control_preproc.py` is the command line that uses these to make control trials.
"""

import bisect
import csv
import threading
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy import io, signal

AMPLIFIER_UV_PER_BIT = 0.195
ANALOG_VOLTS_PER_BIT = 0.0003125  # RHS analog inputs: V = 312.5 uV * (sample - 32768)
MAD_TO_SD = 1 / 0.6745  # noise SD from the median absolute value of Gaussian noise
TRIALS_FILE = "control_trials.npy"  # control trials in a recording folder (control_preproc.py), loaded by paths.control_data_dir


def _enabled_channels(recording_dir: Path, prefix: str | None = None, enabled_only: bool = True) -> list[str]:
    """Native names of the enabled channels, in the order RHX saves them.
    Gets the names from settings.xml
    
    With `prefix`, of that signal group only (e.g. "ANALOG-IN"); without, of the amplifier
    ports, whose prefixes are single letters (A, B, C, D, ...). With `enabled_only` False,
    of every channel, enabled or not.
    """
    root = ET.parse(recording_dir / "settings.xml").getroot()
    groups = [
        g for g in root.iter("SignalGroup")
        if (g.get("Prefix") == prefix if prefix else len(g.get("Prefix", "")) == 1)
    ]
    return [
        ch.get("NativeChannelName")
        for g in groups for ch in g.iter("Channel") if ch.get("Enabled") == "True" or not enabled_only
    ]

# General memmap .dat file reader helper
def _read_rows(path: Path, dtype: str, num_cols: int, start: int, stop: int | None = None) -> np.ndarray:
    """Rows [start, stop) of a growing interleaved (n_rows, num_cols) file; only whole rows.
    Rows are samples, columns are channels.
    """
    num_rows = path.stat().st_size // (np.dtype(dtype).itemsize * num_cols) # size of the file divided by the size of each row
    stop = num_rows if stop is None else min(stop, num_rows)
    if stop <= start:
        return np.zeros((0, num_cols), dtype=dtype)
    # Memmap the file into rows x col (samples x channels)
    data = np.memmap(path, dtype=dtype, mode="r", shape=(num_rows, num_cols))
    # Only return the chunk
    return np.array(data[start:stop])


class IRReader:
    """Reads the IR line, True while it is high.

    DIGITAL-IN-xx is read from digitalin.dat (one uint16 per sample, bit xx-1 is the channel).
    ANALOG-IN-x is read from analogin.dat and thresholded at `analog_threshold` V.
    """

    def __init__(self, recording_dir: Path, ir_channel: str = "DIGITAL-IN-01", analog_threshold: float = 1.0):
        recording_dir = Path(recording_dir)
        self.analog_threshold = analog_threshold
        if ir_channel.startswith("DIGITAL-IN-"):
            self.path = recording_dir / "digitalin.dat"
            if not self.path.exists():
                raise FileNotFoundError(f"{self.path} not found: enable {ir_channel} in RHX before recording")
            self.num_cols, self.col = 1, 0
            self.bit = int(ir_channel.rsplit("-", 1)[1]) - 1
        elif ir_channel.startswith("ANALOG-IN-"):
            enabled = _enabled_channels(recording_dir, "ANALOG-IN")
            if ir_channel not in enabled:
                raise ValueError(f"{ir_channel} was not enabled in this recording (enabled: {enabled})")
            self.path = recording_dir / "analogin.dat"
            self.num_cols, self.col, self.bit = len(enabled), enabled.index(ir_channel), None
        else:
            raise ValueError(f"IR channel must be DIGITAL-IN-xx or ANALOG-IN-x, got {ir_channel}")

    def num_samples(self) -> int:
        return self.path.stat().st_size // (2 * self.num_cols) # size of the file divided by the size of each row (2 bytes per sample) and num_channels

    def read(self, start: int, stop: int) -> np.ndarray:
        raw = _read_rows(self.path, "<u2", self.num_cols, start, stop)[:, self.col]
        if self.bit is not None: # TODO check if this logic works for analog signal
            return (raw >> self.bit) & 1 == 1
        return ANALOG_VOLTS_PER_BIT * (raw.astype(float) - 32768) > self.analog_threshold

# Neighbor calculation helpers
def load_positions(path: Path, channels: list[str], all_channels: list[str] | None = None) -> np.ndarray:
    """(N, 2) positions of `channels`, NaN where a site is not connected or not in the map.

    From a CSV channel,x,y, by name, or a Kilosort channel map .mat, by data column: the
    channel's index in `all_channels` (every amplifier channel, enabled or not, so disabled
    channels leave gaps), or in `channels` without it. Channels past the map are left out.
    """
    path = Path(path)
    if path.suffix == ".mat":
        m = io.loadmat(path, squeeze_me=True)
        columns = np.full((len(m["chanMap0ind"]), 2), np.nan)  # position of each data column
        connected = m.get("connected", np.ones(len(columns), dtype=bool)).astype(bool)
        columns[m["chanMap0ind"][connected]] = np.c_[m["xcoords"], m["ycoords"]][connected]
        index = np.array([(all_channels or channels).index(ch) for ch in channels])
        unmapped = [ch for ch, i in zip(channels, index) if i >= len(columns)]
        if unmapped:
            warnings.warn(f"{path.name} maps {len(columns)} columns; no position for {unmapped}", stacklevel=2)
        positions = np.full((len(channels), 2), np.nan)
        positions[index < len(columns)] = columns[index[index < len(columns)]]
        return positions
    with open(path, newline="") as f:
        rows = {row["channel"]: (float(row["x"]), float(row["y"])) for row in csv.DictReader(f)}
    missing = [ch for ch in channels if ch not in rows]
    if missing:
        raise ValueError(f"{path} has no position for {missing}")
    return np.array([rows[ch] for ch in channels])


def local_neighbors(positions: np.ndarray, local_radius: tuple[float, float]) -> list[np.ndarray]:
    """For each channel, get the channels between distances with inner < d <= outer."""
    inner, outer = local_radius
    if not 0 <= inner < outer:
        raise ValueError(f"local_radius must be (inner, outer) with 0 <= inner < outer, got {local_radius}")
    distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)  # NaN: never a neighbor
    return [np.flatnonzero((d > inner) & (d <= outer)) for d in distances]


class SpikeDetector:
    """Causal bandpass, common reference, matched filter and threshold on (n_samples, N) chunks.

    Feed consecutive chunks to `process`; it returns the detected spikes as (sample, channel), samples counted from the first chunk.
    
    Before processing, it holds a buffer of up to `noise_s` seconds of data, to estimate the thresholds used for matched filter, and then starts estimating the spikes.
    """

    def __init__(
        self,
        num_channels: int,
        sample_rate: float,
        band_hz: tuple[float, float] = (300.0, 6000.0),
        filter_order: int = 3,
        reference: str = "median",
        threshold: float = 4.5,
        refractory_ms: float = 1.0,
        noise_s: float = 10.0,
        neighbors: list[np.ndarray] | None = None,
        template: np.ndarray | None = None,
    ):
        """`neighbors[i]`: the channels whose median (or mean) channel i subtracts; None: all.

        `template`: spike waveform to match; None detects negative crossings of the signal.
        """
        if reference not in ("median", "mean", "none"):
            raise ValueError(f"reference must be median, mean or none, got {reference!r}")
        
        self.sos = signal.butter(filter_order, band_hz, btype="bandpass", fs=sample_rate, output="sos")
        self.zi = None  # filter state, (n_sections, 2, N)
        self.reference = reference
        self.groups = None if neighbors is None else _group_by_count(neighbors)
        if template is None:
            self.template = np.array([-1.0], dtype=np.float32)
        else:
            template = np.asarray(template, dtype=float) - np.mean(template)
            self.template = (template / np.linalg.norm(template)).astype(np.float32)
        self.lag = len(self.template) - 1 - int(np.argmin(self.template)) # Where the trough of the template sits
        self.tail = np.zeros((len(self.template) - 1, num_channels), dtype=np.float32) # Carry over previous 89 samples so the template can be matched across chunks
        self.threshold = threshold
        self.refractory = round(refractory_ms / 1000.0 * sample_rate)
        self.noise_samples = round(noise_s * sample_rate)
        self.num_channels = num_channels
        self.thresholds = None  # (N,) matched filter output, set once the noise is estimated
        self.pending: list[np.ndarray] = []  # matched filter output held back until then
        self.num_processed = 0  # samples filtered so far
        self.num_detected = 0  # samples searched for spikes so far
        self.start_sample = 0  # first sample of the data fed so far, without gaps (see `restart`)
        self.above = np.zeros(num_channels, dtype=bool)  # last searched sample was above threshold
        self.last_spike = np.full(num_channels, -np.inf)

    # Skip ahead: the next chunk starts at `sample`, so drop the state carried over from the last one
    def restart(self, sample: int) -> None:
        """Continue at recording sample `sample` after a gap, keeping the thresholds but dropping the states from last chunk."""
        if self.thresholds is None or self.pending:
            raise RuntimeError("restart only after the thresholds are set and every chunk is searched")
        self.zi = None  # the filter starts again in steady state at the next chunk's first sample
        self.tail[:] = 0
        self.num_processed = self.num_detected = self.start_sample = sample
        self.above[:] = False
        self.last_spike[:] = -np.inf

    # Step 1-2: bandpass, then subtract the common reference
    def _filter(self, raw: np.ndarray) -> np.ndarray:
        """Bandpass and common-reference the next chunk, (n_samples, N) int16 -> uV."""
        x = raw.astype(np.float32) * AMPLIFIER_UV_PER_BIT
        if self.zi is None:  # start in steady state at the first sample, without a step transient
            self.zi = signal.sosfilt_zi(self.sos)[:, :, None] * x[0]
        x, self.zi = signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
        x = x.astype(np.float32)
        
        # Common mode median reference
        if self.reference != "none":
            x -= self._common_reference(x)
        self.num_processed += len(x)
        return x

    # Median (or mean) of all channels, or of each channel's ring of neighbors
    def _common_reference(self, x: np.ndarray) -> np.ndarray:
        """The reference subtracted from bandpassed `x`: (n, 1) global, or (n, N) local."""
        operator = np.median if self.reference == "median" else np.mean
        if self.groups is None:
            return operator(x, axis=1, keepdims=True)
        xt = np.ascontiguousarray(x.T)  # channel-major, so gathering neighbors copies whole rows
        ref = np.zeros_like(xt)
        for channels, neighbors in self.groups:  # (m,), (m, k): k neighbors each
            v = xt[neighbors]  # (m, k, n)
            if self.reference == "mean":
                ref[channels] = v.mean(axis=1)
            else:  # median by sorting: faster than np.median
                k = neighbors.shape[1]
                v.sort(axis=1)
                ref[channels] = 0.5 * (v[:, (k - 1) // 2] + v[:, k // 2]) # Get median
        return ref.T

    # Step 3: matched filter, correlate each channel with the template
    def _match(self, x: np.ndarray) -> np.ndarray:
        """Matched filter output of the next filtered chunk; y[n] uses samples up to n."""
        if len(self.template) == 1:
            return x * self.template[0]
        # Pad the last samples of the previous chunk
        padded = np.concatenate([self.tail, x])
        self.tail = padded[len(x):]
        if not len(x):
            return x
        # Reverse template for correlation
        kernel = self.template[::-1, None] 
        # Convolving with the reversed template
        return signal.oaconvolve(padded, kernel, mode="valid", axes=0).astype(np.float32)

    # Step 4: threshold crossings, timed at the template's trough
    def _detect(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        above = y > self.thresholds
        previous = np.vstack([self.above[None], above[:-1]])
        samples, channels = np.nonzero(above & ~previous)  # sorted by sample
        samples = samples + self.num_detected - self.lag
        # Drop crossings within the refractory period of the previous one on the channel
        keep = np.ones(len(samples), dtype=bool)
        for ch in np.unique(channels):  # crossings are sparse, so this loop is short
            idx = np.flatnonzero(channels == ch)
            gaps = np.diff(samples[idx], prepend=self.last_spike[ch])
            keep[idx] = gaps > self.refractory
            self.last_spike[ch] = samples[idx[-1]]
        self.above = above[-1]
        self.num_detected += len(y)
        keep &= samples >= self.start_sample  # matches against the zero padding before the first sample
        return samples[keep], channels[keep]

    # All four steps on the next chunk
    def process(self, raw: np.ndarray, final: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Spikes (sample indices, channel indices) found in `raw` and any chunks held back.

        With `final`, held-back chunks are searched even if fewer than `noise_s` seconds have been seen (the noise is then estimated from all of them).
        
        This step is run for each 0.25-second chunk of raw data, except for the first 10s where the noise estimate needs to be done.
        """
        # Bandpass filter, common mode reference, and matched filter
        if len(raw):
            self.pending.append(self._match(self._filter(raw)))
        # Hold chunks back until noise_s seconds are in, then set the thresholds from them
        if self.thresholds is None:
            if not self.pending or (self.num_processed < self.noise_samples and not final):
                return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
            # Get noise_s of data for noise estimation from pending chunks # TODO consider robust adaptive noise estimation?
            noise = np.concatenate(self.pending)[: self.noise_samples]
            # MAD
            self.thresholds = self.threshold * MAD_TO_SD * np.median(np.abs(noise), axis=0)
        # Search for spikes in held back window (10s for first chunk, 0.25s for subsequent chunks)
        found = [self._detect(x) for x in self.pending]
        self.pending = []
        if not found:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        return np.concatenate([f[0] for f in found]), np.concatenate([f[1] for f in found])


def _group_by_count(neighbors: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Channels with the same number of neighbors, so each group is referenced in one call."""
    alone = [i for i, neighbor in enumerate(neighbors) if len(neighbor) == 0]
    if alone:
        warnings.warn(f"channels {alone} have no neighbors in local_radius and are left unreferenced", stacklevel=3)
    counts = sorted({len(neighbor) for neighbor in neighbors} - {0})
    groups = []
    for k in counts:
        channels = np.array([i for i, neighbor in enumerate(neighbors) if len(neighbor) == k])
        groups.append((channels, np.stack([neighbors[i] for i in channels])))
    return groups


def make_detector(
    channels: list[str],
    sample_rate: float,
    local_radius: tuple[float, float] | None = None,
    positions_file: Path | None = None,
    template_file: Path | None = None,
    all_channels: list[str] | None = None,
    **detection,
) -> SpikeDetector:
    """A `SpikeDetector` for `channels`: local reference with `local_radius`, matched filter with `template_file`."""
    neighbors = None
    if local_radius is not None:
        if positions_file is None:
            raise ValueError("local_radius needs positions_file, the channel positions")
        neighbors = local_neighbors(load_positions(positions_file, channels, all_channels), local_radius)
    template = None if template_file is None else np.load(template_file)
    if template is not None and template.ndim != 1:
        raise ValueError(f"{template_file} must hold one 1-D waveform, got shape {template.shape}")
    return SpikeDetector(len(channels), sample_rate, neighbors=neighbors, template=template, **detection)


def find_trial_onsets(ir_high: np.ndarray, trial_samples: int, edge: str = "falling") -> np.ndarray:
    """Sample indices of the IR crossings that start a trial.

    `edge` is "falling" (line goes low: the IR beam is crossed) or "rising" (line goes high).
    Crossings within `trial_samples` of the previous trial's onset are ignored, so a
    flickering beam does not start overlapping trials.
    """
    change = np.diff(ir_high.astype(np.int8))
    crossings = np.flatnonzero(change == (1 if edge == "rising" else -1)) + 1
    onsets = []
    for c in crossings:
        if not onsets or c >= onsets[-1] + trial_samples:
            onsets.append(c)
    return np.array(onsets, dtype=int)


class IntanStream:
    """Follows an RHX recording: detects spikes in amplifier.dat and IR crossings as they are written.

    `update` processes the samples written since the last call, in chunks of `chunk_s`
    seconds; `start` does so continuously in a background thread so the spikes are up to
    date whenever a trial is asked for. Without `ir_channel` only spikes are tracked.
    """

    def __init__(
        self,
        recording_dir: Path,
        sample_rate: float = 30000.0,
        ir_channel: str | None = "DIGITAL-IN-01",
        edge: str = "falling",
        trial_ms: float = 1400.0,
        offset_ms: float = -800.0,
        bin_ms: float = 20.0,
        analog_threshold: float = 1.0, # TODO need to set this when we include the analog inputs
        chunk_s: float = 0.25, # TODO Can make this a hyperparameter?
        **detection,
    ):
        # Recording: files and metadata
        self.recording_dir = Path(recording_dir)
        self.amplifier = self.recording_dir / "amplifier.dat"
        if not self.amplifier.exists():
            raise FileNotFoundError(f"{self.amplifier} not found: save as one file per signal type")
        self.channels = _enabled_channels(self.recording_dir)  # From settings.xml
        if not self.channels:
            raise ValueError(f"no enabled amplifier channels in {self.recording_dir / 'settings.xml'}")
        self.sample_rate = sample_rate
        
        self.ir = IRReader(self.recording_dir, ir_channel, analog_threshold) if ir_channel else None  # digitalin.dat or analogin.dat
        
        time_path = self.recording_dir / "time.dat"
        self.t0 = int(np.fromfile(time_path, dtype="<i4", count=1)[0]) if time_path.exists() else 0  # First timestamp of the recording

        # Spike detection
        all_channels = _enabled_channels(self.recording_dir, enabled_only=False)
        self.detector = make_detector(self.channels, sample_rate, all_channels=all_channels, **detection)

        # Trial and binning geometry, in samples
        self.edge = edge
        self.bin_samples = round(bin_ms / 1000.0 * sample_rate)
        if not np.isclose(self.bin_samples, bin_ms / 1000.0 * sample_rate):
            raise ValueError(f"{bin_ms:g} ms bins are not a whole number of samples at {sample_rate:g} Hz")
        self.num_bins = int(trial_ms // bin_ms)
        self.trial_samples = self.num_bins * self.bin_samples
        self.offset_samples = round(offset_ms / 1000.0 * sample_rate)

        # Processing settings
        self.chunk_samples = round(chunk_s * sample_rate)

        # Stream position
        self.num_read = 0  # Samples read from amplifier.dat (and the IR file)
        self._ir_last = None  # IR level at the last sample read
        self.valid_from = 0  # Trials must start here or later; samples before a jump were skipped
        self.holding = False  # Stop once the thresholds are set, until `resume`

        # Detected events
        self._spike_chunks: list[tuple[np.ndarray, np.ndarray]] = []  # (samples, channels) per chunk, merged by `spikes`
        self._spikes = (np.zeros(0, dtype=int), np.zeros(0, dtype=int))
        self._onsets: list[int] = []

        # Diagnostics
        self.process_s = 0.0  # Total time spent reading and detecting
        self._seen_samples: list[int] = []  # Samples on disk at each `update` that found new ones...
        self._seen_at: list[float] = []  # ...and the wall time (time.time) it found them

        # Threading
        self.lock = threading.Lock()  # Guards stream position, detected events and detector
        self._stop = threading.Event()
        self._thread = None
        self.error: Exception | None = None

    def _available(self) -> int:
        """
            Returns the number of samples available for reading from both the amplifier and IR files.
            2 bytes x channels per sample is the number of complete samples, so the division drops incomplete samples.
        """
        n = self.amplifier.stat().st_size // (2 * len(self.channels)) # size of the file divided by the size of each row (2 bytes per sample) and num_channels
        return min(n, self.ir.num_samples()) if self.ir else n

    def num_written(self) -> int:
        """Samples on disk so far, in both amplifier.dat and the IR file"""
        return self._available()

    def seen_at(self, stop: int) -> float:
        """Wall time (time.time) samples [0, stop) were first all seen on disk.

        Seen by `update`, so later than written by up to the thread's poll interval, or by the
        processing still under way when they were written.
        """
        with self.lock:
            i = bisect.bisect_left(self._seen_samples, stop)
            if i == len(self._seen_samples):
                raise ValueError(f"sample {stop} has not been seen on disk yet")
            return self._seen_at[i]

    # Find IR crossings in the next chunk, including one right at the chunk boundary
    def _read_ir(self, start: int, stop: int) -> None:
        high = self.ir.read(start, stop)
        if self._ir_last is not None:
            high = np.concatenate([[self._ir_last], high])
            start -= 1
        self._ir_last = high[-1]
        for c in find_trial_onsets(high, self.trial_samples, self.edge) + start:
            if not self._onsets or c >= self._onsets[-1] + self.trial_samples:
                self._onsets.append(int(c))

    # Read and process new samples, chunk by chunk
    def update(self, final: bool = False) -> int:
        """Process every sample written so far; returns how many were new.

        With `final` (the recording is done), samples held back for the noise estimate are searched for spikes even if the recording is shorter than `noise_s`.
        While holding, stops once the thresholds are set.
        """
        stop = self._available() # current available samples on disk
        with self.lock:
            if not self._seen_samples or stop > self._seen_samples[-1]:
                self._seen_samples.append(stop)
                self._seen_at.append(time.time())
        num_new = 0
        # Chunking into 0.25-second blocks for processing (7500 samples, one each row of raw).
        # The lock is taken per chunk, so another thread waits for one chunk at most
        while True:
            with self.lock: # Lock this chunk from other threads
                if self.num_read >= stop or (self.holding and self.detector.thresholds is not None):
                    # Stop if no new samples, or if a hold is placed, and thresholds are already set (control fit)
                    break
                t0 = time.perf_counter()
                end = min(self.num_read + self.chunk_samples, stop) # The REAL ending
                raw = _read_rows(self.amplifier, "<i2", len(self.channels), self.num_read, end)
                
                # Do the same for IR
                if self.ir:
                    self._read_ir(self.num_read, end)
                
                # Detect spikes
                self._add_spikes(self.detector.process(raw))
                
                # Update values
                num_new += end - self.num_read
                self.num_read = end
                self.process_s += time.perf_counter() - t0
        if final:
            with self.lock:
                t0 = time.perf_counter()
                self._add_spikes(self.detector.process(np.zeros((0, len(self.channels))), final=True))
                self.process_s += time.perf_counter() - t0
        return num_new

    # Process only the first noise_s seconds (for the thresholds), then wait for `resume`
    def hold(self) -> None:
        """Stop processing once the thresholds are set, e.g. while the control fit runs."""
        self.holding = True

    def resume(self, sample: int) -> None:
        """Carry on processing from recording sample `sample`, skipping what came before it.

        Without a jump (the thresholds not set yet, or `sample` already read), it carries on from where it stopped.
        """
        with self.lock:
            self.holding = False
            if self.detector.thresholds is None or sample <= self.num_read:
                return
            self.detector.restart(sample)
            self.num_read = self.valid_from = sample
            self._ir_last = None  # a crossing right at the jump is missed

    # Wait for the background thread to process everything on disk now
    def catch_up(self, poll_s: float = 0.005) -> None:
        """Block until every sample written so far has been processed (by `update` if no thread runs)."""
        if self._thread is None:
            self.update()
            return
        target = self._available()
        while self.num_read < target and not (self.holding and self.detector.thresholds is not None):
            self.check()
            time.sleep(poll_s)

    def _add_spikes(self, spikes: tuple[np.ndarray, np.ndarray]) -> None:
        if len(spikes[0]):
            self._spike_chunks.append(spikes)

    # Background thread: keep calling update while the recording grows
    def _run(self, poll_s: float) -> None:
        try:
            while not self._stop.is_set():
                if self.update() == 0:
                    self._stop.wait(poll_s)
                else:
                    self._stop.wait(0.001)  # let a thread waiting for the lock in
        except Exception as e:  # surfaced by `check`, the main thread cannot see it otherwise
            self.error = e

    def start(self, poll_s: float = 0.1) -> None:
        """Keep processing new samples in a background thread."""
        self._thread = threading.Thread(target=self._run, args=(poll_s,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def check(self) -> None:
        """Raise the error that stopped the background thread, if any."""
        if self.error is not None:
            raise RuntimeError("Intan processing stopped") from self.error

    # Merge newly detected spikes into one sorted list
    @property
    def spikes(self) -> tuple[np.ndarray, np.ndarray]:
        """(sample indices, channel indices) of every spike detected so far, sorted by sample."""
        with self.lock:
            if self._spike_chunks:
                samples, channels = zip(self._spikes, *self._spike_chunks)
                self._spikes = (np.concatenate(samples), np.concatenate(channels))
                self._spike_chunks = []
            return self._spikes

    # Bin spike times into counts
    def bin(self, start: int, num_bins: int) -> np.ndarray:
        """(num_bins, N) spike counts in bins from sample `start`."""
        samples, channels = self.spikes
        stop = start + num_bins * self.bin_samples
        lo, hi = np.searchsorted(samples, [start, stop])
        counts = np.zeros((num_bins, len(self.channels)))
        np.add.at(counts, ((samples[lo:hi] - start) // self.bin_samples, channels[lo:hi]), 1)
        return counts

    # Get IR crossings for times that have a complete trial time
    def onsets(self) -> np.ndarray:
        """Sample indices of the IR crossings whose whole trial window has been searched for spikes."""
        with self.lock:
            onsets = np.array(self._onsets, dtype=int)
            searched = self.detector.num_detected - self.detector.lag  # spikes known up to here
            valid_from = self.valid_from
        starts = onsets + self.offset_samples
        return onsets[(starts >= valid_from) & (starts + self.trial_samples <= searched)]

    # Extract a trial given IR crossing time (onset)
    def trial(self, onset: int) -> np.ndarray:
        """(T, N) spike counts of the trial at IR crossing sample `onset`."""
        return self.bin(onset + self.offset_samples, self.num_bins)

    # Convert IR crossing sample index to recording time
    def onset_time(self, onset: int | np.ndarray) -> float | np.ndarray:
        """Recording time (s, by time.dat) of sample `onset`."""
        return (self.t0 + onset) / self.sample_rate


def load_binned(recording_dir: Path, bin_ms: float = 20.0, **stream_kwargs) -> tuple[np.ndarray, list[str]]:
    """Bin the whole recording in `recording_dir`, from its first sample to its last whole bin.

    Returns (counts, native channel names of the columns).
    """
    stream = IntanStream(recording_dir, ir_channel=None, bin_ms=bin_ms, **stream_kwargs)
    stream.update(final=True)
    return stream.bin(0, stream.num_read // stream.bin_samples), stream.channels


def load_trials(recording_dir: Path, **stream_kwargs) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Cut the recording into trials at IR crossings and bin each one (see `IntanStream`).

    Returns (counts (n_trials, T, N), onset times in s of the IR crossings, native channel
    names of the columns); only complete trials are kept.
    """
    # Preprocessing: detect spikes and IR crossings over the whole recording
    stream = IntanStream(recording_dir, **stream_kwargs)
    stream.update(final=True)
    onsets = stream.onsets()

    # Bin the spikes of each trial
    counts = (
        np.stack([stream.trial(o) for o in onsets]) if len(onsets)
        else np.zeros((0, stream.num_bins, len(stream.channels)))
    )
    return counts, stream.onset_time(onsets), stream.channels

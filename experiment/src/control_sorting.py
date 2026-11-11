"""PLACEHOLDER: spike template for the matched filter, from automated sorting of a control recording.

Not implemented yet; the steps below are pseudo code, to revisit once there is real data.

Why: `spikes.template_file` is now the median waveform of Kilosort units on the RCP_analysis
preprocessed recording, which was most likely filtered forward and backward (zero phase).
Online, `intan_rt_proc.SpikeDetector` filters causally (`sosfilt`), which changes the spike's
shape (correlation ~0.8 with the zero-phase shape), so the matched filter loses sensitivity.
The template should be cut from data filtered exactly as online.

Plan:
    1. sort the control recording (Kilosort4 through SpikeInterface), for spike times and units
    2. filter the same recording causally, as online: `SpikeDetector.filter` (bandpass + local
       median reference), chunk by chunk
    3. cut snippets around each unit's spike times on its extremum channel from the causal
       data, aligned at the trough
    4. average per unit, normalize, take the median over units -> 1-D template
       (90 samples, trough at sample 30, as the current one)
    5. save to experiment/config/waveform_templates/ and point spikes.template_file at it

    uv run python experiment/src/control_sorting.py                 # paths.control_data_dir
    uv run python experiment/src/control_sorting.py RECORDING_DIR --out waveform_templates/causal_NPRW.npy
"""

import argparse
from pathlib import Path

import numpy as np

from config_loading import CONFIG_DIR, DEFAULT_CONFIG, load_config

MS_BEFORE, MS_AFTER = 1.0, 2.0  # snippet around the trough: 30 + 60 samples at 30 kHz, as now


def sort_recording(recording_dir: Path, sorter: str = "kilosort4") -> tuple[np.ndarray, np.ndarray]:
    """Automated sorting of the control recording -> (spike sample indices, unit ids).

    Pseudo code:
        rec = si.read_intan / the RHX folder
        rec = preprocess as for sorting (bandpass, local reference; zero phase is fine here,
              only the spike times are used)
        sorting = ss.run_sorter(sorter, rec, folder=...)
        keep units passing quality metrics (firing rate, ISI violations, SNR)
        return the spike samples and unit of each spike
    """
    raise NotImplementedError("control_sorting: automated sorting not implemented yet")


def extremum_channels(recording_dir: Path, samples: np.ndarray, units: np.ndarray) -> dict[int, int]:
    """Channel with the largest trough of each unit (e.g. si.get_template_extremum_channel)."""
    raise NotImplementedError


def causal_snippets(
    recording_dir: Path, samples: np.ndarray, channels: np.ndarray, cfg
) -> np.ndarray:
    """(n_spikes, n_samples) snippets of the causally filtered data, as online.

    Pseudo code:
        detector = make_detector(channels of the recording, sample_rate, **spike settings)
        for each 0.25 s chunk of amplifier.dat (intan_rt_proc._read_rows):
            x = detector.filter(raw)          # same bandpass state and local reference as online
            for spikes whose snippet lies in this chunk (carry the edges over):
                cut x[s - before : s + after, channel]
        re-align each snippet at its trough (+- a few samples), drop clipped or artifact ones
    """
    raise NotImplementedError


def make_template(snippets: np.ndarray, units: np.ndarray) -> np.ndarray:
    """Median over units of each unit's mean snippet, normalized as the current template.

    Pseudo code:
        per unit: mean snippet, divided by its largest absolute value
        template = median over units
        (SpikeDetector makes it zero mean and unit norm itself)
    """
    raise NotImplementedError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording_dir", type=Path, nargs="?", help="default: paths.control_data_dir of the config")
    parser.add_argument(
        "--out", type=Path, default=Path("waveform_templates/causal_template.npy"),
        help="template .npy, relative to experiment/config",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    recording_dir = args.recording_dir or cfg.data_path(cfg.paths.control_data_dir)

    samples, units = sort_recording(recording_dir)
    extremum = extremum_channels(recording_dir, samples, units)
    snippets = causal_snippets(recording_dir, samples, np.array([extremum[u] for u in units]), cfg)
    template = make_template(snippets, units)

    out = args.out if args.out.is_absolute() else CONFIG_DIR / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, template)
    print(f"saved {len(template)}-sample template to {out}; set spikes.template_file to it")


if __name__ == "__main__":
    main()

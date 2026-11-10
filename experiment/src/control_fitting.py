"""Cut a finished Intan RHX recording into control trials for `wasabi_online`.

Detects spikes in the recording's amplifier.dat, cuts it into trials at the IR crossings and
bins each one, all with the functions of `intan_binning` and the settings of the online
config: bin width and trial window ('trial'), IR channel and edge, spike detection
('spikes') and recording.sample_rate. The (n_trials, T, N) counts are saved as control_trials.npy in
the recording folder, so `paths.control_data_dir` (or --control-data-dir) can be that folder.

    uv run python experiment/src/control_fitting.py                  # the folder in paths.control_data_dir
    uv run python experiment/src/control_fitting.py RECORDING_DIR
    uv run python experiment/src/control_fitting.py RECORDING_DIR --out control.npy
"""

import argparse
from pathlib import Path

import numpy as np

from config_loading import DEFAULT_CONFIG, load_config
from intan_binning import TRIALS_FILE, load_trials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "recording_dir", type=Path, nargs="?",
        help="Intan RHX recording folder (one file per signal type); default: paths.control_data_dir of the config",
    )
    parser.add_argument("--out", type=Path, help=f"where to save the trials (default: RECORDING_DIR/{TRIALS_FILE})")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="bin width, trial window, IR and spike detection settings come from it",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    recording_dir = args.recording_dir or cfg.data_path(cfg.paths.control_data_dir)
    if recording_dir is None or not recording_dir.is_dir():
        raise SystemExit(f"give a recording folder, or set paths.control_data_dir to one in {args.config} (got {recording_dir})")
    counts, onsets, channels = load_trials(recording_dir, **cfg.intan_stream_kwargs())
    print(
        f"{len(counts)} trials of {counts.shape[1]} bins of {cfg.trial.bin_ms:g} ms x "
        f"{counts.shape[2]} channels ({channels[0]} .. {channels[-1]}); "
        f"IR crossings at {np.round(onsets, 3)} s"
    )
    if not len(counts):
        raise SystemExit(f"no complete IR trials: check trial.ir_channel and trial.ir_edge in {args.config}")
    out = args.out or recording_dir / TRIALS_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, counts)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
"""Preprocess a finished Intan recording into control trials to warmstart `wasabi_online`.
Using functions in `intan_rt_proc` so stays consistent with online config (e.g. bin width, trial window ... etc.)

1. Detects spikes in the recording's amplifier.dat
2. Cuts it into trials at the IR crossings
3. Binning each trial
4. Save processed counts in `paths.control_data_dir/control_trials.npy`
.json file in the same folder saves the preprocessing time, IR onsets and channel names.

    uv run python experiment/src/control_preproc.py                  # the folder in paths.control_data_dir
    
    # specify recording directory
    uv run python experiment/src/control_preproc.py RECORDING_DIR
    # skips preprocessing
    uv run python experiment/src/control_preproc.py RECORDING_DIR --out control.npy
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from config_loading import DEFAULT_CONFIG, load_config
from intan_rt_proc import TRIALS_FILE, load_trials


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
    # Parse inputs
    args = parse_args()
    cfg = load_config(args.config)
    recording_dir = args.recording_dir or cfg.data_path(cfg.paths.control_data_dir)
    if recording_dir is None or not recording_dir.is_dir():
        raise SystemExit(f"give a recording folder, or set paths.control_data_dir to one in {args.config} (got {recording_dir})")
    
    # Load and preprocess control trials
    t0 = time.perf_counter()
    counts, onsets, channels = load_trials(recording_dir, **cfg.intan_stream_kwargs())
    preprocess_s = time.perf_counter() - t0
    print(
        f"{len(counts)} trials of {counts.shape[1]} bins of {cfg.trial.bin_ms:g} ms x "
        f"{counts.shape[2]} channels ({channels[0]} .. {channels[-1]}); "
        f"IR crossings at {np.round(onsets, 3)} s\n"
        f"load {preprocess_s:.2f} s (reading, spike detection and binning)"
    )
    if not len(counts):
        raise SystemExit(f"no complete IR trials: check trial.ir_channel and trial.ir_edge in {args.config}")
    
    # Save .npy file
    out = args.out or recording_dir / TRIALS_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, counts)
    # Alongside: how long the preprocessing took (counted in wasabi_online's control load time)
    info = {"preprocess_s": preprocess_s, "onsets_s": np.asarray(onsets).tolist(), "channels": list(channels)}
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(f"saved to {out} (and {out.with_suffix('.json').name})")

if __name__ == "__main__":
    main()
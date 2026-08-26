# WASABI
Weighted Adaptive Stimulation After Batch-testing Identification (WASABI)

# Overview
This repository contains the simulation code for the identification of stimulation parameters.  

Folder structure:
```
.
├── docs                     # Problem formulation
├── src
│   └── wasabi               # Functions for scripts 
├── scripts                  # Scripts for plotting
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
pip install -e .                  # or: pip install -e ".[cuda]"
```

---
# Details of the pipeline
1. 
2. 


## Outputs
1. 
2. 


## TODOs

- 

---

## Evaluation

## Contact
For questions, contact Bryan Tseng btseng2@jh.edu.

## Acknowledgements
Adam S. Charles, Sai Koukuntla.

## License
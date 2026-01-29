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
Use **Anaconda Prompt** on windows, make sure you have CUDA installed
```bash
*navigate to desired directory*
# 1) clone repo
git clone https://github.com/bryan-tseng/WASABI
cd WASABI

# 2) create environment
conda env create -f environment.yml -n wasabi
conda activate wasabi
python -m pip install --upgrade pip

# 3) install WASABI package in editable mode
pip install -e .
```

---
# Details of the pipeline
1. 
2. 


## Outputs
1. Raster plots of first trial
2. Spike times of simulated data (nTrial x nChannel x Time of spikes)


## TODOs

- 

---

## Evaluation

## Contact
For questions, contact Bryan Tseng btseng2@jh.edu.

## Acknowledgements
Adam S. Charles, Sai Koukuntla.

## License
# SOLWEIG-GPU: GPU-Accelerated Thermal Comfort Modeling

This repository provides a command-line interface (CLI) for running the SOLWEIG (Solar and LongWave Environmental Irradiance Geometry) model with GPU acceleration. It supports urban microclimate modeling by computing thermal comfort indices such as mean radiant temperature using high-resolution spatial data.

## Features

- GPU-accelerated processing for efficient computation
- Support for custom or reanalysis meteorological input
- Modular input configuration
- Optional outputs: SVF, radiation fluxes, shadow maps, etc.
- Compatible with WRF and ERA5 meteorological data

## Installation

Clone the repository and set up the environment (assumes `conda` or `venv`):

```bash
git clone https://github.com/yourusername/solweig-gpu.git
conda install -c conda-forge gdal
cd solweig-gpu
pip install -r requirements.txt


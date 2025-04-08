# SOLWEIG-GPU: GPU-Accelerated Thermal Comfort Modeling

This repository provides a Python package and a command-line interface (CLI) for running the SOLWEIG (Solar and LongWave Environmental Irradiance Geometry) model with GPU acceleration. It supports urban microclimate modeling by using high-resolution spatial data to compute thermal comfort indices such as mean radiant temperature.

## Features

- GPU-accelerated processing for efficient computation
- Support for custom or reanalysis meteorological input
- Modular input configuration
- Optional outputs: SVF, radiation fluxes, shadow maps, etc.
- Compatible with WRF and ERA5 meteorological data

## Installation

Clone the repository and set up the environment:

```bash
conda create -n solweig python=3.10
conda install -c conda-forge gdal
git clone https://github.com/yourusername/solweig-gpu.git
cd solweig-gpu
pip install .


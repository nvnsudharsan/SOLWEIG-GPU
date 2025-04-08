# SOLWEIG-GPU: GPU-Accelerated Thermal Comfort Modeling

This repository provides a Python package and a command-line interface for running the SOLWEIG (Solar and LongWave Environmental Irradiance Geometry) model with GPU acceleration. It supports urban microclimate modeling using high-resolution spatial data to compute thermal comfort indices such as Mean Radiant Temperature (Tmrt) and Universal Thermal Climate Index (UTCI).

## Features

- GPU-accelerated processing for efficient computation
- Support for custom or reanalysis meteorological input
- Modular input configuration
- Optional outputs: SVF, radiation fluxes, shadow maps, etc.
- Compatible with WRF and ERA5 meteorological data

![UTCI for New Delhi](/UTCI_New_Delhi.jpeg)
UTCI for New Delhi, India, generated using SOLWEIG-GPU and visualized using ArcGIS online.

## Installation

Clone the repository and set up the environment:

```bash
conda create -n solweig python=3.10
conda activate solweig
conda install -c conda-forge gdal
git clone https://github.com/yourusername/solweig-gpu.git
cd solweig-gpu
pip install .



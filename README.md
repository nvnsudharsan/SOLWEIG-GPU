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
```
## Usage in Python

```bash
from solweig_gpu import thermal_comfort
thermal_comfort(
    base_path,
    selected_date_str,
    building_dsm_filename='Building_DSM.tif',
    dem_filename='DEM.tif',
    trees_filename='Trees.tif',
    tile_size=3600,
    use_own_met=True,
    start_time=None,
    end_time=None,
    data_source_type=None,
    data_folder=None,
    own_met_file=None,
    save_tmrt=True,
    save_svf=False,
    save_kup=False,
    save_kdown=False,
    save_lup=False,
    save_ldown=False,
    save_shadow=False
)
```

## Usage in Command Line
``` bash
thermal_comfort(
    base_path,
    selected_date_str,
    building_dsm_filename='Building_DSM.tif',
    dem_filename='DEM.tif',
    trees_filename='Trees.tif',
    tile_size=3600,
    use_own_met=True,
    start_time=None,
    end_time=None,
    data_source_type=None,
    data_folder=None,
    own_met_file=None,
    save_tmrt=True,
    save_svf=False,
    save_kup=False,
    save_kdown=False,
    save_lup=False,
    save_ldown=False,
    save_shadow=False
)
```

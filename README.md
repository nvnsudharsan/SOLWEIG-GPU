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
conda install -c conda-forge gdal pytorch
git clone https://github.com/nvnsudharsan/solweig-gpu.git
cd solweig-gpu
pip install .
```
## Usage in Python

```bash
from solweig_gpu import thermal_comfort
thermal_comfort(
    base_path, #base directory where your input data is available
    selected_date_str, #date for which the model should be run
    building_dsm_filename='Building_DSM.tif', #name of building dsm raster
    dem_filename='DEM.tif', #name of dem raster
    trees_filename='Trees.tif', #name of trees raster
    tile_size=3600, #desired tile size, set as per your GPU
    use_own_met=True, #True, if you are using your met file, otherwise False
    own_met_file=None, #met file directory
    start_time=None, #start time of the meteorological data in the format 'YYYY-MM-DD HH:MM:SS'
    end_time=None, #end time of the meteorological data in the format 'YYYY-MM-DD HH:MM:SS'
    data_source_type=None, #'ERA5' or 'WRF' if not using your own met file
    data_folder=None,#Directory of data if ERA5 or WRF
    save_tmrt=True, #True to output Mean Radiant Temperature 
    save_svf=False, #True to output Sky View Factor
    save_kup=False, #True to output Short wave upward
    save_kdown=False, #True to output Short wave downward
    save_lup=False, #True to output Long wave upward
    save_ldown=False, #True to output Long wave downward
    save_shadow=False #True to output Shadow map
)
```

## Usage in Command Line
Type the following on the command line
``` bash
conda activate solweig
thermal_comfort(
    base_path, #base directory where your input data is available
    selected_date_str, #date for which the model should be run
    building_dsm_filename='Building_DSM.tif', #name of building dsm raster
    dem_filename='DEM.tif', #name of dem raster
    trees_filename='Trees.tif', #name of trees raster
    tile_size=3600, #desired tile size, set as per your GPU
    use_own_met=True, #True, if you are using your met file, otherwise False
    own_met_file=None, #met file directory
    start_time=None, #start time of the meteorological data in the format 'YYYY-MM-DD HH:MM:SS'
    end_time=None, #end time of the meteorological data in the format 'YYYY-MM-DD HH:MM:SS'
    data_source_type=None, #'ERA5' or 'WRF' if not using your own met file
    data_folder=None,#Directory of data if ERA5 or WRF
    save_tmrt=True, #True to output Mean Radiant Temperature 
    save_svf=False, #True to output Sky View Factor
    save_kup=False, #True to output Short wave upward
    save_kdown=False, #True to output Short wave downward
    save_lup=False, #True to output Long wave upward
    save_ldown=False, #True to output Long wave downward
    save_shadow=False #True to output Shadow map
)
```

## Usage of GUI
Type the following on the command line
```bash
conda activate solweig
solweig_gpu
```
![GUI](/GUI.png)

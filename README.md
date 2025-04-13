# SOLWEIG-GPU: GPU-Accelerated Thermal Comfort Modeling Framework

This repository provides a Python package and a command-line interface for running the SOLWEIG (Solar and LongWave Environmental Irradiance Geometry) model on CPU as well as with GPU acceleration (if available). It supports urban microclimate modeling by providing a lucid framework to compute sky view factor and thermal comfort indices such as Mean Radiant Temperature (Tmrt) and Universal Thermal Climate Index (UTCI).

## Features
- Can run on CPU
- GPU-accelerated processing for efficient computation (if GPU is available)
- Support for custom or reanalysis meteorological input
- Modular input configuration
- Can calculate: Sky view factor, short- and longwave radiation fluxes, shadow maps, mean radiant temperature (Tmrt) and universal thermal climate index (UTCI)
- Compatible with WRF and ERA5 meteorological data

![UTCI for New Delhi](/UTCI_New_Delhi.jpeg)
UTCI for New Delhi, India, generated using SOLWEIG-GPU and visualized using ArcGIS online.

## Required input datasets    
- Building digital surface model (DSM) which has builings + digital elevation model (DEM)
- DEM
- Tree DSM which has only the height of vegetation (no DEM) 

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

The simplest way is to run the model is using the GUI.

Type the following on the command line
```bash
conda activate solweig
solweig_gpu
```
![GUI](/GUI.png)

In the GUI, select the base path as the folder where you have the input datasets and choose the input Building DSM, DEM and Tree height files. The tile size is the number of pixels in x and y directions. For example, if we set tile size to 700 input rasters are of the size 1000x1200 pixels, this creates create 4 tiles. Alternately if you set it to 1200, there will be only 1 tile created but we recommend splitting into tiles. For the source of meteorology, if you select metfile (.txt), you will have to provide the meteorological forcing text file. Alternatively, you can use ERA5 or wrfout for meteorological forcing. If you are using ERA5, you will need 2 files: instantaneous and accumulated (both downloaded simultaneously from the website). The start time and end time are the UTC times the first and last timestamp in the downloaded ERA-5 data or wrfout files. Note that wrfout and ERA5 need to be hourly in the current implementation. Lastly, you can select which outputs from SOLWEIG you need. If the model run is successful, there will be a folder created named 'Outputs' in the base directory. In this folder, you will have subfolders for each tile. Within each tile folder, you will find the selected outputs. Note that except for sky view factor (SVF), all other rasters will have 24 bands (or time dimension) that are the hourly outputs for selected variables.

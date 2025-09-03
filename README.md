# SOLWEIG-GPU: GPU-Accelerated Thermal Comfort Modeling Framework

<p align="center">
  <img src="https://raw.githubusercontent.com/nvnsudharsan/solweig-gpu/main/Logo.png" alt="SOLWEIG-GPU Logo" width="400"/>
</p>

<p align="center">
  <a href="https://www.repostatus.org/#active"><img src="https://www.repostatus.org/badges/latest/active.svg" alt="Project Status: Active"></a>
  <a href="https://pypi.org/project/solweig-gpu/"><img src="https://badge.fury.io/py/solweig-gpu.svg" alt="PyPI version"></a>
  <!-- <a href="https://anaconda.org/conda-forge/solweig-gpu"><img src="https://anaconda.org/conda-forge/solweig-gpu/badges/version.svg" alt="Conda version"></a> -->
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3+-blue.svg" alt="License: GPL v3 or later"></a>
  <a href="https://pepy.tech/project/solweig-gpu"><img src="https://pepy.tech/badge/solweig-gpu" alt="PyPI Downloads"></a>
</p>

**SOLWEIG-GPU** is a Python package and command-line interface for running standalone SOLWEIG (Solar and LongWave Environmental Irradiance Geometry) model on CPU or GPU (if available). It enables high-resolution urban microclimate modeling by computing key variables such as Sky View Factor (SVF), Mean Radiant Temperature (Tmrt), and the Universal Thermal Climate Index (UTCI).

**SOLWEIG** was originally developed by Dr. Fredrik Lindberg's group. Journal reference: Lindberg, F., Holmer, B. & Thorsson, S. SOLWEIG 1.0 – Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings. *Int J Biometeorol* 52, 697–713 (2008). https://doi.org/10.1007/s00484-008-0162-7

**SOLWEIG GPU** code is an extension of the original **SOLWEIG** Python model that is part of the Urban Multi-scale Environmental Predictor (UMEP). GitHub code: https://github.com/UMEP-dev/UMEP  
UMEP journal reference: Lindberg, F., Grimmond, C.S.B., Gabey, A., Huang, B., Kent, C.W., Sun, T., Theeuwes, N.E., Järvi, L., Ward, H.C., Capel-Timms, I. and Chang, Y., 2018. Urban Multi-scale Environmental Predictor (UMEP): An integrated tool for city-based climate services. *Environmental Modelling & Software*, 99, pp.70-87. https://doi.org/10.1016/j.envsoft.2017.09.020

---

## Features

- CPU and GPU support (automatically uses GPU if available)
- Divides larger areas into tiles based on the selected tile size
- CPU-based computations of wall height and aspect are parallelized across multiple CPUs
- GPU-based computation of SVF, shortwave/longwave radiation, shadows, Tmrt, and UTCI
- Compatible with meteorological data from UMEP, ERA5, and WRF (`wrfout`)

![SOLWEIG-GPU workflow ](https://raw.githubusercontent.com/nvnsudharsan/solweig-gpu/main/solweig_diagram.png)  
*Flowchart of the SOLWEIG-GPU modeling framework*

---

## Required Input Data

- `Building DSM`: Includes both buildings and terrain elevation (e.g., `Building_DSM.tif`)
- `DEM`: Digital Elevation Model excluding buildings (e.g., `DEM.tif`)
- `Tree DSM`: Vegetation height data only (e.g., `Trees.tif`)
- Meteorological forcing:
  - Custom `.txt` file (from UMEP)
  - ERA5 (both instantaneous and accumulated)
  - WRF output NetCDF (`wrfout`)


### ERA5 Variables Required
- 2-meter air temperature  
- 2-meter dew point temperature  
- Surface pressure  
- 10-meter U and V wind components  
- Downwelling shortwave radiation (accumulated)  
- Downwelling longwave radiation (accumulated)  

---

## Output Details

- Output directory: `Outputs/`
- Structure: One folder per tile (e.g., `tile_0_0/`, `tile_0_600/`)
- SVF: Single-band raster
- Other outputs: Multi-band raster (e.g., 24 bands for hourly results)

![UTCI for New Delhi](https://raw.githubusercontent.com/nvnsudharsan/solweig-gpu/main/UTCI_New_Delhi.jpeg)  
*UTCI for New Delhi, India, generated using SOLWEIG-GPU and visualized with ArcGIS Online.*

---

## Installation

```bash
conda create -n solweig python=3.10
conda activate solweig
conda install -c conda-forge gdal cudnn pytorch timezonefinder matplotlib pyqt=5 sip
pip install solweig-gpu

```

---

## Sample Data

Please refer to the sample dataset to familiarize yourself with the expected inputs. Sample data can be found at:  
https://utexas.box.com/s/288e33gak03agrck8v25l7ywj9d6yn87

---

## Python Usage

### Notes on sample data and forcing options

- The `Input_raster` folder in the sample contains the raster files required by SOLWEIG-GPU:
  1. `Building_DSM.tif`
  2. `DEM.tif`
  3. `Trees.tif`
  4. `Landcover.tif` *(optional)*

- SOLWEIG-GPU can be meteorologically forced in three ways:
  1. Using your own meteorological `.txt` file
  2. ERA5 reanalysis
  3. Weather Research and Forecasting (WRF) output files. **Make sure filenames follow one of:**
     - `wrfout_d0x_yyyy-mm-dd_hh_mm_ss` *(preferred; works across operating systems)*
     - `wrfout_d0x_yyyy-mm-dd_hh:mm:ss`
     - `wrfout_d0x_yyyy-mm-dd_hh`

- The `Forcing_data` folder in the sample data contains example data for all forcing methods.

---

### Examples with the provided sample data

#### Example 1: WRF

```python
from solweig_gpu import thermal_comfort

thermal_comfort(
    base_path='/path/to/input',
    selected_date_str='2020-08-13',
    building_dsm_filename='Building_DSM.tif',
    dem_filename='DEM.tif',
    trees_filename='Trees.tif',
    landcover_filename=None,
    tile_size=1000,
    overlap=100,
    use_own_met=False,
    own_met_file='/path/to/met.txt',  # Placeholder as use_own_met=False
    start_time='2020-08-13 06:00:00',
    end_time='2020-08-14 05:00:00',
    data_source_type='wrfout',
    data_folder='/path/to/era5_or_wrfout',
    save_tmrt=False,  # True if you want to save TMRT, likewise below
    save_svf=False,
    save_kup=False,
    save_kdown=False,
    save_lup=False,
    save_ldown=False,
    save_shadow=False
)


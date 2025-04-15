# SOLWEIG-GPU: GPU-Accelerated Thermal Comfort Modeling Framework

**SOLWEIG-GPU** is a Python package and command-line interface for running the SOLWEIG (Solar and LongWave Environmental Irradiance Geometry) model on CPU or GPU (if available). It enables high-resolution urban microclimate modeling by computing key variables such as Sky View Factor (SVF), Mean Radiant Temperature (Tmrt), and the Universal Thermal Climate Index (UTCI).

---

## Features

- CPU and GPU support (automatically uses GPU if available)
- Divides larger areas into tiles based on the selected tile size
- CPU-based computations of wall height and aspect parallelized across multiple CPUs
- GPU-based computation of SVF, shortwave/longwave radiation, shadows, Tmrt, and UTCI
- Compatible with meteorological data from UMEP, ERA5, and WRF (`wrfout`)
- Tile-based processing for large urban domains

![SOLWEIG-GPU workflow ](/solweig_diagram.png)
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

> Please refer to the sample dataset to familiarize yourself with the expected input formats.

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

![UTCI for New Delhi](/UTCI_New_Delhi.jpeg)
*UTCI for New Delhi, India, generated using SOLWEIG-GPU and visualized with ArcGIS Online.*

---

## Installation

```bash
conda create -n solweig python=3.10
conda activate solweig
conda install -c conda-forge gdal pytorch
git clone https://github.com/nvnsudharsan/solweig-gpu.git
cd solweig-gpu
pip install .
```

---

## Python Usage

```python
from solweig_gpu import thermal_comfort

thermal_comfort(
    base_path='/path/to/input',
    selected_date_str='2020-08-13',
    building_dsm_filename='Building_DSM.tif',
    dem_filename='DEM.tif',
    trees_filename='Trees.tif',
    tile_size=3600,
    use_own_met=True,
    own_met_file='/path/to/met.txt',
    start_time='2020-08-13 00:00:00',
    end_time='2020-08-13 23:00:00',
    data_source_type='ERA5',  # or 'WRF'
    data_folder='/path/to/era5_or_wrf',
    save_tmrt=True,
    save_svf=False,
    save_kup=False,
    save_kdown=False,
    save_lup=False,
    save_ldown=False,
    save_shadow=False
)
```

---

## Command-Line Interface (CLI)

```bash
conda activate solweig
thermal_comfort --base_path /path/to/input \
                --selected_date_str 2020-08-13 \
                --building_dsm_filename Building_DSM.tif \
                --dem_filename DEM.tif \
                --trees_filename Trees.tif \
                --tile_size 3600 \
                --use_own_met True \
                --own_met_file /path/to/met.txt \
                --start_time "2020-08-13 00:00:00" \
                --end_time "2020-08-13 23:00:00" \
                --data_source_type ERA5 \
                --data_folder /path/to/era5_or_wrf \
                --save_tmrt True \
                --save_svf False
```

> Tip: Use `--help` to list all CLI options.

---

## GUI Usage

To launch the GUI:
```bash
conda activate solweig
solweig_gpu
```

![GUI](/GUI.png)

### GUI Workflow
1. Select the **base path** containing input datasets.
2. Choose the **Building DSM**, **DEM**, and **Tree DSM** raster files.
3. Set the **tile size** (e.g., 600 or 1200 pixels).
4. Select a **meteorological source** (`metfile`, `ERA5`, or `wrfout`):
   - If `metfile`: Provide a `.txt` file.
   - If `ERA5`: Provide a folder with both instantaneous and accumulated files.
   - If `wrfout`: Provide a folder with WRF output NetCDF files.
5. Set the **start** and **end times** in UTC (`YYYY-MM-DD HH:MM:SS`).
6. Choose which outputs to generate (e.g., Tmrt, UTCI, radiation fluxes).
7. Output will be saved in `Outputs/`, with subfolders for each tile.

---

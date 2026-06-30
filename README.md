# SOLWEIG-GPU: GPU-Accelerated Thermal Comfort Modeling Framework


<p align="center">
  <img src="https://raw.githubusercontent.com/nvnsudharsan/solweig-gpu/main/Logo_solweig.jpg" alt="SOLWEIG Logo" width="400"/>
</p>

<p align="center">
  <a href="https://www.repostatus.org/#active"><img src="https://img.shields.io/badge/Status-Active-%232ecc71.svg" alt="Project Status: Active"></a>
  <a href="https://pypi.org/project/solweig-gpu/"><img src="https://img.shields.io/pypi/v/solweig-gpu.svg?color=%230d6efd" alt="PyPI version"></a>
  <a href="https://solweig-gpu.readthedocs.io/en/latest/?badge=latest"><img src="https://img.shields.io/badge/docs-latest-%235bc0ff.svg" alt="Documentation Status"></a>
  <a href="https://doi.org/10.5281/zenodo.18561860"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18561860-%23ff6b6b.svg" alt="DOI"></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-%230ab5b3.svg" alt="License: GPL v3"></a>
  <a href="https://pepy.tech/projects/solweig-gpu"><img src="https://static.pepy.tech/personalized-badge/solweig-gpu?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=GREEN&left_text=downloads" alt="PyPI Downloads"></a>
  <a href="https://joss.theoj.org/papers/27faa2bf5f6058d981df8b565f8e9a34"><img src="https://joss.theoj.org/papers/27faa2bf5f6058d981df8b565f8e9a34/status.svg"></a>
  <a href="https://github.com/nvnsudharsan/solweig-gpu/actions/workflows/tests.yml"><img src="https://img.shields.io/badge/Tests-Passing-%23ffb703.svg" alt="Tests"></a>
</p>


**SOLWEIG-GPU** is a Python package and command-line interface for running the standalone SOLWEIG (Solar and LongWave Environmental Irradiance Geometry) model on CPU or GPU (if available). It enables high-resolution urban microclimate modeling by computing key variables such as Sky View Factor (SVF), Mean Radiant Temperature (Tmrt), and the Universal Thermal Climate Index (UTCI).

## What is new in Version 2

- Modular code to calculate wall and aspect, sky-view factor, and TMRT/UTCI
- Ability to compute wet bulb globe temperature (WBGT)
- Implements GLIDE-SOL (Zonato et al., 2026) features:
  - Download and process the required input datasets
  - Wind direction based wind-extension coefficient calculation (requires ERA5 data)
  - Compute diagnostic urban heat island intensity (UHII) when ERA5 forcing data is used

**Cite this work as**

1. Kamath, H. G., Sudharsan, N., Singh, M., Wallenberg, N., Lindberg, F., & Niyogi, D. (2026). SOLWEIG-GPU: GPU-Accelerated Thermal Comfort Modeling Framework for Urban Digital Twins. Journal of Open Source Software, 11(118), 9535. https://doi.org/10.21105/joss.09535

2. Zonato, A., Kamath, H.G., Sudharsan, N., Monaco, L., Kittner, J., Wolf, L., Demuzere, M.A., Middel, A., Bechtel, B. and Milelli, M., 2026. GLIDE-SOL: A GPU-accelerated Global Lightweight Infrastructure for Diagnostic Environmental Modeling with SOLWEIG. EGUsphere, 2026, pp.1-30.

**SOLWEIG** was originally developed by Dr. Fredrik Lindberg's group. Journal reference: Lindberg, F., Holmer, B. & Thorsson, S. SOLWEIG 1.0 – Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings. *Int J Biometeorol* 52, 697–713 (2008). https://doi.org/10.1007/s00484-008-0162-7

**SOLWEIG GPU** code is an extension of the original **SOLWEIG** Python model that is part of the Urban Multi-scale Environmental Predictor (UMEP). GitHub code: https://github.com/UMEP-dev/UMEP  
UMEP journal reference: Lindberg, F., Grimmond, C.S.B., Gabey, A., Huang, B., Kent, C.W., Sun, T., Theeuwes, N.E., Järvi, L., Ward, H.C., Capel-Timms, I. and Chang, Y., 2018. Urban Multi-scale Environmental Predictor (UMEP): An integrated tool for city-based climate services. *Environmental Modelling & Software*, 99, pp.70-87. https://doi.org/10.1016/j.envsoft.2017.09.020

---

For detailed documentation, see [Solweig-GPU Documentation](https://solweig-gpu.readthedocs.io/en/latest/index.html)

## Features

- CPU and GPU support (automatically uses GPU if available)
- Divides larger areas into tiles based on the selected tile size
- CPU-based computations of wall height and aspect are parallelized across multiple CPUs
- GPU-based computation of SVF, shortwave/longwave radiation, shadows, Tmrt, and UTCI
- Compatible with meteorological data from UMEP, ERA5, and WRF (`wrfout`)
- Pipeline can be run in stages (`preprocess`, `run_walls_aspect`, `run_utci_tiles`) for subset-of-tiles or reuse; see [documentation](https://solweig-gpu.readthedocs.io/en/latest/) (Developer Guide and API Reference)

![SOLWEIG-GPU workflow ](https://raw.githubusercontent.com/nvnsudharsan/solweig-gpu/main/solweig_diagram.png)  
*Flowchart of the SOLWEIG-GPU modeling framework*

---

## Required Input Data

- `Building DSM`: Includes both buildings and terrain elevation (e.g., `Building_DSM.tif`)
- `DEM`: Digital Elevation Model excluding buildings (e.g., `DEM.tif`)
- `Tree DSM`: Vegetation height data only (e.g., `Trees.tif`)

### Currently tested only for hourly data
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
- Forecasted surface roughness  

---

## Output Details

- Output directory: `output_folder/` (under the directory you pass as `base_path`)
- Structure: One folder per tile (e.g., `0_0/`, `1000_0/`)
- SVF: Single-band raster
- Other outputs: Multi-band raster (e.g., 24 bands for hourly results)

If you need outputs in a different folder, set `base_path` to that directory and pass **complete paths** for the rasters: `building_dsm_filename`, `dem_filename`, `trees_filename`, and `landcover_filename` (optional).

![UTCI for New Delhi](https://raw.githubusercontent.com/nvnsudharsan/solweig-gpu/main/UTCI_New_Delhi.jpeg)  
*UTCI for New Delhi, India, generated using SOLWEIG-GPU and visualized with ArcGIS Online.*

---

## Installation

We recommend using conda environment (please see [documentation](./docs/installation.md))

```bash
conda create -n solweig python=3.10
conda activate solweig
conda install -c conda-forge gdal cudnn pytorch timezonefinder matplotlib sip #cudnn is required only if you are using nvidia GPU
pip install PyQt5
pip install solweig-gpu
#if you have older versions installed
pip install --upgrade solweig-gpu

```
## Testing

Run the test suite with:

```bash
pytest -q
```

With coverage:

```bash
pytest --cov=solweig_gpu --cov-report=term-missing
```

CI runs tests on Linux and macOS across Python 3.10–3.12.


---

## Sample Data

Please refer to the sample dataset to familiarize yourself with the expected inputs. Sample data can be found at:  <a href="https://doi.org/10.5281/zenodo.18561860"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.18561860.svg" alt="DOI"></a>

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

### Examples

#### Data download (optional)

Download the required data for SOLWEIG-GPU from near-globally available urban datasets. Google Earth Engine must be authenticated before this process.

```python
import os
from solweig_gpu import build_inputs

os.environ["EE_PROJECT"] = "your-gee-project-id"  # Your own GEE/GCP project ID

base_path = build_inputs(
    lat=latitude,
    lon=longitude,
    city="City name",
    km_buffer=2,        # Kilometers from the central lat-lon to set the download extent
    km_reduced_lat=1,
    km_reduced_lon=1,
    base_folder="/path/to/save/inputs",
    resolution=2,       # Spatial resolution of the generated rasters in meters
)

print("SOLWEIG input folder:", base_path)
```

#### Compute direction-based wind coefficients (optional)

This requires ERA5 data with the variable `Forecasted surface roughness`.

```python
from solweig_gpu import build_wind_ext_coeff

build_wind_ext_coeff(
    "/path/to/solweig/input",                            # Base path where input rasters are present
    "/path/to/era5/data_stream-oper_stepType-instant.nc"  # ERA5 instantaneous file
)
```

#### Example 1: Modular way of running the model with ERA5

##### Step 1: Preprocess and create inputs in the required format

- The model simulation date is `2020-08-13`.
- The start and end dates provided to the model are `2020-08-13 06:00:00 UTC` and `2020-08-14 05:00:00 UTC`, respectively. UTC to local time conversion is handled internally. For Austin, TX, this corresponds to `2020-08-13 01:00:00` to `2020-08-13 23:00:00` local time.
- The `tile_size` depends on the RAM available on the GPU. A smaller value is safer for lower-memory GPUs, while larger tiles can improve throughput on high-memory GPUs.
- The `overlap` controls the additional pixels used for shadow transfer between neighboring tiles. For example, with `tile_size=1000` and `overlap=100`, the processed tile size becomes `1100 × 1100` pixels.

```python
from solweig_gpu import preprocess

preprocess(
    base_path="/path/to/solweig/input",
    selected_date_str="2020-08-13",
    building_dsm_filename="Building_DSM.tif",
    dem_filename="DEM.tif",
    trees_filename="Trees.tif",
    landcover_filename="Landuse.tif",        # Use None if land cover is not used
    windcoeff_folder="/path/to/solweig/input", # Use None if wind coefficients are not used
    tile_size=400,
    overlap=0,
    use_own_met=False,
    start_time="2020-08-13 06:00:00",
    end_time="2020-08-14 05:00:00",
    data_source_type="ERA5",
    data_folder="/path/to/era5",
    own_met_file=None,
    preprocess_dir="/path/to/solweig/input",
    use_uhi=True,  # Use only with ERA5. Calculates diagnostic urban heat island intensity.
)
```

##### Step 2: Calculate wall height and aspect

```python
from solweig_gpu import run_walls_aspect

run_walls_aspect("/path/to/solweig/input")
```

##### Step 3: Calculate the sky-view factor

```python
from solweig_gpu import calculate_svf

calculate_svf(
    base_path="/path/to/solweig/input",
    patch_option=2,
    overwrite=False,
)
```

##### Step 4: Run the SOLWEIG-GPU model

```python
from solweig_gpu import run_utci_tiles

run_utci_tiles(
    base_path="/path/to/solweig/input",
    preprocess_dir="/path/to/solweig/input",
    selected_date_str="2020-08-13",
    save_tmrt=True,
    save_svf=False,
    save_kup=False,
    save_kdown=False,
    save_lup=False,
    save_ldown=False,
    save_shadow=False,
    save_wbgt=False,
)
```

#### Example 2: Run the model end-to-end with ERA5

```python
from solweig_gpu import thermal_comfort

thermal_comfort(
    base_path="/path/to/solweig/input",
    selected_date_str="2020-08-13",
    building_dsm_filename="Building_DSM.tif",
    dem_filename="DEM.tif",
    trees_filename="Trees.tif",
    landcover_filename="Landuse.tif",  # Use None if land cover is not used
    ERA_5_z0_find=True,  # If True, expects data_stream-oper_stepType-instant.nc in data_folder
    tile_size=400,
    overlap=0,
    use_own_met=False,
    start_time="2020-08-13 06:00:00",
    end_time="2020-08-14 05:00:00",
    data_source_type="ERA5",
    data_folder="/path/to/era5",
    use_uhi=True,
    save_wbgt=True,
)
```

#### Example 3: Run the model end-to-end with WRF

This can also be run in the modular way by following Example 1 and replacing `data_source_type` with `wrfout`.

```python
from solweig_gpu import thermal_comfort

thermal_comfort(
    base_path="/path/to/solweig/input",
    selected_date_str="2020-08-13",
    building_dsm_filename="Building_DSM.tif",
    dem_filename="DEM.tif",
    trees_filename="Trees.tif",
    landcover_filename=None,
    ERA_5_z0_find=False,  # Set True only if data_folder contains ERA5 data_stream-oper_stepType-instant.nc
    tile_size=3600,
    overlap=20,
    use_own_met=False,
    start_time="2020-08-13 06:00:00",
    end_time="2020-08-14 05:00:00",
    data_source_type="wrfout",
    data_folder="/path/to/wrfout/files",
    own_met_file=None,
    use_uhi=False,  # Always keep False when using WRF forcing
    save_tmrt=True,
    save_svf=False,
    save_kup=False,
    save_kdown=False,
    save_lup=False,
    save_ldown=False,
    save_shadow=False,
    save_wbgt=False,
)
```

- The model simulation date is `2020-08-13`.
- The start and end dates provided to the model are `2020-08-13 06:00:00 UTC` and `2020-08-14 05:00:00 UTC`, respectively. These are the start and end times of the WRF output in UTC. In local time, this corresponds to `2020-08-13 01:00:00` to `2020-08-13 23:00:00` for Austin, TX. UTC to local time conversion is handled internally.
- The `tile_size` depends on the RAM available on the GPU. The value can be reduced for lower-memory GPUs.
- The `overlap` controls the additional pixels used for shadow transfer between neighboring tiles. For example, with `tile_size=3600` and `overlap=20`, the processed tile size becomes `3620 × 3620` pixels.
- If `ERA_5_z0_find=True`, SOLWEIG-GPU calculates wind-extension coefficients and expects the ERA5 file `data_stream-oper_stepType-instant.nc` to be available in `data_folder`. If `data_folder` points to WRF output files, keep `data_stream-oper_stepType-instant.nc` in that folder.

#### Example 4: Own File

```python
from solweig_gpu import thermal_comfort

thermal_comfort(
    base_path="/path/to/solweig/input",
    selected_date_str="2020-08-13",
    building_dsm_filename="Building_DSM.tif",
    dem_filename="DEM.tif",
    trees_filename="Trees.tif",
    landcover_filename=None,
    ERA_5_z0_find=False,  # Set True only if data_folder contains ERA5 data_stream-oper_stepType-instant.nc
    tile_size=3600,
    overlap=20,
    use_own_met=True,
    start_time=None,
    end_time=None,
    data_source_type=None,
    data_folder=None,
    own_met_file="/path/to/met.txt",
    use_uhi=False,  # Not recommended with user-provided meteorological files
    save_tmrt=True,
    save_svf=False,
    save_kup=False,
    save_kdown=False,
    save_lup=False,
    save_ldown=False,
    save_shadow=False,
    save_wbgt=False,
)
```

- Use this option when forcing SOLWEIG-GPU with a user-provided meteorological `.txt` file.
- Keep `use_own_met=True` and provide the meteorological file through `own_met_file`.
- Keep `use_uhi=False` for user-provided meteorological files.
- If `ERA_5_z0_find=True`, SOLWEIG-GPU expects the ERA5 file `data_stream-oper_stepType-instant.nc` in `data_folder`. Because this example does not use ERA5 forcing, the safer default is `ERA_5_z0_find=False`.

### Note for Windows Users
On Windows, Python uses the *spawn* start method for new processes: each worker re-imports your script. Without guarding the entry point, a top-level call to `thermal_comfort()` would run again in every child process, causing repeated execution and failures (e.g. `BrokenProcessPool`). Always call `thermal_comfort()` inside a `main()` function and use `if __name__ == "__main__":` (see example below).
```python
from solweig_gpu import thermal_comfort
import multiprocessing as mp

def main():
    thermal_comfort(
        base_path='/path/to/input',
        selected_date_str="2020-08-13",
        building_dsm_filename="Building_DSM.tif",
        dem_filename="DEM.tif",
        trees_filename="Trees.tif",
        landcover_filename="Landcover.tif",
        tile_size=1000,
        overlap=100,
        use_own_met=False,
        own_met_file='/path/to/met.txt',  # placeholder; ignored when use_own_met=False
        start_time="2020-08-13 06:00:00",
        end_time="2020-08-14 05:00:00",
        data_source_type="era5",
        data_folder='/path/to/era5_or_wrfout',
        save_tmrt=False,
        save_svf=False,
        save_kup=False,
        save_kdown=False,
        save_lup=False,
        save_ldown=False,
        save_shadow=False,
    )

if __name__ == "__main__":
    mp.freeze_support()
    main()
```
---

## Command-Line Interface (CLI) 

#### Example using sample ERA5 data on Windows

```bash
conda activate solweig
thermal_comfort --base_path '/path/to/input' ^
                --date '2020-08-13' ^
                --building_dsm 'Building_DSM.tif' ^
                --dem 'DEM.tif' ^
                --trees 'Trees.tif' ^
                --tile_size 1000 ^
                --landcover  'Landcover.tif' ^
                --overlap 100 ^
                --use_own_met False ^
                --data_source_type 'ERA5' ^
                --data_folder '/path/to/era5' ^
                --start '2020-08-13 06:00:00' ^
                --end '2020-08-13 23:00:00' ^
                --save_tmrt True ^
                --save_svf False ^
                --save_kup False ^
                --save_kdown False ^
                --save_lup False ^
                --save_ldown False ^
                --save_shadow False
```

> Tip: Use `--help` to list all CLI options.

---

## GUI Usage

To launch the GUI:
```bash
conda activate solweig
solweig_gpu_gui
```

![GUI](https://raw.githubusercontent.com/nvnsudharsan/solweig-gpu/main/GUI_new.png)

### GUI Workflow
1. Select the **base path** containing input datasets.
2. Choose the **Building DSM**, **DEM**, **Tree DSM**, and **Land cover (optional)** raster files.
3. Set the **tile size** (e.g., 600 or 1200 pixels).
4. Select a **meteorological source** (`metfile`, `ERA5`, or `wrfout`):
   - If `metfile`: Provide a `.txt` file.
   - If `ERA5`: Provide a folder with both instantaneous and accumulated files.
   - If `wrfout`: Provide a folder with WRF output NetCDF files.
5. Set the **start** and **end times** in UTC (`YYYY-MM-DD HH:MM:SS`).
6. Choose which outputs to generate (e.g., Tmrt, UTCI, radiation fluxes).
7. Output will be saved in `output_folder/`, with subfolders for each tile.

---

### Contributing
Please refer to the [documentation](./docs/developer_guide.md)

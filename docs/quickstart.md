# Quick Start Guide

This guide will get you running SOLWEIG-GPU in minutes.

## Basic Workflow

1. Prepare input rasters
2. Prepare meteorological data
3. Run simulation
4. Analyze outputs

## Example 1: Using Your Own Met Data

The simplest way to run SOLWEIG-GPU:

```python
from solweig_gpu import thermal_comfort

thermal_comfort(
    base_path='/path/to/your/data',
    selected_date_str='2020-08-13',
    building_dsm_filename='Building_DSM.tif',
    dem_filename='DEM.tif',
    trees_filename='Trees.tif',
    use_own_met=True,
    own_met_file='met_data.txt'
)
```

### Required Input Files

Place these in your `base_path` directory:

1. **Building_DSM.tif** - Building + terrain heights
2. **DEM.tif** - Digital elevation model (terrain only)
3. **Trees.tif** - Vegetation heights
4. **met_data.txt** - Meteorological forcing data

### Met Data Format

Create a text file with hourly data:

```
yyyy id it imin Ta RH G D I radD W P
2020 213 14 0 25.3 45 650 150 500 0.85 3.2 101.3
2020 213 15 0 26.1 43 700 180 520 0.82 3.5 101.3
```

Where:
- `yyyy` = year
- `id` = day of year
- `it` = hour
- `imin` = minute
- `Ta` = air temperature (°C)
- `RH` = relative humidity (%)
- `G` = global radiation (W/m²)
- `D` = diffuse radiation (W/m²)
- `I` = direct radiation (W/m²)
- `radD` = diffuse ratio
- `W` = wind speed (m/s)
- `P` = pressure (kPa)

## Example 2: Using ERA5 Data

Download ERA5 from the Climate Data Store and run:

```python
from solweig_gpu import thermal_comfort

thermal_comfort(
    base_path='/path/to/data',
    selected_date_str='2020-08-13',
    use_own_met=False,
    data_source_type='ERA5',
    data_folder='/path/to/era5/files',
    start_time='2020-08-13 00:00:00',
    end_time='2020-08-13 23:00:00'
)
```

## Example 3: Using WRF Output

```python
from solweig_gpu import thermal_comfort

thermal_comfort(
    base_path='/path/to/data',
    selected_date_str='2020-08-13',
    use_own_met=False,
    data_source_type='wrfout',
    data_folder='/path/to/wrfout/files',
    start_time='2020-08-13 00:00:00',
    end_time='2020-08-13 23:00:00'
)
```

## Command-Line Usage

```bash
# Using own met data
thermal_comfort \
    --base_path /path/to/data \
    --date 2020-08-13 \
    --tile_size 1000 \
    --use_own_met \
    --met_file met_data.txt

# Using ERA5
thermal_comfort \
    --base_path /path/to/data \
    --date 2020-08-13 \
    --data_source ERA5 \
    --data_folder /path/to/era5 \
    --start_time "2020-08-13 00:00:00" \
    --end_time "2020-08-13 23:00:00"
```

## Configuration Options

### Tile Size

Adjust based on GPU memory:

```python
thermal_comfort(
    base_path='/path/to/data',
    selected_date_str='2020-08-13',
    tile_size=1000,  # Smaller = less memory, more tiles
    overlap=100,     # Overlap for shadow continuity
    ...
)
```

**Guidelines:**
- 8GB GPU: `tile_size=1000-2000`
- 16GB GPU: `tile_size=2000-4000`
- 32GB GPU: `tile_size=4000+`

### Output Options

```python
thermal_comfort(
    base_path='/path/to/data',
    selected_date_str='2020-08-13',
    save_tmrt=True,      # Mean radiant temperature
    save_svf=True,       # Sky view factor
    save_kup=True,       # Upward shortwave
    save_kdown=True,     # Downward shortwave
    save_lup=True,       # Upward longwave
    save_ldown=True,     # Downward longwave
    save_shadow=True,    # Shadow maps
    ...
)
```

## Output Files

Results are saved in `{base_path}/Outputs/{tile_key}/`:

```
Outputs/
├── 0_0/
│   ├── UTCI_2020-08-13.tif       # Thermal comfort index
│   ├── Tmrt_2020-08-13.tif       # Mean radiant temperature
│   ├── SVF.tif                    # Sky view factor
│   └── ...
├── 1000_0/
│   └── ...
```

Each `.tif` is a multi-band raster (one band per hour).

## Next Steps

- [Detailed Input Data Guide](input_data.md)
- [Meteorological Forcing Options](configuration.md)
- [API Reference](api.rst)
- [Examples Gallery](examples.md)

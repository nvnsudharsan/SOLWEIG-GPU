# Configuration

This guide explains the configuration parameters available in SOLWEIG-GPU and how to optimize them for your specific use case.

## Basic Configuration

### Required Parameters

These parameters must be specified for every simulation.

#### `base_path`

**Type:** String  
**Description:** The base directory containing your input raster data.

```python
base_path = '/path/to/your/input_data'
```

The directory structure should be:

```
base_path/
├── Building_DSM.tif
├── DEM.tif
├── Trees.tif
└── Landcover.tif (optional)
```

#### `selected_date_str`

**Type:** String  
**Format:** `'YYYY-MM-DD'`  
**Description:** The date for which to run the thermal comfort simulation.

```python
selected_date_str = '2020-08-13'
```

!!! Note:
    This is the date in local time for your study area, not UTC. E.g., If you want to simulate an entire day of 2020-08-13 for Austin,Texas in summer (UTC - 5 hours), the selected_date_str should be '2020-08-13'

### Raster Filenames

You can specify custom filenames for your input rasters. If not specified, default filenames are used.

```python
building_dsm_filename = 'Building_DSM.tif'  # Default
dem_filename = 'DEM.tif'                    # Default
trees_filename = 'Trees.tif'                # Default
landcover_filename = None                   # Optional
```

## Tiling Configuration

SOLWEIG-GPU uses a tiling system to process large domains efficiently. These parameters control how the study area is divided.

### `tile_size`

**Type:** Integer  
**Default:** 3600  
**Units:** Pixels  
**Description:** Number of pixels in x and y directions. tile_size of 1000 will create rasters with 1000*1000 pixels (plus the overlap chosen by the user)

```python
tile_size = 1000  # Recommended for most GPUs
```

**Choosing Tile Size:**

The optimal tile size depends on your GPU memory. Here are some guidelines:

| GPU Memory | Recommended Tile Size | Approximate Coverage (1m resolution) |
|------------|----------------------|-------------------------------------|
| 4 GB | 600-800 | 0.36-0.64 km² |
| 8 GB | 1000-1200 | 1.0-1.44 km² |
| 12 GB | 1500-2000 | 2.25-4.0 km² |
| 16 GB+ | 2000-3600 | 4.0-12.96 km² |

!!! Tip: "Finding the Right Tile Size"
    If you encounter out-of-memory errors, reduce the tile size. If processing is slow with plenty of memory available, try increasing it. You may use hit and trial to determine the raster size that works with your GPU.

### `overlap`

**Type:** Integer  
**Default:** 20  
**Units:** Pixels  
**Description:** To consider building the tree shading between adjacent raster tiles.

```python
overlap = 100  
```

The overlap ensures that shadows cast from buildings in one tile can affect adjacent tiles. A larger overlap is needed when:

- Buildings are very tall
- High accuracy is required at tile boundaries

!!! Warning:
    Overlap must be less than tile_size. The actual processed tile will be `tile_size + overlap` pixels.

## Meteorological Data Configuration

### Using Custom Meteorological File

```python
use_own_met = True
own_met_file = '/path/to/your/metfile.txt'
```

When `use_own_met=True`, you must provide the path to your custom meteorological file. The `start_time`, `end_time`, `data_source_type`, and `data_folder` parameters are ignored.

!!! Note:
    It is recomended to create the meteorological file with the UMEP Met Processor tool: <https://umep-docs.readthedocs.io/en/latest/pre-processor/Meteorological%20Data%20MetPreprocessor.html>
    
### Using ERA5 or WRF Data

**WRF:**

```python
use_own_met = False
data_source_type = 'wrfout'
data_folder = '/path/to/wrf_data'
start_time = '2020-08-13 06:00:00'  # UTC
end_time = '2020-08-13 23:00:00'    # UTC
```

**ERA-5:**

```python
use_own_met = False
data_source_type = 'ERA5'
data_folder = '/path/to/era5_data'
start_time = '2020-08-13 06:00:00'  # UTC
end_time = '2020-08-14 05:00:00'    # UTC
```

!!! Important Notes:
   1.  When using ERA5 or WRF data, `start_time` and `end_time` must be in **UTC**. The package will automatically convert to local time based on the geographic location of your study area.
   2.  When using wrfout, the `start_time` and `end_time` should be the first and last time stamps in the wrfout dataset. The package won't compare the selecetd `start_time` and `end_time` to the wrfout file timestamps and automatically fetch the corresponding data.
   3.  When using ERA-5 dataset, the package can find the corresponding data for `start_time` and `end_time`. For example, if ERA-5 data is downloaded from 2020-08-13 00 UTC to 2020-08-14 23 UTC and the model is to be run from 2020-08-13 06 UTC to 2020-08-14 05 UTC, the package can select data from 2020-08-13 06 UTC to 2020-08-14 05 UTC (selected_date_str = '2020-08-13', start_time = '2020-08-13 06:00:00', and end_time = '2020-08-14 05:00:00')
    

## Output Configuration

Control which outputs are saved to disk. All outputs are optional except UTCI, which is always saved.

```python
save_tmrt = True      # Mean Radiant Temperature
save_svf = False      # Sky View Factor
save_kup = False      # Upwelling shortwave radiation
save_kdown = False    # Downwelling shortwave radiation
save_lup = False      # Upwelling longwave radiation
save_ldown = False    # Downwelling longwave radiation
save_shadow = False   # Shadow maps
```

**Output Descriptions:**

 Except for sky view factor, each output is saved as a multi-band GeoTIFF with one time step per band.
    
## Complete Configuration Example

Here's a complete example showing all configuration options:

```python
from solweig_gpu import thermal_comfort

thermal_comfort(
    # Required parameters
    base_path='/data/austin_study',
    selected_date_str='2020-08-13',
    
    # Raster filenames
    building_dsm_filename='Building_DSM.tif',
    dem_filename='DEM.tif',
    trees_filename='Trees.tif',
    landcover_filename='Landcover.tif',
    
    # Tiling configuration
    tile_size=1000,
    overlap=100,
    
    # Meteorological data (ERA5)
    use_own_met=False,
    data_source_type='ERA5',
    data_folder='/data/austin_study/era5',
    start_time='2020-08-13 06:00:00',  # UTC
    end_time='2020-08-13 23:00:00',    # UTC
    
    # Output configuration
    save_tmrt=True,
    save_svf=True,
    save_kup=False,
    save_kdown=False,
    save_lup=False,
    save_ldown=False,
    save_shadow=False
)
```

## Performance Optimization

### GPU vs CPU

The package automatically detects and uses GPU if available. To force CPU execution:

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Memory Management

If you encounter memory issues:

1. **Reduce tile size**: Smaller tiles use less GPU memory
2. **Reduce overlap**: Less overlap means less data to process

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
**Description:** The size of each tile in pixels.

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

!!! tip "Finding the Right Tile Size"
    If you encounter out-of-memory errors, reduce the tile size. If processing is slow with plenty of memory available, try increasing it.

### `overlap`

**Type:** Integer  
**Default:** 20  
**Units:** Pixels  
**Description:** The overlap between adjacent tiles.

```python
overlap = 100  # Recommended for accurate shadow calculations
```

The overlap ensures that shadows cast from buildings in one tile can affect adjacent tiles. A larger overlap is needed when:

- Buildings are very tall
- The sun angle is low (early morning or late afternoon)
- High accuracy is required at tile boundaries

**Recommended Overlap Values:**

| Building Height | Sun Angle | Recommended Overlap |
|----------------|-----------|-------------------|
| < 20m | > 30° | 50-100 pixels |
| 20-50m | > 30° | 100-150 pixels |
| > 50m or low sun | < 30° | 150-200 pixels |

!!! warning
    Overlap must be less than tile_size. The actual processed tile will be `tile_size + overlap` pixels.

## Meteorological Data Configuration

### Using Custom Meteorological File

```python
use_own_met = True
own_met_file = '/path/to/your/metfile.txt'
```

When `use_own_met=True`, you must provide the path to your custom meteorological file. The `start_time`, `end_time`, `data_source_type`, and `data_folder` parameters are ignored.

### Using ERA5 or WRF Data

```python
use_own_met = False
data_source_type = 'ERA5'  # or 'wrfout'
data_folder = '/path/to/era5_or_wrf_data'
start_time = '2020-08-13 06:00:00'  # UTC
end_time = '2020-08-13 23:00:00'    # UTC
```

**Parameters:**

- `data_source_type`: Either `'ERA5'` or `'wrfout'`
- `data_folder`: Directory containing the NetCDF files
- `start_time` and `end_time`: Time range in UTC format `'YYYY-MM-DD HH:MM:SS'`

!!! important "Time Zone"
    When using ERA5 or WRF data, `start_time` and `end_time` must be in **UTC**. The package will automatically convert to local time based on the geographic location of your study area.

## Output Configuration

Control which outputs are saved to disk. All outputs are optional except UTCI, which is always computed.

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

| Parameter | Output | Description | File Size |
|-----------|--------|-------------|-----------|
| `save_tmrt` | Tmrt | Mean Radiant Temperature (°C) | Large |
| `save_svf` | SVF | Sky View Factor (0-1) | Small |
| `save_kup` | Kup | Reflected shortwave radiation (W/m²) | Large |
| `save_kdown` | Kdown | Incoming shortwave radiation (W/m²) | Large |
| `save_lup` | Lup | Emitted longwave radiation (W/m²) | Large |
| `save_ldown` | Ldown | Incoming longwave radiation (W/m²) | Large |
| `save_shadow` | Shadow | Binary shadow maps (0/1) | Medium |

!!! tip "Disk Space"
    Each output is saved as a multi-band GeoTIFF with one band per hour. For a 24-hour simulation of a 1000×1000 pixel tile, expect approximately:
    
    - UTCI/Tmrt: ~100 MB per tile
    - Radiation outputs: ~100 MB each per tile
    - SVF: ~4 MB per tile (single band)
    - Shadow: ~25 MB per tile

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

## Advanced Configuration

### Physical Parameters

Some physical parameters are hardcoded in the package but can be modified by editing `utci_process.py`:

```python
# Surface properties
albedo_b = 0.2      # Building albedo
albedo_g = 0.15     # Ground albedo
ewall = 0.9         # Wall emissivity
eground = 0.95      # Ground emissivity
absK = 0.7          # Shortwave absorption
absL = 0.95         # Longwave absorption

# Human body model
Fside = 0.22        # Side fraction
Fup = 0.06          # Up fraction
Fcyl = 0.28         # Cylinder fraction
```

### Vegetation Parameters

```python
firstdayleaf = 97   # First day with leaves (day of year)
lastdayleaf = 300   # Last day with leaves (day of year)
usevegdem = 1       # Use vegetation DEM (1=yes, 0=no)
```

!!! warning "Modifying Source Code"
    Modifying these parameters requires editing the source code. Future versions may expose these as configuration options.

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
3. **Disable unnecessary outputs**: Only save the outputs you need
4. **Process tiles sequentially**: The package already does this, but ensure you're not running multiple simulations simultaneously

### Processing Time

Typical processing times on a modern GPU (NVIDIA RTX 3080):

| Domain Size | Tile Size | Hours | Time |
|------------|-----------|-------|------|
| 1 km² | 1000 | 24 | ~5 min |
| 4 km² | 1000 | 24 | ~20 min |
| 10 km² | 1000 | 24 | ~50 min |

!!! tip
    Processing time scales roughly linearly with the number of pixels and hours simulated.

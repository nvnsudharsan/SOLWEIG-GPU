# SOLWEIG-GPU Package Analysis

## Package Overview

**SOLWEIG-GPU** is a GPU-accelerated Python package for urban thermal comfort modeling. It computes key microclimate variables including Sky View Factor (SVF), Mean Radiant Temperature (Tmrt), and Universal Thermal Climate Index (UTCI) at high spatial resolution.

## Package Structure

```
solweig_gpu/
├── __init__.py                    # Package initialization, exports thermal_comfort
├── solweig_gpu.py                 # Main entry point, orchestrates the workflow
├── preprocessor.py                # Data preprocessing, tiling, met data extraction
├── walls_aspect.py                # Wall height and aspect calculation (CPU parallel)
├── utci_process.py                # Main computation orchestrator
├── solweig.py                     # Core SOLWEIG radiation model (GPU)
├── shadow.py                      # SVF and shadow calculation (GPU)
├── sun_position.py                # Solar position calculations
├── calculate_utci.py              # UTCI calculation from meteorological data
├── Tgmaps_v1.py                   # Ground temperature mapping
├── cli.py                         # Command-line interface
├── solweig_gpu_gui.py             # PyQt5 graphical user interface
├── solweig_gpu_multi_gpu.py       # Multi-GPU support (placeholder)
└── landcoverclasses_2016a.txt     # Land cover classification data
```

## Module Responsibilities

### 1. **solweig_gpu.py** (Main Entry Point)
- Exports the `thermal_comfort()` function
- Orchestrates the entire workflow:
  1. Calls preprocessor to prepare data and create tiles
  2. Runs wall/aspect calculation in parallel (CPU)
  3. Iterates over tiles and calls `compute_utci()` for each
  4. Manages GPU memory between tiles

### 2. **preprocessor.py** (Data Preparation)
- Validates and checks input rasters (dimensions, CRS, pixel size)
- Creates tiles from large rasters with overlap
- Extracts meteorological data from:
  - Custom text files (UMEP format)
  - ERA5 NetCDF files (instantaneous and accumulated)
  - WRF output files (wrfout)
- Converts UTC to local time using timezone detection
- Generates per-tile meteorological files

### 3. **walls_aspect.py** (Geometry Processing)
- Calculates wall heights from building DSM
- Computes aspect (orientation) of walls
- Uses CPU multiprocessing for parallel computation

### 4. **utci_process.py** (Computation Orchestrator)
- Main computation function: `compute_utci()`
- Loads raster data to GPU tensors
- Calls SVF calculation
- Iterates through time steps:
  - Calculates solar position
  - Computes shadows
  - Runs SOLWEIG radiation model
  - Calculates UTCI
- Saves outputs as multi-band GeoTIFFs

### 5. **solweig.py** (Radiation Model)
- Core SOLWEIG algorithm on GPU
- Computes shortwave and longwave radiation fluxes
- Calculates mean radiant temperature (Tmrt)
- Handles reflections from walls and ground
- Key functions:
  - `Solweig_2022a_calc()`: Main calculation
  - `sunonsurface_2018a()`: Surface radiation
  - `clearnessindex_2013b()`: Atmospheric clearness

### 6. **shadow.py** (SVF and Shadow)
- GPU-accelerated shadow casting
- Sky View Factor (SVF) calculation
- Functions:
  - `svf_calculator()`: Computes SVF
  - `create_patches()`: Creates directional patches for SVF

### 7. **sun_position.py** (Solar Geometry)
- Calculates solar position (azimuth, altitude)
- Extracts meteorological data
- Function: `Solweig_2015a_metdata_noload()`

### 8. **calculate_utci.py** (Thermal Comfort)
- Calculates Universal Thermal Climate Index
- Function: `utci_calculator()`
- Inputs: air temperature, mean radiant temperature, wind speed, relative humidity

### 9. **cli.py** (Command Line Interface)
- Argument parsing for command-line usage
- Validation of input parameters
- Entry point: `thermal_comfort` command

### 10. **solweig_gpu_gui.py** (Graphical Interface)
- PyQt5-based GUI
- File selection dialogs
- Parameter configuration
- Entry point: `solweig_gpu_gui` command

## Key Data Flows

### Input Data
1. **Raster Inputs** (GeoTIFF):
   - Building DSM (buildings + terrain)
   - DEM (terrain only)
   - Tree DSM (vegetation height)
   - Land cover (optional)

2. **Meteorological Forcing**:
   - Custom text file (UMEP format)
   - ERA5 NetCDF files
   - WRF output files

### Processing Pipeline
```
Input Rasters → Validation → Tiling → Wall/Aspect Calculation
                                    ↓
Meteorological Data → Extraction → Per-tile met files
                                    ↓
For each tile:
  Load to GPU → SVF Calculation → For each timestep:
                                     Solar Position → Shadow → Radiation → Tmrt → UTCI
                                    ↓
                                  Save outputs
```

### Output Data
- Multi-band GeoTIFF files (one band per hour)
- Saved per tile in `Outputs/tile_X_Y/`
- Optional outputs:
  - UTCI (always computed)
  - Tmrt (mean radiant temperature)
  - SVF (sky view factor)
  - Kup, Kdown (shortwave radiation)
  - Lup, Ldown (longwave radiation)
  - Shadow maps

## Key Algorithms

### 1. Sky View Factor (SVF)
- Measures fraction of sky visible from each point
- Computed using ray casting in multiple directions
- GPU-accelerated for performance

### 2. Shadow Calculation
- Ray tracing from sun position
- Accounts for buildings and vegetation
- Computed for each time step

### 3. Radiation Model
- Shortwave (solar) radiation:
  - Direct beam
  - Diffuse from sky
  - Reflections from walls and ground
- Longwave (thermal) radiation:
  - Emission from surfaces
  - Atmospheric radiation

### 4. Mean Radiant Temperature (Tmrt)
- Weighted average of all radiant fluxes
- Accounts for human body geometry (cylinder model)
- Key input for thermal comfort indices

### 5. UTCI Calculation
- Universal Thermal Climate Index
- Inputs: air temperature, Tmrt, wind speed, humidity
- Represents "feels like" temperature

## GPU Acceleration

### GPU-Accelerated Components
- SVF calculation
- Shadow casting
- Radiation computations
- Tensor operations in SOLWEIG model

### CPU Components
- Wall height and aspect calculation (parallelized)
- File I/O
- Meteorological data extraction

### Memory Management
- Processes tiles sequentially
- Clears GPU cache between tiles
- Tile size configurable based on GPU memory

## Dependencies

### Core Dependencies
- **PyTorch**: GPU acceleration and tensor operations
- **GDAL**: Geospatial raster I/O
- **NumPy**: Array operations
- **SciPy**: Scientific computing

### Meteorological Data
- **netCDF4**: NetCDF file handling
- **xarray**: Multi-dimensional arrays
- **pandas**: Data manipulation

### Utilities
- **timezonefinder**: Timezone detection
- **pytz**: Timezone conversion
- **shapely**: Geometric operations
- **tqdm**: Progress bars

### GUI
- **PyQt5**: Graphical interface
- **matplotlib**: Plotting

## Configuration Parameters

### Spatial Parameters
- `tile_size`: Size of tiles in pixels (default: 3600)
- `overlap`: Overlap between tiles in pixels (default: 20)

### Temporal Parameters
- `selected_date_str`: Simulation date
- `start_time`: Start time (UTC)
- `end_time`: End time (UTC)

### Physical Parameters (hardcoded in utci_process.py)
- `albedo_b`: Building albedo (0.2)
- `albedo_g`: Ground albedo (0.15)
- `ewall`: Wall emissivity (0.9)
- `eground`: Ground emissivity (0.95)
- `absK`: Shortwave absorption (0.7)
- `absL`: Longwave absorption (0.95)

### Human Body Model
- `Fside`: Side fraction (0.22)
- `Fup`: Up fraction (0.06)
- `Fcyl`: Cylinder fraction (0.28)
- `cyl`: Use cylinder model (True)

### Vegetation Parameters
- `firstdayleaf`: First day of leaf (day 97)
- `lastdayleaf`: Last day of leaf (day 300)
- `usevegdem`: Use vegetation DEM (1)

## Testing Considerations

### Unit Testing Needs
1. **Preprocessor**:
   - Raster validation
   - Tiling logic
   - Meteorological data extraction
   - Timezone conversion

2. **Geometry Calculations**:
   - Wall height computation
   - Aspect calculation

3. **Solar Calculations**:
   - Sun position accuracy
   - Shadow casting

4. **Radiation Model**:
   - Flux calculations
   - Tmrt computation

5. **UTCI Calculation**:
   - Index accuracy
   - Edge cases

### Integration Testing Needs
1. End-to-end workflow with sample data
2. Different meteorological sources (text, ERA5, WRF)
3. GPU vs CPU execution
4. Tile boundary handling
5. Multi-tile scenarios

### Performance Testing Needs
1. GPU memory usage
2. Processing time vs tile size
3. Scaling with domain size

## Documentation Needs

### API Documentation
- Function signatures and parameters
- Return types
- Exceptions raised
- Usage examples

### User Guide
- Installation instructions
- Quick start guide
- Input data preparation
- Output interpretation
- Troubleshooting

### Developer Guide
- Architecture overview
- Module interactions
- Adding new features
- GPU optimization tips

### Examples
- Basic usage (all three met sources)
- Advanced configuration
- Custom land cover
- Output visualization
- Batch processing

## Current Gaps

1. **No existing tests**: Package has no test suite
2. **Limited documentation**: Only README with usage examples
3. **No API reference**: Function parameters not fully documented
4. **No examples directory**: Code examples only in README
5. **No developer guide**: Architecture not documented
6. **Multi-GPU support**: Placeholder file, not implemented

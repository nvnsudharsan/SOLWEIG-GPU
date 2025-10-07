># Examples

This section provides a collection of examples to demonstrate how to use SOLWEIG-GPU for different scenarios.

## Example 1: Using WRF Data

This example shows how to run a simulation using meteorological data from WRF output files.

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
    own_met_file='/path/to/met.txt',  # Placeholder
    start_time='2020-08-13 06:00:00',
    end_time='2020-08-14 05:00:00',
    data_source_type='wrfout',
    data_folder='/path/to/wrfout',
)
```

## Example 2: Using ERA5 Data

This example demonstrates how to use ERA5 reanalysis data for the simulation.

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
    own_met_file='/path/to/met.txt',  # Placeholder
    start_time='2020-08-13 06:00:00',
    end_time='2020-08-13 23:00:00',
    data_source_type='ERA5',
    data_folder='/path/to/era5',
)
```

## Example 3: Using a Custom Meteorological File

This example shows how to use your own meteorological data in the UMEP text file format.

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
    use_own_met=True,
    own_met_file='/path/to/met.txt',
)
```


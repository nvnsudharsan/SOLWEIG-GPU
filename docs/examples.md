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
    start_time='2020-08-13 06:00:00',
    end_time='2020-08-13 23:00:00',
    data_source_type='ERA5',
    data_folder='/path/to/era5',
)
```
## You can download ERA5 as below
```python
import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_dewpoint_temperature",
        "2m_temperature",
        "surface_pressure",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downwards"
    ],
    "year": ["2000"], # change to the desired year
    "month": ["08"], # change to the desired month
    "day": ["13", "14"], # change to the desired date
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [31, -98, 29, -97] #change according to your location
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
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


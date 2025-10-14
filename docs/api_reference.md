># API Reference

This section provides a detailed reference for the functions and classes in the SOLWEIG-GPU package. The primary entry point for running simulations is the `thermal_comfort` function.

## `thermal_comfort()`

This is the main function to run a SOLWEIG-GPU simulation.

```python
from solweig_gpu import thermal_comfort

thermal_comfort(
    base_path,
    selected_date_str,
    building_dsm_filename='Building_DSM.tif',
    dem_filename='DEM.tif',
    trees_filename='Trees.tif',
    landcover_filename: Optional[str] = None, 
    tile_size=3600, 
    overlap = 20,
    use_own_met=True,
    start_time=None, 
    end_time=None, 
    data_source_type=None, 
    data_folder=None,
    own_met_file=None,
    save_tmrt=True,
    save_svf=False,
    save_kup=False,
    save_kdown=False,
    save_lup=False,
    save_ldown=False,
    save_shadow=False
)
```

### Parameters

-   `base_path` (str): The base directory containing the input raster data.
-   `selected_date_str` (str): The date for the simulation in `YYYY-MM-DD` format.
-   `building_dsm_filename` (str): The filename of the Building DSM raster. Defaults to `'Building_DSM.tif'`.
-   `dem_filename` (str): The filename of the DEM raster. Defaults to `'DEM.tif'`.
-   `trees_filename` (str): The filename of the Tree DSM raster. Defaults to `'Trees.tif'`.
-   `landcover_filename` (str, optional): The filename of the land cover raster. Defaults to `None`.
-   `tile_size` (int): The size of the tiles in pixels. Defaults to `3600`.
-   `overlap` (int): The overlap between tiles in pixels. Defaults to `20`.
-   `use_own_met` (bool): Whether to use a custom meteorological file. Defaults to `True`.
-   `start_time` (str, optional): The start time of the simulation in `YYYY-MM-DD HH:MM:SS` format (UTC). Required if `use_own_met` is `False`.
-   `end_time` (str, optional): The end time of the simulation in `YYYY-MM-DD HH:MM:SS` format (UTC). Required if `use_own_met` is `False`.
-   `data_source_type` (str, optional): The type of meteorological data source (`'ERA5'` or `'wrfout'`). Required if `use_own_met` is `False`.
-   `data_folder` (str, optional): The directory containing the meteorological data files. Required if `use_own_met` is `False`.
-   `own_met_file` (str, optional): The path to the custom meteorological file. Required if `use_own_met` is `True`.
-   `save_tmrt` (bool, optional): Whether to save the Mean Radiant Temperature output. Defaults to `True`.
-   `save_svf` (bool, optional): Whether to save the Sky View Factor output. Defaults to `False`.
-   `save_kup` (bool, optional): Whether to save the upwelling shortwave radiation output. Defaults to `False`.
-   `save_kdown` (bool, optional): Whether to save the downwelling shortwave radiation output. Defaults to `False`.
-   `save_lup` (bool, optional): Whether to save the upwelling longwave radiation output. Defaults to `False`.
-   `save_ldown` (bool, optional): Whether to save the downwelling longwave radiation output. Defaults to `False`.
-   `save_shadow` (bool, optional): Whether to save the shadow map output. Defaults to `False`.


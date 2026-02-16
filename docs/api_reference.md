># API Reference

This section provides a detailed reference for the functions and classes in the SOLWEIG-GPU package.

**Entry points:**

- **`thermal_comfort(...)`** – One-shot: runs preprocessing, wall/aspect, and UTCI in one call. Use this for normal runs (CLI and GUI call this).
- **`preprocess(...)`**, **`run_walls_aspect(preprocess_dir)`**, **`run_utci_tiles(...)`** – Staged execution: call these when you need to run preprocessing once, then wall/aspect, then UTCI (optionally for a subset of tiles). See [Developer Guide – Pipeline stages](developer_guide.md#pipeline-stages) and the repository’s [REFACTORING.md](../REFACTORING.md).

---

## `thermal_comfort()`

Main function to run a full SOLWEIG-GPU simulation (preprocess → wall/aspect → UTCI). Same behavior as before the staged API; CLI and GUI are unchanged.

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

-   `base_path` (str): Base directory for `output_folder/` and `processed_inputs/`; also used to resolve relative raster paths. To write outputs elsewhere, set this to that directory and pass complete paths for the raster parameters.
-   `selected_date_str` (str): The date for the simulation in `YYYY-MM-DD` format.
-   `building_dsm_filename` (str): Path or filename of the Building DSM raster (relative to `base_path`, or a complete path). Defaults to `'Building_DSM.tif'`.
-   `dem_filename` (str): Path or filename of the DEM raster. Defaults to `'DEM.tif'`.
-   `trees_filename` (str): Path or filename of the Tree DSM raster. Defaults to `'Trees.tif'`.
-   `landcover_filename` (str, optional): Path or filename of the land cover raster. Defaults to `None`.
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

---

## `preprocess()`

Runs only preprocessing: validates rasters, creates tiles, and prepares metfiles. Use when you want to call the pipeline in stages.

```python
from solweig_gpu import preprocess

preprocess_dir = preprocess(
    base_path="/path/to/data",
    selected_date_str="2020-08-13",
    building_dsm_filename="Building_DSM.tif",
    dem_filename="DEM.tif",
    trees_filename="Trees.tif",
    landcover_filename=None,
    tile_size=1000,
    overlap=100,
    use_own_met=True,
    own_met_file="/path/to/met.txt",
    preprocess_dir=None,  # optional; default is base_path/processed_inputs
)
```

**Returns:** Path to the preprocessing directory (string).

**Parameters:** Same as the first part of `thermal_comfort` (base_path, selected_date_str, raster filenames, tile_size, overlap, met options). Optional `preprocess_dir` overrides the default output directory.

---

## `run_walls_aspect()`

Computes wall heights and aspects for all tiles in the preprocessing directory. Call after `preprocess()`.

```python
from solweig_gpu import run_walls_aspect

run_walls_aspect(preprocess_dir)
```

**Parameters:**

-   `preprocess_dir` (str): Path returned by `preprocess()` (contains Building_DSM/, DEM/, Trees/, etc.).

**Returns:** None. Writes to `{preprocess_dir}/walls/` and `{preprocess_dir}/aspect/`.

---

## `run_utci_tiles()`

Runs the SOLWEIG/UTCI computation for tiles. Call after `preprocess()` and `run_walls_aspect()`.

```python
from solweig_gpu import run_utci_tiles

run_utci_tiles(
    base_path="/path/to/data",
    preprocess_dir=preprocess_dir,
    selected_date_str="2020-08-13",
    tile_keys=None,       # None = all tiles; or ["0_0", "1000_0"] for a subset
    save_tmrt=True,
    save_svf=False,
    save_kup=False,
    save_kdown=False,
    save_lup=False,
    save_ldown=False,
    save_shadow=False,
)
```

**Parameters:**

-   `base_path` (str): Base directory; `output_folder/` is created under it.
-   `preprocess_dir` (str): Path returned by `preprocess()`.
-   `selected_date_str` (str): Simulation date `YYYY-MM-DD`.
-   `tile_keys` (list, optional): If `None`, process all tiles. If a list (e.g. `["0_0", "1000_0"]`), process only those tile keys.
-   `save_tmrt`, `save_svf`, `save_kup`, `save_kdown`, `save_lup`, `save_ldown`, `save_shadow`: Same as in `thermal_comfort()`.

**Returns:** None. Writes GeoTIFFs to `{base_path}/output_folder/{tile_key}/`.


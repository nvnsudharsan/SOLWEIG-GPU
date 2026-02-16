#SOLWEIG-GPU: GPU-accelerated SOLWEIG model for urban thermal comfort simulation
#Copyright (C) 2022–2025 Harsh Kamath and Naveen Sudharsan

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
from typing import Optional, List

# Lazy imports used below to avoid loading heavy deps at import time.


def preprocess(
    base_path: str,
    selected_date_str: str,
    building_dsm_filename: str = 'Building_DSM.tif',
    dem_filename: str = 'DEM.tif',
    trees_filename: str = 'Trees.tif',
    landcover_filename: Optional[str] = None,
    tile_size: int = 3600,
    overlap: int = 20,
    use_own_met: bool = True,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    data_source_type: Optional[str] = None,
    data_folder: Optional[str] = None,
    own_met_file: Optional[str] = None,
    preprocess_dir: Optional[str] = None,
) -> str:
    """
    Run preprocessing only: validate rasters, create tiles, and prepare metfiles.

    Use this when you want to run preprocessing once and then call
    :func:`run_walls_aspect` and :func:`run_utci_tiles` separately (e.g. for
    multiple dates or a subset of tiles).

    Args:
        base_path: Base directory; used to resolve relative raster paths.
        selected_date_str: Simulation date 'YYYY-MM-DD'.
        building_dsm_filename, dem_filename, trees_filename, landcover_filename:
            Raster paths or filenames (relative to base_path or absolute).
        tile_size: Tile size in pixels.
        overlap: Overlap between tiles in pixels.
        use_own_met: If True, use own_met_file; else use ERA5/WRF (requires
            start_time, end_time, data_source_type, data_folder).
        start_time, end_time: Required for ERA5/WRF (UTC 'YYYY-MM-DD HH:MM:SS').
        data_source_type: 'ERA5' or 'wrfout' when use_own_met is False.
        data_folder: Folder with ERA5/WRF NetCDF files when use_own_met is False.
        own_met_file: Path to custom met file when use_own_met is True.
        preprocess_dir: Directory for preprocessing outputs. Defaults to
            ``{base_path}/processed_inputs``.

    Returns:
        The path to the preprocessing directory (tiles and metfiles written there).
    """
    import os
    from .preprocessor import ppr

    if preprocess_dir is None:
        preprocess_dir = os.path.join(base_path, "processed_inputs")
    os.makedirs(preprocess_dir, exist_ok=True)

    ppr(
        base_path, building_dsm_filename, dem_filename, trees_filename,
        landcover_filename, tile_size, overlap, selected_date_str, use_own_met,
        start_time, end_time, data_source_type, data_folder, own_met_file,
        preprocess_dir=preprocess_dir,
    )
    return preprocess_dir


def create_inputs(
    lat: float,
    lon: float,
    city: Optional[str] = None,
    km_buffer: float = 8.0,
    km_reduced_lat: float = 3.0,
    km_reduced_lon: float = 1.0,
    year_start: int = 2024,
    year_end: int = 2025,
    base_folder: Optional[str] = None,
    resolution: float = 2.0,
) -> str:
    """
    Build static and meteorological inputs for SOLWEIG-GPU at a given location.

    Downloads and processes WorldCover, tree DSM, DEM, LCZ, OSM/GBA vectors,
    builds Building_DSM, Trees, DEM, Landuse rasters and meteorological NetCDF,
    and computes wind coefficient. Requires optional dependencies (e.g. earthengine-api,
    geemap, geopandas, osmnx); install them if you use this step.

    Use the returned path as ``base_path`` for :func:`preprocess` (with raster
    filenames like ``Building_DSM.tif``, ``DEM.tif``, ``Trees.tif``, ``Landuse.tif``
    in that folder) and optionally set ``use_own_met=False`` with the generated
    NetCDF in ``data_folder``.

    Args:
        lat, lon: Center of the area (degrees).
        city: Name for the output folder. If None, derived from reverse geocoding.
        km_buffer: Half-size of initial bounding box in km.
        km_reduced_lat, km_reduced_lon: Shrink (N/S and E/W) from bbox for SOLWEIG in km.
        year_start, year_end: Start/end year for meteorology (inclusive).
        base_folder: Workspace root. Defaults to the create_inputs module default.
        resolution: Reference grid resolution in meters.

    Returns:
        Path to the output directory (e.g. ``{base_folder}/{city}_for_solweig``)
        containing Building_DSM.tif, DEM.tif, Trees.tif, Landuse.tif and met NetCDF.
    """
    from .create_inputs import run_create_inputs

    return run_create_inputs(
        lat=lat,
        lon=lon,
        city=city,
        km_buffer=km_buffer,
        km_reduced_lat=km_reduced_lat,
        km_reduced_lon=km_reduced_lon,
        year_start=year_start,
        year_end=year_end,
        base_folder=base_folder,
        resolution=resolution,
    )


def run_walls_aspect(preprocess_dir: str) -> None:
    """
    Run wall height and aspect calculation for all tiles in the preprocessing directory.

    Call this after :func:`preprocess`. Writes to ``{preprocess_dir}/walls`` and
    ``{preprocess_dir}/aspect``.

    Args:
        preprocess_dir: Path returned by :func:`preprocess` (contains
            Building_DSM/, DEM/, Trees/, etc.).
    """
    import os
    from .walls_aspect import run_parallel_processing

    building_dsm_dir = os.path.join(preprocess_dir, "Building_DSM")
    walls_dir = os.path.join(preprocess_dir, "walls")
    aspect_dir = os.path.join(preprocess_dir, "aspect")
    run_parallel_processing(building_dsm_dir, walls_dir, aspect_dir)


def run_utci_tiles(
    base_path: str,
    preprocess_dir: str,
    selected_date_str: str,
    tile_keys: Optional[List[str]] = None,
    save_tmrt: bool = True,
    save_svf: bool = False,
    save_kup: bool = False,
    save_kdown: bool = False,
    save_lup: bool = False,
    save_ldown: bool = False,
    save_shadow: bool = False,
) -> None:
    """
    Run UTCI (and optional outputs) for tiles in the preprocessing directory.

    Call this after :func:`preprocess` and :func:`run_walls_aspect`. Writes
    GeoTIFFs to ``{base_path}/output_folder/{tile_key}/``.

    Args:
        base_path: Base directory; output_folder is created under this.
        preprocess_dir: Path returned by :func:`preprocess`.
        selected_date_str: Simulation date 'YYYY-MM-DD'.
        tile_keys: If None, process all tiles. If a list (e.g. ``["0_0", "1000_0"]``),
            process only those tile keys.
        save_tmrt, save_svf, save_kup, save_kdown, save_lup, save_ldown, save_shadow:
            Which outputs to save (UTCI is always saved).
    """
    import os
    import numpy as np
    import torch
    from .utci_process import compute_utci, map_files_by_key

    base_output_path = os.path.join(base_path, "output_folder")
    input_met = os.path.join(preprocess_dir, "metfiles")
    building_dsm_dir = os.path.join(preprocess_dir, "Building_DSM")
    tree_dir = os.path.join(preprocess_dir, "Trees")
    dem_dir = os.path.join(preprocess_dir, "DEM")
    landcover_dir = os.path.join(preprocess_dir, "Landcover")
    walls_dir = os.path.join(preprocess_dir, "walls")
    aspect_dir = os.path.join(preprocess_dir, "aspect")

    building_dsm_map = map_files_by_key(building_dsm_dir, ".tif")
    tree_map = map_files_by_key(tree_dir, ".tif")
    dem_map = map_files_by_key(dem_dir, ".tif")
    landcover_map = map_files_by_key(landcover_dir, ".tif") if os.path.isdir(landcover_dir) else {}
    walls_map = map_files_by_key(walls_dir, ".tif")
    aspect_map = map_files_by_key(aspect_dir, ".tif")
    met_map = map_files_by_key(input_met, ".txt")

    common_keys = set(building_dsm_map) & set(tree_map) & set(dem_map) & set(met_map)
    if landcover_map:
        common_keys &= set(landcover_map)

    if tile_keys is not None:
        common_keys = common_keys & set(tile_keys)
        if not common_keys:
            raise ValueError(f"No tiles to run; tile_keys={tile_keys} not found in preprocess_dir")

    def _numeric_key(k: str):
        x, y = k.split("_")
        return (int(x), int(y))

    print("Running Solweig ...")
    for key in sorted(common_keys, key=_numeric_key):
        building_dsm_path = building_dsm_map[key]
        tree_path = tree_map[key]
        dem_path = dem_map[key]
        landcover_path = landcover_map.get(key) if landcover_map else None
        walls_path = walls_map[key]
        aspect_path = aspect_map[key]
        met_file_path = met_map[key]

        output_folder = os.path.join(base_output_path, key)
        os.makedirs(output_folder, exist_ok=True)

        met_file_data = np.loadtxt(met_file_path, skiprows=1, delimiter=' ')

        compute_utci(
            building_dsm_path,
            tree_path,
            dem_path,
            walls_path,
            aspect_path,
            landcover_path,
            met_file_data,
            output_folder,
            key,
            selected_date_str,
            save_tmrt=save_tmrt,
            save_svf=save_svf,
            save_kup=save_kup,
            save_kdown=save_kdown,
            save_lup=save_lup,
            save_ldown=save_ldown,
            save_shadow=save_shadow,
        )
        torch.cuda.empty_cache()


def thermal_comfort(
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
):
    """
    Main function to compute urban thermal comfort using the SOLWEIG-GPU model.
    
    This function orchestrates the complete workflow:
    1. Preprocesses input rasters (tiling, validation)
    2. Processes meteorological data (ERA5, WRF, or custom)
    3. Calculates wall heights and aspects (parallel CPU)
    4. Computes shadows, radiation, and SVF (GPU-accelerated)
    5. Calculates UTCI thermal comfort index
    6. Saves outputs as georeferenced rasters
    
    Args:
        base_path (str): Base directory for output_folder/ and processed_inputs/; also used to resolve relative raster paths. For outputs elsewhere, set this to the desired output directory and pass complete paths for the raster arguments.
        selected_date_str (str): Simulation date in format 'YYYY-MM-DD'
        building_dsm_filename (str): Building+terrain DSM path or filename (relative to base_path). Default: 'Building_DSM.tif'. Can be a complete path if rasters live elsewhere.
        dem_filename (str): DEM path or filename (relative to base_path). Default: 'DEM.tif'
        trees_filename (str): Vegetation DSM path or filename (relative to base_path). Default: 'Trees.tif'
        landcover_filename (str, optional): Land cover raster path or filename. Default: None
        tile_size (int): Tile size in pixels. Default: 3600. Adjust based on GPU RAM.
        overlap (int): Overlap between tiles in pixels. Default: 20. Used for shadow transfer.
        use_own_met (bool): Use custom meteorological file. Default: True
        start_time (str, optional): Start datetime 'YYYY-MM-DD HH:MM:SS' (UTC for ERA5/WRF)
        end_time (str, optional): End datetime 'YYYY-MM-DD HH:MM:SS' (UTC for ERA5/WRF)
        data_source_type (str, optional): 'ERA5' or 'wrfout' if not using own met file
        data_folder (str, optional): Folder containing ERA5/WRF NetCDF files
        own_met_file (str, optional): Path to custom meteorological text file
        save_tmrt (bool): Save mean radiant temperature output. Default: True
        save_svf (bool): Save sky view factor output. Default: False
        save_kup (bool): Save upward shortwave radiation. Default: False
        save_kdown (bool): Save downward shortwave radiation. Default: False
        save_lup (bool): Save upward longwave radiation. Default: False
        save_ldown (bool): Save downward longwave radiation. Default: False
        save_shadow (bool): Save shadow maps. Default: False
    
    Returns:
        None: Outputs are saved to `{base_path}/output_folder/` directory
    
    Output Structure:
        - {base_path}/processed_inputs/ - All preprocessing files
          - Building_DSM/ - Preprocessing tiles
          - DEM/ - Preprocessing tiles
          - Trees/ - Preprocessing tiles
          - metfiles/ - Meteorological files
          - walls/ - Wall height rasters
          - aspect/ - Wall aspect rasters
          - Outfile.nc - Processed NetCDF (if using ERA5/WRF)
        - output_folder/{tile_key}/ - One folder per tile (tile_key e.g. "0_0", "1000_0")
          - UTCI_{tile_key}.tif - Universal Thermal Climate Index (always saved)
          - TMRT_{tile_key}.tif - Mean radiant temperature (if save_tmrt=True)
          - SVF_{tile_key}.tif - Sky view factor (if save_svf=True)
          - Kup_{tile_key}.tif - Upward shortwave (if save_kup=True)
          - Kdown_{tile_key}.tif - Downward shortwave (if save_kdown=True)
          - Lup_{tile_key}.tif - Upward longwave (if save_lup=True)
          - Ldown_{tile_key}.tif - Downward longwave (if save_ldown=True)
          - Shadow_{tile_key}.tif - Shadow maps (if save_shadow=True)
    
    Notes:
        - Automatically uses GPU if available, falls back to CPU
        - Processes tiles in parallel for large domains
        - UTC to local time conversion handled automatically
        - Multi-band rasters: one band per hour
    
    Example:
        >>> from solweig_gpu import thermal_comfort
        >>> thermal_comfort(
        ...     base_path='/path/to/input',
        ...     selected_date_str='2020-08-13',
        ...     tile_size=1000,
        ...     overlap=100,
        ...     use_own_met=True,
        ...     own_met_file='/path/to/met.txt'
        ... )
    
    Raises:
        ValueError: If input rasters have mismatched dimensions, CRS, or pixel sizes
        FileNotFoundError: If the required input files are missing
    """
    preprocess_dir = preprocess(
        base_path=base_path,
        selected_date_str=selected_date_str,
        building_dsm_filename=building_dsm_filename,
        dem_filename=dem_filename,
        trees_filename=trees_filename,
        landcover_filename=landcover_filename,
        tile_size=tile_size,
        overlap=overlap,
        use_own_met=use_own_met,
        start_time=start_time,
        end_time=end_time,
        data_source_type=data_source_type,
        data_folder=data_folder,
        own_met_file=own_met_file,
    )
    run_walls_aspect(preprocess_dir)
    run_utci_tiles(
        base_path=base_path,
        preprocess_dir=preprocess_dir,
        selected_date_str=selected_date_str,
        tile_keys=None,
        save_tmrt=save_tmrt,
        save_svf=save_svf,
        save_kup=save_kup,
        save_kdown=save_kdown,
        save_lup=save_lup,
        save_ldown=save_ldown,
        save_shadow=save_shadow,
    )

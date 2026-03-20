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


from typing import Optional, List

def preprocess(
    base_path: str,
    selected_date_str: str,
    building_dsm_filename: str = 'Building_DSM.tif',
    dem_filename: str = 'DEM.tif',
    trees_filename: str = 'Trees.tif',
    landcover_filename: Optional[str] = None,
    windcoeff_filename: Optional[str] = None,
    tile_size: int = 3600,
    overlap: int = 20,
    use_own_met: bool = True,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    data_source_type: Optional[str] = None,
    data_folder: Optional[str] = None,
    own_met_file: Optional[str] = None,
    preprocess_dir: Optional[str] = None,
    use_uhi: bool = True,
) -> str:
    """
    Run preprocessing only: validate rasters, create tiles, and prepare metfiles.

    Use this when you want to run preprocessing once and then call
    :func:`run_walls_aspect` and :func:`run_utci_tiles` separately.

    Args:
        base_path: Base directory; used to resolve relative raster paths.
        selected_date_str: Simulation date 'YYYY-MM-DD'.
        building_dsm_filename, dem_filename, trees_filename, landcover_filename,
        windcoeff_filename:
            Raster paths or filenames (relative to base_path or absolute).
        tile_size: Tile size in pixels.
        overlap: Overlap between tiles in pixels.
        use_own_met: If True, use own_met_file; else use ERA5/WRF.
        start_time, end_time: Required for ERA5/WRF (UTC 'YYYY-MM-DD HH:MM:SS').
        data_source_type: 'ERA5' or 'wrfout' when use_own_met is False.
        data_folder: Folder with ERA5/WRF NetCDF files when use_own_met is False.
        own_met_file: Path to custom met file when use_own_met is True.
        preprocess_dir: Directory for preprocessing outputs. Defaults to
            ``{base_path}/processed_inputs``.
        use_uhi: If True, use UHI-aware ERA5 processing and write UHI_CYCLE/uhii
            into generated metfiles when available. If False, use standard ERA5
            processing and write uhii = 0.0.

    Returns:
        The path to the preprocessing directory (tiles and metfiles written there).
    """
    import os
    from .preprocessor import ppr

    if preprocess_dir is None:
        preprocess_dir = os.path.join(base_path, "processed_inputs")
    os.makedirs(preprocess_dir, exist_ok=True)

    ppr(
        base_path,
        building_dsm_filename,
        dem_filename,
        trees_filename,
        landcover_filename,
        windcoeff_filename,
        tile_size,
        overlap,
        selected_date_str,
        use_own_met,
        start_time,
        end_time,
        data_source_type,
        data_folder,
        own_met_file,
        preprocess_dir=preprocess_dir,
        use_uhi=use_uhi,
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
    save_wbgt: bool = False,
) -> None:
    """
    Run UTCI (and optional outputs) for tiles in the preprocessing directory.

    Call this after :func:`preprocess` and :func:`run_walls_aspect`. Writes
    GeoTIFFs to ``{base_path}/output_folder/{tile_key}/``.

    Args:
        base_path: Base directory; output_folder is created under this.
        preprocess_dir: Path returned by :func:`preprocess`.
        selected_date_str: Simulation date 'YYYY-MM-DD'.
        tile_keys: If None, process all tiles. If a list, process only those tile keys.
        save_tmrt, save_svf, save_kup, save_kdown, save_lup, save_ldown, save_shadow:
            Which outputs to save (UTCI is always saved).
        save_ta: Save diagnostic Ta field.
        save_wind: Save diagnostic wind field.
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
    windcoeff_dir = os.path.join(preprocess_dir, "WindCoeff")
    walls_dir = os.path.join(preprocess_dir, "walls")
    aspect_dir = os.path.join(preprocess_dir, "aspect")

    building_dsm_map = map_files_by_key(building_dsm_dir, ".tif")
    tree_map = map_files_by_key(tree_dir, ".tif")
    dem_map = map_files_by_key(dem_dir, ".tif")
    landcover_map = map_files_by_key(landcover_dir, ".tif") if os.path.isdir(landcover_dir) else {}
    windcoeff_map = map_files_by_key(windcoeff_dir, ".tif") if os.path.isdir(windcoeff_dir) else {}
    walls_map = map_files_by_key(walls_dir, ".tif")
    aspect_map = map_files_by_key(aspect_dir, ".tif")
    met_map = map_files_by_key(input_met, ".txt")

    common_keys = set(building_dsm_map) & set(tree_map) & set(dem_map) & set(met_map) & set(walls_map) & set(aspect_map)

    if landcover_map:
        common_keys &= set(landcover_map)

    # Do NOT force intersection with windcoeff_map, because windcoeff is optional
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
        windcoeff_path = windcoeff_map.get(key) if windcoeff_map else None
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
            windcoeff_path,
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
            save_wbgt=save_wbgt,
        )
        torch.cuda.empty_cache()


def thermal_comfort(
    base_path,
    selected_date_str,
    building_dsm_filename='Building_DSM.tif',
    dem_filename='DEM.tif',
    trees_filename='Trees.tif',
    landcover_filename: Optional[str] = None,
    windcoeff_filename: Optional[str] = None,
    tile_size=3600,
    overlap=20,
    use_own_met=True,
    start_time=None,
    end_time=None,
    data_source_type=None,
    data_folder=None,
    own_met_file=None,
    use_uhi=True,
    save_tmrt=True,
    save_svf=False,
    save_kup=False,
    save_kdown=False,
    save_lup=False,
    save_ldown=False,
    save_shadow=False,
    save_wbgt=False,
):
    """
    Main function to compute urban thermal comfort using the SOLWEIG-GPU model.

    Args:
        base_path (str): Base directory for outputs and relative raster paths.
        selected_date_str (str): Simulation date in format 'YYYY-MM-DD'
        building_dsm_filename (str): Building+terrain DSM path or filename.
        dem_filename (str): DEM path or filename.
        trees_filename (str): Vegetation DSM path or filename.
        landcover_filename (str, optional): Land cover raster path or filename.
        windcoeff_filename (str, optional): Wind coefficient raster path or filename.
        tile_size (int): Tile size in pixels.
        overlap (int): Overlap between tiles in pixels.
        use_own_met (bool): Use custom meteorological file.
        start_time (str, optional): Start datetime 'YYYY-MM-DD HH:MM:SS'
        end_time (str, optional): End datetime 'YYYY-MM-DD HH:MM:SS'
        data_source_type (str, optional): 'ERA5' or 'wrfout'
        data_folder (str, optional): Folder containing ERA5/WRF NetCDF files
        own_met_file (str, optional): Path to custom meteorological text file
        use_uhi (bool): If True, use UHI-aware ERA5 processing and propagate
            UHI_CYCLE into metfiles. If False, disable it and write uhii = 0.0.
        save_tmrt (bool): Save mean radiant temperature output.
        save_svf (bool): Save sky view factor output.
        save_kup (bool): Save upward shortwave radiation.
        save_kdown (bool): Save downward shortwave radiation.
        save_lup (bool): Save upward longwave radiation.
        save_ldown (bool): Save downward longwave radiation.
        save_shadow (bool): Save shadow maps.
        save_ta (bool): Save diagnostic Ta field.
        save_wind (bool): Save diagnostic wind field.

    Returns:
        None
    """
    preprocess_dir = preprocess(
        base_path=base_path,
        selected_date_str=selected_date_str,
        building_dsm_filename=building_dsm_filename,
        dem_filename=dem_filename,
        trees_filename=trees_filename,
        landcover_filename=landcover_filename,
        windcoeff_filename=windcoeff_filename,
        tile_size=tile_size,
        overlap=overlap,
        use_own_met=use_own_met,
        start_time=start_time,
        end_time=end_time,
        data_source_type=data_source_type,
        data_folder=data_folder,
        own_met_file=own_met_file,
        use_uhi=use_uhi,
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
        save_wbgt=save_wbgt,
    )

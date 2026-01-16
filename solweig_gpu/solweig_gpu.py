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
from typing import Optional

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
        base_path (str): Base directory containing input rasters
        selected_date_str (str): Simulation date in format 'YYYY-MM-DD'
        building_dsm_filename (str): Building+terrain DSM filename. Default: 'Building_DSM.tif'
        dem_filename (str): Digital Elevation Model filename. Default: 'DEM.tif'
        trees_filename (str): Vegetation DSM filename. Default: 'Trees.tif'
        landcover_filename (str, optional): Land cover raster filename. Default: None
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
        None: Outputs are saved to `{base_path}/Outputs/` directory
    
    Output Structure:
        - {base_path}/processed_inputs/ - All preprocessing files
          - Building_DSM/ - Preprocessing tiles
          - DEM/ - Preprocessing tiles
          - Trees/ - Preprocessing tiles
          - metfiles/ - Meteorological files
          - walls/ - Wall height rasters
          - aspect/ - Wall aspect rasters
          - Outfile.nc - Processed NetCDF (if using ERA5/WRF)
        - Outputs/{tile_key}/ - One folder per tile
          - UTCI_{date}.tif - Universal Thermal Climate Index (always saved)
          - Tmrt_{date}.tif - Mean radiant temperature (if save_tmrt=True)
          - SVF.tif - Sky view factor (if save_svf=True)
          - Kup_{date}.tif - Upward shortwave (if save_kup=True)
          - Kdown_{date}.tif - Downward shortwave (if save_kdown=True)
          - Lup_{date}.tif - Upward longwave (if save_lup=True)
          - Ldown_{date}.tif - Downward longwave (if save_ldown=True)
          - Shadow_{date}.tif - Shadow maps (if save_shadow=True)
    
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

    from .preprocessor import ppr
    from .utci_process import compute_utci, map_files_by_key
    from .walls_aspect import run_parallel_processing
    import os
    import numpy as np
    import torch
    # Create preprocessing outputs directory
    preprocess_dir = os.path.join(base_path, "processed_inputs")
    os.makedirs(preprocess_dir, exist_ok=True)

    ppr(
        base_path, building_dsm_filename, dem_filename, trees_filename,
        landcover_filename, tile_size, overlap, selected_date_str, use_own_met,
        start_time, end_time, data_source_type, data_folder, own_met_file,
         preprocess_dir=preprocess_dir
    )

    base_output_path = os.path.join(base_path, "output_folder")
    inputMet = os.path.join(preprocess_dir, "metfiles")
    building_dsm_dir = os.path.join(preprocess_dir, "Building_DSM")
    tree_dir = os.path.join(preprocess_dir, "Trees")
    dem_dir = os.path.join(preprocess_dir, "DEM")
    landcover_dir = os.path.join(preprocess_dir, "Landcover") if landcover_filename is not None else None
    walls_dir = os.path.join(preprocess_dir, "walls")
    aspect_dir = os.path.join(preprocess_dir, "aspect")

    run_parallel_processing(building_dsm_dir, walls_dir, aspect_dir)
    print("Running Solweig ...")

    building_dsm_map = map_files_by_key(building_dsm_dir, ".tif")
    tree_map = map_files_by_key(tree_dir, ".tif")
    dem_map = map_files_by_key(dem_dir, ".tif")
    landcover_map = map_files_by_key(landcover_dir, ".tif") if landcover_dir else {}
    walls_map = map_files_by_key(walls_dir, ".tif")
    aspect_map = map_files_by_key(aspect_dir, ".tif")
    met_map = map_files_by_key(inputMet, ".txt")

    common_keys = set(building_dsm_map) & set(tree_map) & set(dem_map) & set(met_map)
    if landcover_dir:
        common_keys &= set(landcover_map)

    def _numeric_key(k: str):
        """Sort tiles by numeric coordinates (x, y)."""
        x, y = k.split("_")
        return (int(x), int(y))

    for key in sorted(common_keys, key=_numeric_key):

        building_dsm_path = building_dsm_map[key]
        tree_path = tree_map[key]
        dem_path = dem_map[key]
        landcover_path = landcover_map.get(key) if landcover_dir else None
        walls_path = walls_map.get(key)
        aspect_path = aspect_map.get(key)
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
            save_shadow=save_shadow
        )

        # Free GPU memory between tiles
        torch.cuda.empty_cache()

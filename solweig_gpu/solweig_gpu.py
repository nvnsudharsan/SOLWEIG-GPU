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

    from .preprocessor import ppr
    from .utci_process import compute_utci, map_files_by_key
    from .walls_aspect import run_parallel_processing
    import os
    import numpy as np
    import torch

    ppr(
        base_path, building_dsm_filename, dem_filename, trees_filename,
        landcover_filename, tile_size, overlap, selected_date_str, use_own_met,
        start_time, end_time, data_source_type, data_folder, own_met_file
    )

    base_output_path = os.path.join(base_path, "Outputs")
    inputMet = os.path.join(base_path, "metfiles")
    building_dsm_dir = os.path.join(base_path, "Building_DSM")
    tree_dir = os.path.join(base_path, "Trees")
    dem_dir = os.path.join(base_path, "DEM")
    landcover_dir = os.path.join(base_path, "Landcover") if landcover_filename is not None else None
    walls_dir = os.path.join(base_path, "walls")
    aspect_dir = os.path.join(base_path, "aspect")

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

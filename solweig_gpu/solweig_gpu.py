def thermal_comfort(
    base_path,
    selected_date_str,
    building_dsm_filename='Building_DSM.tif',
    dem_filename='DEM.tif',
    trees_filename='Trees.tif',
    landcover_filename: str | None = None ,  
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
    from .utci_process import compute_utci, extract_number_from_filename, get_matching_files
    from .walls_aspect import run_parallel_processing
    import os
    import numpy as np
    import torch

    ppr(base_path, building_dsm_filename, dem_filename, trees_filename,
         landcover_filename, tile_size, overlap, selected_date_str, use_own_met,
         start_time, end_time, data_source_type, data_folder,
         own_met_file)

    base_output_path = os.path.join(base_path, "Outputs")
    inputMet = os.path.join(base_path, "metfiles")
    building_dsm_dir = os.path.join(base_path, "Building_DSM")
    tree_dir = os.path.join(base_path, "Trees")
    dem_dir = os.path.join(base_path, "DEM")
    if landcover_filename is not None:
        landcover_dir = os.path.join(base_path, "Landcover")
    walls_dir = os.path.join(base_path, "walls")
    aspect_dir = os.path.join(base_path, "aspect")

    # Wall and aspect generation
    run_parallel_processing(building_dsm_dir, walls_dir, aspect_dir)
    print("Running Solweig ...")
    # Load file lists
    met_files = get_matching_files(inputMet, ".txt")
    building_dsm_files = get_matching_files(building_dsm_dir, ".tif")
    tree_files = get_matching_files(tree_dir, ".tif")
    dem_files = get_matching_files(dem_dir, ".tif")
    if landcover_filename is not None:
        landcover_files = get_matching_files(landcover_dir, ".tif")
    walls_files = get_matching_files(walls_dir, ".tif")
    aspect_files = get_matching_files(aspect_dir, ".tif")

    # Compute UTCI
    for i in range(len(building_dsm_files)):
        building_dsm_path = os.path.join(building_dsm_dir, building_dsm_files[i])
        tree_path = os.path.join(tree_dir, tree_files[i])
        dem_path = os.path.join(dem_dir, dem_files[i])
        if landcover_filename is not None:
            landcover_path = os.path.join(landcover_dir, landcover_files[i])
        else:
            landcover_path = None
        walls_path = os.path.join(walls_dir, walls_files[i])
        aspect_path = os.path.join(aspect_dir, aspect_files[i])
        number = extract_number_from_filename(building_dsm_files[i])
        output_folder = os.path.join(base_output_path, number)
        os.makedirs(output_folder, exist_ok=True)

        met_file_path = os.path.join(inputMet, met_files[i])
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
            number,
            selected_date_str,
            save_tmrt=save_tmrt,
            save_svf=save_svf,
            save_kup=save_kup,
            save_kdown=save_kdown,
            save_lup=save_lup,
            save_ldown=save_ldown,
            save_shadow=save_shadow
        )
        torch.cuda.empty_cache()

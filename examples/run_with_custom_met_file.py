"""
Example of how to run a SOLWEIG-GPU simulation using a custom meteorological file.
"""

from solweig_gpu import thermal_comfort

# --- Configuration ---

# Path to the directory containing your input rasters (Building DSM, DEM, Trees)
base_path = '/path/to/your/input_rasters'

# Path to your custom meteorological file (UMEP format)
own_met_file = '/path/to/your/met_file.txt'

# Date for the simulation
selected_date = '2020-08-13'

# --- Run Simulation ---

if __name__ == '__main__':
    print("Running SOLWEIG-GPU with custom meteorological file...")
    
    thermal_comfort(
        base_path=base_path,
        selected_date_str=selected_date,
        building_dsm_filename='Building_DSM.tif',
        dem_filename='DEM.tif',
        trees_filename='Trees.tif',
        landcover_filename=None,  # Optional
        tile_size=1000,
        overlap=100,
        use_own_met=True,
        own_met_file=own_met_file,
        save_tmrt=True,
        save_svf=True
    )
    
    print("Simulation complete. Outputs are saved in the 'Outputs' directory.")


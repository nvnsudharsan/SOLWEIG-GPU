"""
Example of how to run a SOLWEIG-GPU simulation using ERA5 data.
"""

from solweig_gpu import thermal_comfort

# --- Configuration ---

# Path to the directory containing your input rasters (Building DSM, DEM, Trees)
base_path = '/path/to/your/input_rasters'

# Path to the directory containing your ERA5 NetCDF files
data_folder = '/path/to/your/era5_data'

# Date for the simulation
selected_date = '2020-08-13'

# Start and end times for the simulation (UTC)
start_time = '2020-08-13 06:00:00'
end_time = '2020-08-13 23:00:00'

# --- Run Simulation ---

if __name__ == '__main__':
    print("Running SOLWEIG-GPU with ERA5 data...")
    
    thermal_comfort(
        base_path=base_path,
        selected_date_str=selected_date,
        building_dsm_filename='Building_DSM.tif',
        dem_filename='DEM.tif',
        trees_filename='Trees.tif',
        landcover_filename=None,  # Optional
        tile_size=1000,
        overlap=100,
        use_own_met=False,
        data_source_type='ERA5',
        data_folder=data_folder,
        start_time=start_time,
        end_time=end_time,
        save_tmrt=True,
        save_svf=True
    )
    
    print("Simulation complete. Outputs are saved in the 'Outputs' directory.")


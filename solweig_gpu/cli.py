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
import argparse
import os
from .solweig_gpu import thermal_comfort
from . import __version__

# Helper for parsing boolean arguments from strings
def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', '1'): return True
    elif v.lower() in ('no', 'false', 'f', '0'): return False
    raise argparse.ArgumentTypeError("Boolean value expected (True/False)")

def main():
    parser = argparse.ArgumentParser(description="Run SOLWEIG with GPU acceleration.")
    parser.add_argument('--version', action='version', version=f'solweig_gpu {__version__}')

    # Required arguments
    parser.add_argument('--base_path', required=True, help='Base directory containing input data')
    parser.add_argument('--date', required=True, help='Date for which thermal comfort is computed (e.g., 2021-07-01)')

    # Raster inputs
    parser.add_argument('--building_dsm', default='Building_DSM.tif', help='Building DSM raster filename')
    parser.add_argument('--dem', default='DEM.tif', help='DEM raster filename')
    parser.add_argument('--trees', default='Trees.tif', help='Trees raster filename')
    parser.add_argument('--landcover', default=None, help='Landcover raster filename (optional)')

    # Tiling config
    parser.add_argument('--tile_size', type=int, default=3600, help='Tile size in pixels (e.g., 100–4000)')
    parser.add_argument('--overlap', type=int, default=20, help='Tile overlap in pixels (less than tile_size)')

    # Meteorological inputs
    parser.add_argument('--use_own_met', type=str2bool, default=True, help='Use your own meteorological file (True/False)')
    parser.add_argument('--own_metfile', default=None, help='Path to your own meteorological file (NetCDF)')
    parser.add_argument('--data_source_type', default=None, help='Meteorological source (e.g., ERA5, WRF)')
    parser.add_argument('--data_folder', default=None, help='Directory containing ERA5/WRF data files')

    # Optional time range (required if using data_source_type)
    parser.add_argument('--start', default=None, help="Start time (e.g., '2020-08-12 00:00:00')")
    parser.add_argument('--end', default=None, help="End time (e.g., '2020-08-12 23:00:00')")

    # Output options
    parser.add_argument('--save_tmrt', type=str2bool, default=False, help='Save mean radiant temperature output')
    parser.add_argument('--save_svf', type=str2bool, default=False, help='Save sky view factor output')
    parser.add_argument('--save_kup', type=str2bool, default=False, help='Save upward shortwave radiation output')
    parser.add_argument('--save_kdown', type=str2bool, default=False, help='Save downward shortwave radiation output')
    parser.add_argument('--save_lup', type=str2bool, default=False, help='Save upward longwave radiation output')
    parser.add_argument('--save_ldown', type=str2bool, default=False, help='Save downward longwave radiation output')
    parser.add_argument('--save_shadow', type=str2bool, default=False, help='Save shadow map output')

    args = parser.parse_args()

    # Validation logic
    if args.use_own_met:
        if not args.own_metfile:
            parser.error("--own_metfile is required when --use_own_met=True")
        if not os.path.isfile(args.own_metfile):
            parser.error(f"File not found: {args.own_metfile}")
    else:
        if not args.data_source_type:
            parser.error("--data_source_type is required when --use_own_met=False")
        if not args.data_folder:
            parser.error("--data_folder is required when --use_own_met=False")
        if not os.path.isdir(args.data_folder):
            parser.error(f"Directory not found: {args.data_folder}")
        if not args.start or not args.end:
            parser.error("--start and --end are required when using --data_source_type")

    # Run main function
    thermal_comfort(
        base_path=args.base_path,
        selected_date_str=args.date,
        building_dsm_filename=args.building_dsm,
        dem_filename=args.dem,
        trees_filename=args.trees,
        landcover_filename=args.landcover,
        tile_size=args.tile_size,
        overlap=args.overlap,
        use_own_met=args.use_own_met,
        own_met_file=args.own_metfile,
        data_source_type=args.data_source_type,
        data_folder=args.data_folder,
        start_time=args.start,
        end_time=args.end,
        save_tmrt=args.save_tmrt,
        save_svf=args.save_svf,
        save_kup=args.save_kup,
        save_kdown=args.save_kdown,
        save_lup=args.save_lup,
        save_ldown=args.save_ldown,
        save_shadow=args.save_shadow
    )

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
    """
    Convert string to boolean for argparse.

    Args:
        v: Input value (str or bool)

    Returns:
        bool: Converted boolean value

    Raises:
        argparse.ArgumentTypeError: If value cannot be converted to boolean
    """
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', '1'): return True
    elif v.lower() in ('no', 'false', 'f', '0'): return False
    raise argparse.ArgumentTypeError("Boolean value expected (True/False)")

def main():
    """
    Command-line interface for SOLWEIG-GPU thermal comfort modeling.

    Parses command-line arguments and runs the thermal_comfort function.
    This is the entry point for the 'thermal_comfort' console script.

    Usage:
        thermal_comfort --base_path /path/to/input --date 2020-08-13 [options]

    For full help:
        thermal_comfort --help
    """
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
    parser.add_argument('--own_metfile', default=None, help='Path to your own meteorological file (UMEP-format text file)')
    parser.add_argument('--data_source_type', default=None, help='Meteorological source (ERA5 or wrfout)')
    parser.add_argument('--data_folder', default=None, help='Directory containing ERA5/WRF data files')

    # Optional time range (required if using data_source_type)
    parser.add_argument('--start', default=None, help="Start time in UTC (e.g., '2020-08-12 00:00:00')")
    parser.add_argument('--end', default=None, help="End time in UTC (e.g., '2020-08-12 23:00:00')")

    # Wind coefficients and UHI (new in v2)
    parser.add_argument('--era5_z0_find', type=str2bool, default=None,
                        help='Compute directional wind-extension coefficients from ERA5 forecast surface roughness; '
                             'requires data_stream-oper_stepType-instant.nc in --data_folder. '
                             'Defaults to True when --data_folder is provided, False otherwise (True/False)')
    parser.add_argument('--use_uhi', type=str2bool, default=True,
                        help='Compute diagnostic urban heat island intensity (ERA5 forcing only; '
                             'set False for WRF or own met file) (True/False)')

    # Output options
    parser.add_argument('--save_tmrt', type=str2bool, default=True, help='Save mean radiant temperature output')
    parser.add_argument('--save_svf', type=str2bool, default=False, help='Save sky view factor output')
    parser.add_argument('--save_kup', type=str2bool, default=False, help='Save upward shortwave radiation output')
    parser.add_argument('--save_kdown', type=str2bool, default=False, help='Save downward shortwave radiation output')
    parser.add_argument('--save_lup', type=str2bool, default=False, help='Save upward longwave radiation output')
    parser.add_argument('--save_ldown', type=str2bool, default=False, help='Save downward longwave radiation output')
    parser.add_argument('--save_shadow', type=str2bool, default=False, help='Save shadow map output')
    parser.add_argument('--save_wbgt', type=str2bool, default=False, help='Save wet bulb globe temperature output')
    parser.add_argument('--save_ta', type=str2bool, default=False, help='Save diagnostic air temperature output')
    parser.add_argument('--save_wind', type=str2bool, default=False, help='Save diagnostic wind speed output')
    parser.add_argument('--band_interleave', type=str2bool, default=False,
                        help='Write output GeoTIFFs band-interleaved, tiled and compressed (fast per-hour reads, half the size); default False')

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

    if args.era5_z0_find is None:
        # Default: compute wind coefficients only when a data folder is available.
        args.era5_z0_find = args.data_folder is not None
    elif args.era5_z0_find and not args.data_folder:
        parser.error("--era5_z0_find=True requires --data_folder containing data_stream-oper_stepType-instant.nc "
                     "(set --era5_z0_find False to run without wind coefficients)")

    # Run main function
    thermal_comfort(
        base_path=args.base_path,
        selected_date_str=args.date,
        building_dsm_filename=args.building_dsm,
        dem_filename=args.dem,
        trees_filename=args.trees,
        landcover_filename=args.landcover,
        ERA_5_z0_find=args.era5_z0_find,
        tile_size=args.tile_size,
        overlap=args.overlap,
        use_own_met=args.use_own_met,
        own_met_file=args.own_metfile,
        data_source_type=args.data_source_type,
        data_folder=args.data_folder,
        start_time=args.start,
        end_time=args.end,
        use_uhi=args.use_uhi,
        save_tmrt=args.save_tmrt,
        save_svf=args.save_svf,
        save_kup=args.save_kup,
        save_kdown=args.save_kdown,
        save_lup=args.save_lup,
        save_ldown=args.save_ldown,
        save_shadow=args.save_shadow,
        save_wbgt=args.save_wbgt,
        save_ta=args.save_ta,
        save_wind=args.save_wind,
        band_interleave=args.band_interleave,
    )

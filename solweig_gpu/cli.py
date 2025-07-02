import argparse
import os
from .solweig_gpu import thermal_comfort
from . import __version__  # assumes __version__ is defined in solweig_gpu/__init__.py

# Boolean string parser
def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', '1'): return True
    elif v.lower() in ('no', 'false', 'f', '0'): return False
    raise argparse.ArgumentTypeError("Boolean value expected (True/False)")

def main():
    parser = argparse.ArgumentParser(description="Run SOLWEIG with GPU acceleration.")
    parser.add_argument('--version', action='version', version=f'solweig_gpu {__version__}')

    # Required
    parser.add_argument('--base_path', required=True, help='Base directory containing input data')
    parser.add_argument('--date', required=True, help='Date for which thermal comfort is computed (e.g., 2021-07-01)')

    # Raster inputs
    parser.add_argument('--building_dsm', default='Building_DSM.tif', help='Building DSM raster')
    parser.add_argument('--dem', default='DEM.tif', help='DEM raster')
    parser.add_argument('--trees', default='Trees.tif', help='Trees raster')
    parser.add_argument('--landcover', default=None, help='Landcover raster (optional)')

    # Tiling config
    parser.add_argument('--tile_size', type=int, default=3600, help='Tile size (100–4000)')
    parser.add_argument('--overlap', type=int, default=20, help='Overlap between tiles')

    # Meteorological inputs
    parser.add_argument('--use_own_met', type=str2bool, default=True, help='Use your own meteorological file (True/False)')
    parser.add_argument('--own_metfile', default=None, help='Path to your own meteorological file')
    parser.add_argument('--data_source_type', default=None, help='Meteorological source type (e.g., ERA5, WRF)')
    parser.add_argument('--data_folder', default=None, help='Folder containing met data if not using own met')

    # Optional time range
    parser.add_argument('--start', default=None, help="Start time (e.g., '2020-08-12 00:00:00')")
    parser.add_argument('--end', default=None, help="End time (e.g., '2020-08-12 23:00:00')")

    # Output flags
    parser.add_argument('--save_tmrt', type=str2bool, default=False, help='Save mean radiant temperature')
    parser.add_argument('--save_svf', type=str2bool, default=False, help='Save sky view factor')
    parser.add_argument('--save_kup', type=str2bool, default=False, help='Save upward shortwave radiation')
    parser.add_argument('--save_kdown', type=str2bool, default=False, help='Save downward shortwave radiation')
    parser.add_argument('--save_lup', type=str2bool, default=False, help='Save upward longwave radiation')
    parser.add_argument('--save_ldown', type=str2bool, default=False, help='Save downward longwave radiation')
    parser.add_argument('--save_shadow', type=str2bool, default=False, help='Save shadow map')

    args = parser.parse_args()

    # Validation
    if args.use_own_met:
        if not args.own_metfile:
            parser.error("--own_metfile is required when --use_own_met=True")
        if not os.path.isfile(args.own_metfile):
            parser.error(f"own_metfile not found: {args.own_metfile}")
    else:
        if not args.data_source_type:
            parser.error("--data_source_type is required when --use_own_met=False")
        if not args.data_folder:
            parser.error("--data_folder is required when --use_own_met=False")
        if not os.path.isdir(args.data_folder):
            parser.error(f"data_folder not found: {args.data_folder}")

    # Call main function
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

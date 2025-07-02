import argparse
import os
from .solweig_gpu import thermal_comfort

# Custom parser for boolean strings
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (True/False)")

def main():
    parser = argparse.ArgumentParser(description="Run SOLWEIG with GPU acceleration.")

    # Required arguments
    parser.add_argument('--base_path', required=True, help='Base directory containing input data')
    parser.add_argument('--date', required=True, help='Date for which thermal comfort is computed (e.g., 2021-07-01)')

    # Raster inputs
    parser.add_argument('--building_dsm', default='Building_DSM.tif', help='Building DSM raster')
    parser.add_argument('--dem', default='DEM.tif', help='DEM raster')
    parser.add_argument('--trees', default='Trees.tif', help='Trees raster')
    parser.add_argument('--landcover', default=None, help='Landcover raster (optional)')

    # Tile processing config
    parser.add_argument('--tile_size', type=int, default=3600, help='Tile size (100–4000)')
    parser.add_argument('--overlap', type=int, default=20, help='Overlap between tiles')

    # Meteorological forcing options
    parser.add_argument('--use_own_met', type=str2bool, default=True, help='Use your own meteorological file (True/False)')
    parser.add_argument('--own_metfile', default=None, help='Path to your own meteorological file (NetCDF format)')
    parser.add_argument('--data_source_type', default=None, help='Source of met data (e.g., ERA5, WRF) if not using own met')
    parser.add_argument('--data_folder', default=None, help='Folder containing met data files if not using own met')

    # Optional time range
    parser.add_argument('--start', default=None, help="Start time (e.g., '2020-08-12 00:00:00')")
    parser.add_argument('--end', default=None, help="End time (e.g., '2020-08-12 23:00:00')")

    # Optional outputs
    parser.add_argument('--save_tmrt', type=str2bool, default=False, help='Save mean radiant temperature')
    parser.add_argument('--save_svf', type=str2bool, default=False, help='Save sky view factor')
    parser.add_argument('--save_kup', type=str2bool, default=False, help='Save upward shortwave radiation')
    parser.add_argument('--save_kdown', type=str2bool, default=False, help='Save downward shortwave radiation')
    parser.add_argument('--save_lup', type=str2bool, default=False, help='Save upward longwave radiation')
    parser.add_argument('--save_ldown', type=str2bool, default=False, help='Save downward longwave radiation')
    parser.add_argument('--save_shadow', type=str2bool, default=False, help='Save shadow map')

    args = parser.parse_args()

    # Consistency checks
    if args.use_own_met:
        if not args.own_metfile:
            parser.error("You set --use_own_met=True but did not provide --own_metfile.")
        if not os.path.isfile(args.own_metfile):
            parser.error(f"own_metfile not found: {args.own_metfile}")
    else:
        if not args.data_source_type:
            parser.error("You set --use_own_met=False but did not provide --data_source_type.")
        if not args.data_folder:
            parser.error("You set --use_own_met=False but did not provide --data_folder.")
        if not os.path.isdir(args.data_folder):
            parser.error(f"data_folder not found: {args.data_folder}")

    # Run thermal comfort model
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

if __name__ == '__main__':
    main()

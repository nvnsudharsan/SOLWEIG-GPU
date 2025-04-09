import argparse
from .solweig_gpu import thermal_comfort

def main():
    parser = argparse.ArgumentParser(description="Run SOLWEIG with GPU acceleration.")
    parser.add_argument('--base_path', required=True, help='Base directory containing input data')
    parser.add_argument('--date', required=True, help='Date for which thermal comfort is computed (e.g., 2021-07-01)')
    parser.add_argument('--building_dsm', default='Building_DSM.tif', help='Filename of the building DSM raster')
    parser.add_argument('--dem', default='DEM.tif', help='Filename of the DEM raster')
    parser.add_argument('--trees', default='Trees.tif', help='Filename of the trees raster')
    parser.add_argument('--tile_size', type=int, default=3600, help='Tile size for GPU processing (e.g., 100 to 4000)')
    parser.add_argument('--use_own_met', type=bool, default=True, help='Set to True if using your own meteorological file')
    parser.add_argument('--own_metfile', default = None, help='Path to your own meteorological file')
    parser.add_argument('--data_source_type', default=None, help='Specify source of meteorological data (e.g., ERA5 or WRF), if not providing met file')
    parser.add_argument('--data_folder', default = None,help='folder for meteorological data source, if it is ERA5 to WRF')
    parser.add_argument('--start', default = None,help='Start time of the input file(e.g., \'2020-08-12 00:00:00'\)')
    parser.add_argument('--end', default= None, help='End time of the input file(e.g., \'2020-08-14 23:00:00'\)')
    parser.add_argument('--save_tmrt', action='store_true', help='Flag to save mean radiant temperature')
    parser.add_argument('--save_svf', action='store_true', help='Flag to save sky view factor')
    parser.add_argument('--save_kup', action='store_true', help='Flag to save upward shortwave radiation')
    parser.add_argument('--save_kdown', action='store_true', help='Flag to save downward shortwave radiation')
    parser.add_argument('--save_lup', action='store_true', help='Flag to save upward longwave radiation')
    parser.add_argument('--save_ldown', action='store_true', help='Flag to save downward longwave radiation')
    parser.add_argument('--save_shadow', action='store_true', help='Flag to save shadow map')

    args = parser.parse_args()

    thermal_comfort(
        base_path=args.base_path,
        selected_date_str=args.date,
        building_dsm_filename=args.building_dsm,
        dem_filename=args.dem,
        trees_filename=args.trees,
        tile_size=args.tile_size,
        use_own_met=args.use_own_met,
        own_met_file=args.own_metfile,
        start_time=args.start,
        end_time=args.end,
        data_source_type=args.data_source_type,
        data_folder=args.data_folder,
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


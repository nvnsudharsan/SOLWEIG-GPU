import argparse
from .solweig_gpu import thermal_comfort

def main():
    parser = argparse.ArgumentParser(description="Run SOLWEIG with GPU accelaration.")
    parser.add_argument('--base_path', required=True, help='Base directory containing input data')
    parser.add_argument('--building_dsm', default='Building_DSM.tif')
    parser.add_argument('--dem', default='DEM.tif')
    parser.add_argument('--trees', default='Trees.tif')
    parser.add_argument('--tile_size', type=int, default=3600)
    parser.add_argument('--netcdf', default='Control.nc')
    parser.add_argument('--date', default='2020-08-13')
    args = parser.parse_args()

    thermal_comfort(
        base_path=args.base_path,
        building_dsm_filename=args.building_dsm,
        dem_filename=args.dem,
        trees_filename=args.trees,
        tile_size=args.tile_size,
        netcdf_filename=args.netcdf,
        selected_date_str=args.date
    )

if __name__ == '__main__':
    main()

import argparse

from .runtime import thermal_comfort


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SOLWEIG with GPU acceleration.")
    parser.add_argument("--input-data", required=True, help="Folder containing SOLWEIG rasters and metadata")
    parser.add_argument("--output", required=True, help="Destination directory for results")
    parser.add_argument("--start", required=True, help="Simulation start time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--end", required=True, help="Simulation end time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument(
        "--data-source",
        default=None,
        help="Name of the gridded data source (era5, cosmo)",
    )
    parser.add_argument(
        "--one-tile-per-gpu",
        action="store_true",
        help="Force a single tile per GPU regardless of heuristics",
    )
    parser.add_argument("--save-tmrt", action="store_true", help="Persist mean radiant temperature")
    parser.add_argument("--save-svf", action="store_true", help="Persist SVF")
    parser.add_argument("--save-kup", action="store_true", help="Persist upward shortwave flux")
    parser.add_argument("--save-kdown", action="store_true", help="Persist downward shortwave flux")
    parser.add_argument("--save-lup", action="store_true", help="Persist upward longwave flux")
    parser.add_argument("--save-ldown", action="store_true", help="Persist downward longwave flux")
    parser.add_argument("--save-shadow", action="store_true", help="Persist shadow maps")

    args = parser.parse_args()

    thermal_comfort(
        input_data_path=args.input_data,
        start_time=args.start,
        end_time=args.end,
        data_source_type=args.data_source,
        output_path=args.output,
        save_tmrt=args.save_tmrt,
        save_svf=args.save_svf,
        save_kup=args.save_kup,
        save_kdown=args.save_kdown,
        save_lup=args.save_lup,
        save_ldown=args.save_ldown,
        save_shadow=args.save_shadow,
        one_tile_per_gpu=args.one_tile_per_gpu,
    )

if __name__ == '__main__':
    main()

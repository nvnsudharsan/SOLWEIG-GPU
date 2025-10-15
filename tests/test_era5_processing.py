"""
Tests for ERA5 processing: multi-file loading and time coordinate handling.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from osgeo import gdal, osr


def _create_tiny_dem_tile(folder_path: str, filename: str,
                          origin_x: float, origin_y: float,
                          pixel_size: float = 0.01,
                          width: int = 10, height: int = 10) -> str:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    driver = gdal.GetDriverByName('GTiff')
    filepath = os.path.join(folder_path, filename)
    ds = driver.Create(filepath, width, height, 1, gdal.GDT_Float32)
    geotransform = (origin_x, pixel_size, 0.0, origin_y, 0.0, -pixel_size)
    ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    data = np.ones((height, width), dtype=np.float32)
    band.WriteArray(data)
    ds.FlushCache()
    ds = None
    return filepath


class TestERA5Processing(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.era5_dir = os.path.join(self.temp_dir, "era5")
        os.makedirs(self.era5_dir, exist_ok=True)

        # Create synthetic ERA5-like datasets split across two files
        # Grid 2x3 centered over a plausible location
        lat = np.array([29.0, 29.25], dtype=np.float32)
        lon = np.array([77.0, 77.25, 77.5], dtype=np.float32)

        # 24 hourly timesteps on a specific UTC date
        times_all = np.array(
            np.arange('2000-08-13T00:00', '2000-08-14T00:00', dtype='datetime64[h]')
        )
        # Split into two halves to test open_mfdataset
        times_1 = times_all[:12]
        times_2 = times_all[12:]

        def make_vars(time_len):
            shape = (time_len, lat.size, lon.size)
            return {
                't2m': (('valid_time', 'latitude', 'longitude'), 300.0 + np.zeros(shape, dtype=np.float32)),
                'd2m': (('valid_time', 'latitude', 'longitude'), 295.0 + np.zeros(shape, dtype=np.float32)),
                'sp': (('valid_time', 'latitude', 'longitude'), 100000.0 + np.zeros(shape, dtype=np.float32)),
                'u10': (('valid_time', 'latitude', 'longitude'), 2.0 + np.zeros(shape, dtype=np.float32)),
                'v10': (('valid_time', 'latitude', 'longitude'), 1.0 + np.zeros(shape, dtype=np.float32)),
                'ssrd': (('valid_time', 'latitude', 'longitude'), 3600.0 + np.zeros(shape, dtype=np.float32)),
                'strd': (('valid_time', 'latitude', 'longitude'), 3600.0 + np.zeros(shape, dtype=np.float32)),
            }

        ds1 = xr.Dataset(
            data_vars=make_vars(len(times_1)),
            coords={
                'valid_time': times_1,
                'latitude': lat,
                'longitude': lon,
            },
            attrs={'Conventions': 'CF-1.7'}
        )
        ds2 = xr.Dataset(
            data_vars=make_vars(len(times_2)),
            coords={
                'valid_time': times_2,
                'latitude': lat,
                'longitude': lon,
            },
            attrs={'Conventions': 'CF-1.7'}
        )

        self.nc1 = os.path.join(self.era5_dir, "era5_part1.nc")
        self.nc2 = os.path.join(self.era5_dir, "era5_part2.nc")
        ds1.to_netcdf(self.nc1)
        ds2.to_netcdf(self.nc2)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_process_era5_data_mfdataset_valid_time(self):
        from solweig_gpu.preprocessor import process_era5_data

        out_nc = os.path.join(self.temp_dir, "Outfile.nc")
        process_era5_data(
            start_time='2000-08-13 00:00:00',
            end_time='2000-08-13 23:00:00',
            folder_path=self.era5_dir,
            output_file=out_nc,
        )

        self.assertTrue(os.path.exists(out_nc))

        with Dataset(out_nc, 'r') as d:
            # Dimensions
            self.assertIn('time', d.dimensions)
            self.assertIn('lat', d.dimensions)
            self.assertIn('lon', d.dimensions)

            # Variables present
            for var in ['T2', 'PSFC', 'RH2', 'WIND', 'SWDOWN']:
                self.assertIn(var, d.variables)

            # Shapes match expected (24, 2, 3)
            self.assertEqual(d.variables['T2'].shape, (24, 2, 3))
            self.assertEqual(d.variables['PSFC'].shape, (24, 2, 3))
            self.assertEqual(d.variables['RH2'].shape, (24, 2, 3))
            self.assertEqual(d.variables['WIND'].shape, (24, 2, 3))
            self.assertEqual(d.variables['SWDOWN'].shape, (24, 2, 3))

    def test_process_metfiles_utc_local(self):
        from solweig_gpu.preprocessor import process_era5_data, process_metfiles

        # First produce an output NetCDF from synthetic ERA5
        out_nc = os.path.join(self.temp_dir, "Outfile.nc")
        process_era5_data(
            start_time='2000-08-13 00:00:00',
            end_time='2000-08-13 23:00:00',
            folder_path=self.era5_dir,
            output_file=out_nc,
        )

        # Prepare a minimal base_path structure with a DEM tile
        base_path = os.path.join(self.temp_dir, "workspace")
        dem_folder = os.path.join(base_path, "DEM")
        os.makedirs(dem_folder, exist_ok=True)

        # Place the DEM around longitude/latitude matching our synthetic grid
        _create_tiny_dem_tile(
            folder_path=dem_folder,
            filename='DEM_0_0.tif',
            origin_x=77.0,
            origin_y=29.25,
            pixel_size=0.25,
            width=10,
            height=10,
        )

        # Run metfiles processing for the selected date
        process_metfiles(
            netcdf_file=out_nc,
            raster_folder=dem_folder,
            base_path=base_path,
            selected_date_str='2000-08-13',
        )

        # Verify a metfile was created
        met_dir = os.path.join(base_path, 'metfiles')
        self.assertTrue(os.path.isdir(met_dir))
        metfiles = [f for f in os.listdir(met_dir) if f.endswith('.txt')]
        self.assertTrue(len(metfiles) >= 1)

        # Basic header check
        first_met = os.path.join(met_dir, metfiles[0])
        with open(first_met, 'r') as f:
            header = f.readline().strip()
            self.assertIn('iy id it imin', header.replace('  ', ' '))


if __name__ == '__main__':
    unittest.main()



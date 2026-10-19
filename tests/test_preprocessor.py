"""
Test suite for the preprocessor module.

This module tests the data preprocessing functionality including raster validation,
tiling, and meteorological data extraction.
"""

import unittest
import numpy as np
import os
import shutil
import tempfile
from osgeo import gdal, osr
import datetime


class TestRasterValidation(unittest.TestCase):
    """Test raster validation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_rasters = []

    def tearDown(self):
        """Clean up test fixtures."""
        for raster in self.test_rasters:
            if os.path.exists(raster):
                os.remove(raster)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def create_test_raster(self, filename, width=100, height=100, pixel_size=1.0):
        """Create a test raster with specified dimensions."""
        driver = gdal.GetDriverByName('GTiff')
        filepath = os.path.join(self.temp_dir, filename)
        dataset = driver.Create(filepath, width, height, 1, gdal.GDT_Float32)
        
        # Set geotransform
        geotransform = (0, pixel_size, 0, 0, 0, -pixel_size)
        dataset.SetGeoTransform(geotransform)
        
        # Set projection (UTM Zone 14N)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32614)
        dataset.SetProjection(srs.ExportToWkt())
        
        # Write some test data
        band = dataset.GetRasterBand(1)
        data = np.random.rand(height, width).astype(np.float32) * 100
        band.WriteArray(data)
        
        dataset.FlushCache()
        dataset = None
        
        self.test_rasters.append(filepath)
        return filepath

    def test_matching_dimensions(self):
        """Test that rasters with matching dimensions pass validation."""
        from solweig_gpu.preprocessor import check_rasters
        
        raster1 = self.create_test_raster('test1.tif', 100, 100, 1.0)
        raster2 = self.create_test_raster('test2.tif', 100, 100, 1.0)
        
        # Should not raise an exception
        result = check_rasters([raster1, raster2])
        self.assertTrue(result)

    def test_mismatched_dimensions(self):
        """Test that rasters with different dimensions fail validation."""
        from solweig_gpu.preprocessor import check_rasters
        
        raster1 = self.create_test_raster('test1.tif', 100, 100, 1.0)
        raster2 = self.create_test_raster('test2.tif', 200, 200, 1.0)
        
        with self.assertRaises(ValueError):
            check_rasters([raster1, raster2])

    def test_mismatched_pixel_size(self):
        """Test that rasters with different pixel sizes fail validation."""
        from solweig_gpu.preprocessor import check_rasters
        
        raster1 = self.create_test_raster('test1.tif', 100, 100, 1.0)
        raster2 = self.create_test_raster('test2.tif', 100, 100, 2.0)
        
        with self.assertRaises(ValueError):
            check_rasters([raster1, raster2])


class TestDatetimeExtraction(unittest.TestCase):
    """Test datetime extraction from WRF filenames."""

    def test_wrfout_format_underscore(self):
        """Test extraction from wrfout filename with underscores."""
        from solweig_gpu.preprocessor import extract_datetime_strict
        
        filename = 'wrfout_d01_2020-08-13_12_00_00'
        dt, domain = extract_datetime_strict(filename)
        
        self.assertEqual(dt, datetime.datetime(2020, 8, 13, 12, 0, 0))
        self.assertEqual(domain, 1)

    def test_wrfout_format_colon(self):
        """Test extraction from wrfout filename with colons."""
        from solweig_gpu.preprocessor import extract_datetime_strict
        
        filename = 'wrfout_d02_2020-08-13_12:00:00'
        dt, domain = extract_datetime_strict(filename)
        
        self.assertEqual(dt, datetime.datetime(2020, 8, 13, 12, 0, 0))
        self.assertEqual(domain, 2)

    def test_wrfout_format_hour_only(self):
        """Test extraction from wrfout filename with hour only."""
        from solweig_gpu.preprocessor import extract_datetime_strict
        
        filename = 'wrfout_d03_2020-08-13_12'
        dt, domain = extract_datetime_strict(filename)
        
        self.assertEqual(dt, datetime.datetime(2020, 8, 13, 12, 0, 0))
        self.assertEqual(domain, 3)

    def test_invalid_filename(self):
        """Test that invalid filenames raise ValueError."""
        from solweig_gpu.preprocessor import extract_datetime_strict
        
        with self.assertRaises(ValueError):
            extract_datetime_strict('invalid_filename.nc')


class TestPreprocessorTrunkZone(unittest.TestCase):
    """Test that ppr tiles an optional trunk-zone DSM raster."""

    def setUp(self):
        self.base_path = tempfile.mkdtemp()
        for name in ('Building_DSM.tif', 'DEM.tif', 'Trees.tif', 'TrunkZone.tif'):
            self._create_raster(os.path.join(self.base_path, name))

        self.metfile = os.path.join(self.base_path, 'met.txt')
        with open(self.metfile, 'w') as f:
            f.write('iy id it imin Q* QH QE Qs Qf Wind RH Td press rain Kdn snow ldown fcld wuh xsmd lai_hr Kdiff Kdir Wd\n')
            for h in range(24):
                f.write(f"2020 200 {h} 0 -999 -999 -999 -999 -999 1.0 50 20 100 0 0 -999 -999 -999 -999 -999 -999 -999 -999 -999\n")

    def tearDown(self):
        shutil.rmtree(self.base_path, ignore_errors=True)

    def _create_raster(self, path, width=50, height=50, pixel_size=1.0):
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(path, width, height, 1, gdal.GDT_Float32)
        ds.SetGeoTransform((0, pixel_size, 0, 0, 0, -pixel_size))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32614)
        ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).WriteArray(np.ones((height, width), dtype=np.float32) * 10)
        ds.FlushCache()
        ds = None
        return path

    def test_trunkzone_tiles_created(self):
        """A provided trunk-zone DSM should be validated and tiled alongside the DSM."""
        from solweig_gpu.preprocessor import ppr

        preprocess_dir = os.path.join(self.base_path, 'processed_inputs')
        ppr(
            base_path=self.base_path,
            building_dsm_filename='Building_DSM.tif',
            dem_filename='DEM.tif',
            trees_filename='Trees.tif',
            landcover_filename=None,
            windcoeff_filename=None,
            tile_size=3600,
            overlap=20,
            selected_date_str='2020-07-18',
            use_own_met=True,
            own_met_file=self.metfile,
            preprocess_dir=preprocess_dir,
            trunkzone_filename='TrunkZone.tif',
        )

        trunkzone_dir = os.path.join(preprocess_dir, 'TrunkZone')
        self.assertTrue(os.path.isdir(trunkzone_dir))
        tiles = [f for f in os.listdir(trunkzone_dir) if f.endswith('.tif')]
        self.assertTrue(len(tiles) >= 1)

    def test_no_trunkzone_dir_when_not_provided(self):
        """Without a trunk-zone DSM, no TrunkZone tiles should be created."""
        from solweig_gpu.preprocessor import ppr

        preprocess_dir = os.path.join(self.base_path, 'processed_inputs')
        ppr(
            base_path=self.base_path,
            building_dsm_filename='Building_DSM.tif',
            dem_filename='DEM.tif',
            trees_filename='Trees.tif',
            landcover_filename=None,
            windcoeff_filename=None,
            tile_size=3600,
            overlap=20,
            selected_date_str='2020-07-18',
            use_own_met=True,
            own_met_file=self.metfile,
            preprocess_dir=preprocess_dir,
            trunkzone_filename=None,
        )

        self.assertFalse(os.path.isdir(os.path.join(preprocess_dir, 'TrunkZone')))


if __name__ == '__main__':
    unittest.main()

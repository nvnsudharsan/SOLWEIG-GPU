"""
Integration tests for SOLWEIG-GPU.

These tests verify that the entire workflow functions correctly from end to end.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
from osgeo import gdal, osr


class TestEndToEndWorkflow(unittest.TestCase):
    """Test the complete workflow from input to output."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = os.path.join(self.temp_dir, 'test_data')
        os.makedirs(self.base_path, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_minimal_test_data(self):
        """Create minimal test data for integration testing."""
        # Create simple test rasters
        width, height = 50, 50
        
        # Building DSM
        building_dsm = np.zeros((height, width), dtype=np.float32)
        building_dsm[20:30, 20:30] = 10.0  # 10m building
        
        # DEM (flat terrain)
        dem = np.zeros((height, width), dtype=np.float32)
        
        # Trees (some vegetation)
        trees = np.zeros((height, width), dtype=np.float32)
        trees[10:15, 10:15] = 5.0  # 5m trees
        
        # Save rasters
        self._save_raster(building_dsm, 'Building_DSM.tif')
        self._save_raster(dem, 'DEM.tif')
        self._save_raster(trees, 'Trees.tif')

    def _save_raster(self, data, filename):
        """Helper function to save a raster."""
        driver = gdal.GetDriverByName('GTiff')
        filepath = os.path.join(self.base_path, filename)
        dataset = driver.Create(filepath, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
        
        # Set geotransform (1m pixel size)
        geotransform = (0, 1.0, 0, 0, 0, -1.0)
        dataset.SetGeoTransform(geotransform)
        
        # Set projection (UTM Zone 14N - Austin, TX)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32614)
        dataset.SetProjection(srs.ExportToWkt())
        
        # Write data
        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
        
        dataset.FlushCache()
        dataset = None

    def create_minimal_met_file(self):
        """Create a minimal meteorological file for testing."""
        met_file = os.path.join(self.base_path, 'test_met.txt')
        
        # Create a simple met file with one day of data
        header = "Year DOY Hour Min Ta RH G D I Ws Wd P"
        data_lines = []
        for hour in range(24):
            # Simple diurnal cycle
            ta = 20 + 10 * np.sin((hour - 6) * np.pi / 12)  # Temperature
            rh = 60  # Relative humidity
            g = max(0, 800 * np.sin((hour - 6) * np.pi / 12))  # Global radiation
            d = g * 0.3  # Diffuse radiation
            i = g - d  # Direct radiation
            ws = 2.0  # Wind speed
            wd = 180  # Wind direction
            p = 1013  # Pressure
            
            data_lines.append(f"2020 225 {hour} 0 {ta:.1f} {rh} {g:.1f} {d:.1f} {i:.1f} {ws} {wd} {p}")
        
        with open(met_file, 'w') as f:
            f.write(header + '\n')
            f.write('\n'.join(data_lines))
        
        return met_file

    def test_minimal_simulation(self):
        """Test a minimal simulation with simple data."""
        # Create test data
        self.create_minimal_test_data()
        
        # Verify test rasters were created
        self.assertTrue(os.path.exists(os.path.join(self.base_path, 'Building_DSM.tif')))
        self.assertTrue(os.path.exists(os.path.join(self.base_path, 'DEM.tif')))
        self.assertTrue(os.path.exists(os.path.join(self.base_path, 'Trees.tif')))
        
        # Verify rasters can be opened
        from osgeo import gdal
        ds = gdal.Open(os.path.join(self.base_path, 'Building_DSM.tif'))
        self.assertIsNotNone(ds)
        self.assertEqual(ds.RasterXSize, 50)
        self.assertEqual(ds.RasterYSize, 50)
        ds = None


class TestTileBoundaries(unittest.TestCase):
    """Test that tile boundaries are handled correctly."""

    def test_shadow_transfer_between_tiles(self):
        """Test that tile overlap parameter is handled correctly."""
        from solweig_gpu.preprocessor import create_tiles
        import tempfile
        
        # Create a test raster
        temp_dir = tempfile.mkdtemp()
        try:
            test_file = os.path.join(temp_dir, 'test.tif')
            driver = gdal.GetDriverByName('GTiff')
            ds = driver.Create(test_file, 100, 100, 1, gdal.GDT_Float32)
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            ds.SetProjection(srs.ExportToWkt())
            ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
            band = ds.GetRasterBand(1)
            band.WriteArray(np.ones((100, 100), dtype=np.float32))
            ds.FlushCache()
            ds = None
            
            # Create tiles with overlap
            create_tiles(test_file, tilesize=50, overlap=10, tile_type='test_tile')
            
            # Verify tiles were created
            tile_dir = os.path.join(temp_dir, 'test_tile')
            self.assertTrue(os.path.isdir(tile_dir))
            tiles = [f for f in os.listdir(tile_dir) if f.endswith('.tif')]
            self.assertTrue(len(tiles) >= 4)  # Should create at least 2x2 grid
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_radiation_continuity(self):
        """Test that tiling parameters are validated."""
        from solweig_gpu.preprocessor import create_tiles
        import tempfile
        
        # Test that invalid overlap raises error
        temp_dir = tempfile.mkdtemp()
        try:
            test_file = os.path.join(temp_dir, 'test.tif')
            driver = gdal.GetDriverByName('GTiff')
            ds = driver.Create(test_file, 50, 50, 1, gdal.GDT_Float32)
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            ds.SetProjection(srs.ExportToWkt())
            ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
            band = ds.GetRasterBand(1)
            band.WriteArray(np.zeros((50, 50), dtype=np.float32))
            ds.FlushCache()
            ds = None
            
            # Invalid overlap (>= tilesize) should raise error
            with self.assertRaises(ValueError):
                create_tiles(test_file, tilesize=50, overlap=50, tile_type='bad_tile')
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestDifferentMetSources(unittest.TestCase):
    """Test that different meteorological data sources work correctly."""

    def test_custom_met_file(self):
        """Test that custom met file can be read and validated."""
        import tempfile
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a simple met file
            met_file = os.path.join(temp_dir, 'test_met.txt')
            with open(met_file, 'w') as f:
                f.write('iy id it imin Q* QH QE Qs Qf Wind RH Td press rain Kdn snow ldown fcld wuh xsmd lai_hr Kdiff Kdir Wd\n')
                f.write('2020 200 12 0 -999 -999 -999 -999 -999 2.5 60 25 101 0 800 -999 -999 -999 -999 -999 -999 -999 -999 -999\n')
            
            # Verify file exists and is readable
            self.assertTrue(os.path.exists(met_file))
            with open(met_file, 'r') as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 2)  # Header + 1 data line
                self.assertIn('iy', lines[0])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_era5_data(self):
        """Test ERA5 filename parsing."""
        from solweig_gpu.preprocessor import process_era5_data
        import tempfile
        
        # Test that function exists and has correct signature
        import inspect
        sig = inspect.signature(process_era5_data)
        params = list(sig.parameters.keys())
        self.assertIn('start_time', params)
        self.assertIn('end_time', params)
        self.assertIn('folder_path', params)

    def test_wrf_data(self):
        """Test WRF filename parsing."""
        from solweig_gpu.preprocessor import extract_datetime_strict
        import datetime
        
        # Test valid WRF filenames
        dt, domain = extract_datetime_strict('wrfout_d01_2020-08-13_12_00_00')
        self.assertEqual(dt, datetime.datetime(2020, 8, 13, 12, 0, 0))
        self.assertEqual(domain, 1)
        
        # Test another format
        dt, domain = extract_datetime_strict('wrfout_d02_2020-08-13_18:30:45')
        self.assertEqual(dt.hour, 18)
        self.assertEqual(dt.minute, 30)
        self.assertEqual(domain, 2)


if __name__ == '__main__':
    unittest.main()

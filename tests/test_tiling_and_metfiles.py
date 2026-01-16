import os
import shutil
import tempfile
import unittest
import numpy as np
from osgeo import gdal, osr


def _create_raster(path: str, width=64, height=48, pixel_size=1.0, epsg=4326):
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(path, width, height, 1, gdal.GDT_Float32)
    geotransform = (0.0, pixel_size, 0.0, 0.0, 0.0, -pixel_size)
    ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference(); srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.WriteArray(np.random.rand(height, width).astype(np.float32))
    ds.FlushCache(); ds = None
    return path


class TestTilingAndMetfiles(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = self.temp_dir
        # Create base rasters
        self.building = os.path.join(self.base_path, 'Building_DSM.tif')
        self.dem = os.path.join(self.base_path, 'DEM.tif')
        self.trees = os.path.join(self.base_path, 'Trees.tif')
        _create_raster(self.building)
        _create_raster(self.dem)
        _create_raster(self.trees)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_tiles_outputs(self):
        from solweig_gpu.preprocessor import create_tiles
        # Tile size and overlap
        tile_size = 32
        overlap = 8
        # Create tiles for DEM
        create_tiles(self.dem, tile_size, overlap, 'DEM', preprocess_dir=self.base_path)
        out_dir = os.path.join(self.base_path, 'DEM')
        self.assertTrue(os.path.isdir(out_dir))
        # Expect multiple tiles (64x48 raster with 32px tiles = 2x2 grid)
        tiles = [f for f in os.listdir(out_dir) if f.endswith('.tif')]
        self.assertTrue(len(tiles) >= 2, f"Expected at least 2 tiles, got {len(tiles)}")
        
        # Verify tile naming follows pattern DEM_X_Y.tif
        for tile in tiles:
            self.assertTrue(tile.startswith('DEM_'), f"Tile {tile} doesn't start with 'DEM_'")
            self.assertTrue('_' in tile[4:], f"Tile {tile} doesn't follow naming pattern")
        
        # Verify a tile can be opened and has correct properties
        first_tile = os.path.join(out_dir, tiles[0])
        ds = gdal.Open(first_tile)
        self.assertIsNotNone(ds)
        # Tile should be tile_size + overlap or smaller
        self.assertLessEqual(ds.RasterXSize, tile_size + overlap)
        self.assertLessEqual(ds.RasterYSize, tile_size + overlap)
        ds = None

    def test_create_met_files_from_single_file(self):
        from solweig_gpu.preprocessor import create_tiles, create_met_files
        # Prepare tiles of Building_DSM required by create_met_files naming
        create_tiles(self.building, tilesize=64, overlap=0, tile_type='Building_DSM', preprocess_dir=self.base_path)
        
        # Verify Building_DSM tiles were created
        building_dir = os.path.join(self.base_path, 'Building_DSM')
        self.assertTrue(os.path.isdir(building_dir))
        building_tiles = [f for f in os.listdir(building_dir) if f.endswith('.tif')]
        self.assertTrue(len(building_tiles) >= 1, "Building_DSM tiles should exist")
        
        # Create a simple source met file
        met_src = os.path.join(self.base_path, 'source_met.txt')
        with open(met_src, 'w') as f:
            f.write('iy id it imin Q* QH QE Qs Qf Wind RH Td press rain Kdn snow ldown fcld wuh xsmd lai_hr Kdiff Kdir Wd\n')
            for h in range(24):
                f.write(f'2000 1 {h} 0 -999 -999 -999 -999 -999 1.0 50 20 100 0 0 -999 -999 -999 -999 -999 -999 -999 -999 -999\n')
        
        # Run copier
        create_met_files(self.base_path, met_src, self.base_path)
        met_dir = os.path.join(self.base_path, 'metfiles')
        self.assertTrue(os.path.isdir(met_dir))
        
        # Should have one met file per Building_DSM tile
        met_files = [f for f in os.listdir(met_dir) if f.endswith('.txt')]
        self.assertTrue(len(met_files) >= 1, f"Expected at least 1 metfile, got {len(met_files)}")
        self.assertEqual(len(met_files), len(building_tiles), 
                        "Should have one metfile per Building_DSM tile")
        
        # Verify metfile naming follows pattern metfile_X_Y.txt
        for mf in met_files:
            self.assertTrue(mf.startswith('metfile_'), f"Metfile {mf} should start with 'metfile_'")
        
        # Verify metfile content is copied correctly
        first_met = os.path.join(met_dir, met_files[0])
        with open(first_met, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 25)  # Header + 24 hours
            self.assertIn('iy', lines[0])  # Header check
            self.assertIn('2000', lines[1])  # Data check


if __name__ == '__main__':
    unittest.main()



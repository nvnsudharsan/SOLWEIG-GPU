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
        create_tiles(self.dem, tile_size, overlap, 'DEM')
        out_dir = os.path.join(self.base_path, 'DEM')
        self.assertTrue(os.path.isdir(out_dir))
        # Expect multiple tiles
        tiles = [f for f in os.listdir(out_dir) if f.endswith('.tif')]
        self.assertTrue(len(tiles) >= 2)

    def test_create_met_files_from_single_file(self):
        from solweig_gpu.preprocessor import create_tiles, create_met_files
        # Prepare tiles of Building_DSM required by create_met_files naming
        create_tiles(self.building, tilesize=64, overlap=0, tile_type='Building_DSM')
        # Create a simple source met file
        met_src = os.path.join(self.base_path, 'source_met.txt')
        with open(met_src, 'w') as f:
            f.write('iy id it imin Q* QH QE Qs Qf Wind RH Td press rain Kdn snow ldown fcld wuh xsmd lai_hr Kdiff Kdir Wd\n')
            f.write('2000 1 0 0 -999 -999 -999 -999 -999 1.0 50 20 100 0 0 -999 -999 -999 -999 -999 -999 -999 -999 -999\n')
        # Run copier
        create_met_files(self.base_path, met_src)
        met_dir = os.path.join(self.base_path, 'metfiles')
        self.assertTrue(os.path.isdir(met_dir))
        # Should have one met file matching a tile suffix
        files = [f for f in os.listdir(met_dir) if f.endswith('.txt')]
        self.assertTrue(len(files) >= 1)


if __name__ == '__main__':
    unittest.main()



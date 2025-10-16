import os
import shutil
import tempfile
import pytest
import numpy as np
from osgeo import gdal, osr


@pytest.fixture
def temp_workspace():
    d = tempfile.mkdtemp()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def create_raster(path: str, width=10, height=10, pixel_size=1.0, epsg=4326):
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(path, width, height, 1, gdal.GDT_Float32)
    geotransform = (0.0, pixel_size, 0.0, 0.0, 0.0, -pixel_size)
    ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference(); srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.WriteArray(np.zeros((height, width), dtype=np.float32))
    ds.FlushCache(); ds = None
    return path



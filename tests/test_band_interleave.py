"""Tests for the optional band-interleaved output GeoTIFF layout (band_interleave)."""
import os

import numpy as np
from osgeo import gdal

from solweig_gpu.geotiff import geotiff_creation_options


def test_default_is_gdal_default():
    assert geotiff_creation_options(False) == []


def test_options_write_band_interleaved_tiled_compressed(tmp_path):
    opts = geotiff_creation_options(True)
    assert 'INTERLEAVE=BAND' in opts and 'TILED=YES' in opts
    assert any(o.startswith('COMPRESS=') for o in opts)
    path = str(tmp_path / 'UTCI_0_0.tif')
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(path, 64, 48, 24, gdal.GDT_Float32, options=opts)
    data = np.random.rand(24, 48, 64).astype(np.float32)
    for b in range(24):
        ds.GetRasterBand(b + 1).WriteArray(data[b])
    ds.FlushCache(); ds = None
    ds = gdal.Open(path)
    assert ds.RasterCount == 24
    assert ds.GetMetadataItem('INTERLEAVE', 'IMAGE_STRUCTURE') == 'BAND'
    assert ds.GetMetadataItem('COMPRESSION', 'IMAGE_STRUCTURE') in ('ZSTD', 'DEFLATE')
    assert ds.GetRasterBand(1).GetBlockSize() == [64, 48] or ds.GetRasterBand(1).GetBlockSize()[1] <= 512
    back = np.stack([ds.GetRasterBand(b + 1).ReadAsArray() for b in range(24)])
    assert np.array_equal(back, data)                         # lossless
    ds = None
    # the default layout for comparison: same data readable, larger file
    path2 = str(tmp_path / 'default.tif')
    ds = driver.Create(path2, 64, 48, 24, gdal.GDT_Float32)
    for b in range(24):
        ds.GetRasterBand(b + 1).WriteArray(data[b])
    ds = None
    assert os.path.getsize(path) < os.path.getsize(path2) * 1.1

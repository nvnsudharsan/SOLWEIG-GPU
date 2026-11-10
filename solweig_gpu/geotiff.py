"""GeoTIFF creation options for the per-tile outputs (kept free of torch/numba imports so that it can be used and tested on its own)."""
from osgeo import gdal


def geotiff_creation_options(band_interleave=False):
    """
    GDAL creation options for the per-tile output GeoTIFFs (one band per time step).

    GDAL's default is a pixel-interleaved, stripped, uncompressed file: reading ONE band (one hour) of a
    24-band tile forces a read of the whole file, which makes mosaicking many tiles hour by hour slow on
    network file systems. With ``band_interleave=True`` the file is written band-interleaved
    (INTERLEAVE=BAND, each band contiguous), tiled (512 x 512 blocks) and losslessly compressed (ZSTD level 1
    when the GDAL build supports it, DEFLATE otherwise, about half the size). Readers need no change
    (rasterio, GDAL, QGIS, ArcGIS). Default False keeps the previous file layout.
    """
    if not band_interleave:
        return []
    opts = gdal.GetDriverByName('GTiff').GetMetadataItem('DMD_CREATIONOPTIONLIST') or ''
    compress = ['COMPRESS=ZSTD', 'ZSTD_LEVEL=1'] if 'ZSTD' in opts else ['COMPRESS=DEFLATE', 'ZLEVEL=1']
    return ['INTERLEAVE=BAND', 'TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512', 'BIGTIFF=IF_SAFER'] + compress

"""Tests for solweig_gpu.hrrr (HRRR forcing). Offline except the test marked 'network'."""
import os
import datetime as dt

import numpy as np
import pytest

from solweig_gpu import hrrr

IDX_SAMPLE = """1:0:d=2026090906:REFC:entire atmosphere:3 hour fcst:
2:154203:d=2026090906:RETOP:cloud top:3 hour fcst:
3:217660:d=2026090906:TMP:2 m above ground:3 hour fcst:
4:1127913:d=2026090906:DPT:2 m above ground:3 hour fcst:
5:2011220:d=2026090906:PRES:surface:3 hour fcst:
6:2900031:d=2026090906:UGRD:10 m above ground:3 hour fcst:
7:3700122:d=2026090906:VGRD:10 m above ground:3 hour fcst:
8:4500010:d=2026090906:DSWRF:surface:3 hour fcst:
9:5000000:d=2026090906:SFCR:surface:3 hour fcst:
"""


def test_parse_fhours():
    assert hrrr.parse_fhours("0-3") == [0, 1, 2, 3]
    assert hrrr.parse_fhours("0-2,5,7-8") == [0, 1, 2, 5, 7, 8]
    assert hrrr.parse_fhours(4) == [4]


def test_parse_idx_byte_ranges():
    recs = hrrr.parse_idx(IDX_SAMPLE)
    keys = [k for _, _, k in recs]
    for key in hrrr.FIELDS.values():
        assert key in keys
    start, end, key = recs[2]
    assert (start, end, key) == (217660, 1127912, "TMP:2 m above ground")
    assert recs[-1][1] is None   # last record runs to the end of the file


def test_file_url():
    c = dt.datetime(2026, 9, 9, 6)
    assert hrrr.file_url(c, 3) == "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20260909/conus/hrrr.t06z.wrfsfcf03.grib2"


def test_select_cycles_latest_published_cycle():
    """Start 07Z -> cycle 06Z plus the cycle 24 h earlier; an unpublished cycle falls back to the previous one."""
    start, end = "2026-09-09 07:00:00", "2026-09-11 06:00:00"
    now = dt.datetime(2026, 9, 9, 12)
    cycles, fh, horizon = hrrr.select_cycles(start, end, now=now, available=lambda c: True)
    assert cycles == [dt.datetime(2026, 9, 8, 6), dt.datetime(2026, 9, 9, 6)]
    assert fh == "0-48" and horizon == dt.datetime(2026, 9, 11, 6)
    cycles, _, _ = hrrr.select_cycles(start, end, now=now, available=lambda c: c.hour != 6)
    assert cycles[-1] == dt.datetime(2026, 9, 9, 0)
    # a window in the future uses the newest cycle not after 'now'
    cycles, _, _ = hrrr.select_cycles("2026-09-10 07:00:00", "2026-09-11 06:00:00", now=now, available=lambda c: True)
    assert cycles[-1] == dt.datetime(2026, 9, 9, 12)
    with pytest.raises(RuntimeError):
        hrrr.select_cycles(start, end, now=now, available=lambda c: False, max_back=2)


def test_target_grid_covers_bbox():
    lats, lons = hrrr.target_grid((-118.7, -117.6, 33.6, 34.4), 0.03)
    assert lats[0] > lats[-1]                       # descending like ERA5
    assert lons[0] <= -118.7 and lons[-1] >= -117.6
    assert lats[-1] <= 33.6 and lats[0] >= 34.4
    assert np.allclose(np.diff(lons), 0.03) and np.allclose(np.diff(lats), -0.03)


def test_bbox_from_raster(tmp_path):
    from osgeo import gdal, osr
    p = str(tmp_path / "dsm.tif")
    ds = gdal.GetDriverByName("GTiff").Create(p, 100, 100, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((384000.0, 10.0, 0.0, 3770000.0, 0.0, -10.0))
    srs = osr.SpatialReference(); srs.ImportFromEPSG(32611); ds.SetProjection(srs.ExportToWkt()); ds = None
    lonmin, lonmax, latmin, latmax = hrrr.bbox_from_raster(p, buffer_deg=0.0)
    assert -118.3 < lonmin < lonmax < -118.2 and 34.0 < latmin < latmax < 34.1
    b = hrrr.bbox_from_raster(p, buffer_deg=0.3)
    assert np.isclose(b[0], lonmin - 0.3) and np.isclose(b[3], latmax + 0.3)


def test_covers_missing_folder(tmp_path):
    assert hrrr.covers(str(tmp_path), "2026-09-09 07:00:00", "2026-09-10 07:00:00") is False


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("SOLWEIG_TEST_NETWORK"), reason="set SOLWEIG_TEST_NETWORK=1 to download a few HRRR messages")
def test_fetch_small_window(tmp_path):
    out = hrrr.fetch_hrrr("2026-09-09 06:00:00", "2026-09-09 08:00:00", str(tmp_path), bbox=(-118.5, -118.2, 33.9, 34.2),
                          cycles=[dt.datetime(2026, 9, 9, 6)], fhours="0-2", workers=3)
    import xarray as xr
    with xr.open_dataset(os.path.join(out, hrrr.INSTANT_FILE)) as ds:
        assert ds.sizes["time"] == 3 and {"t2m", "d2m", "sp", "u10", "v10", "fsr"} <= set(ds.data_vars)
        assert 270 < float(ds["t2m"].mean()) < 320
    assert hrrr.covers(out, "2026-09-09 06:00:00", "2026-09-09 08:00:00")

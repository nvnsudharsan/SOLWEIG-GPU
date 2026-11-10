"""Tests for the 2-D meteorological fields (tile-seam fix): preprocessor.process_metfields and the
use_met_fields switch of run_utci_tiles. CPU only, no network."""
import os
import shutil
import tempfile
import datetime

import numpy as np
import pytest
from netCDF4 import Dataset, date2num
from osgeo import gdal, osr


def _write_raster(path, width, height, pixel_size, x0, y0, epsg):
    """Raster with its top-left corner at (x0, y0) in the given projected CRS."""
    ds = gdal.GetDriverByName('GTiff').Create(path, width, height, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((x0, pixel_size, 0.0, y0, 0.0, -pixel_size))
    srs = osr.SpatialReference(); srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(np.zeros((height, width), dtype=np.float32))
    ds.FlushCache(); ds = None


def _write_weather(path, lats, lons, times_utc, t2_fn, rh=50.0, wind=2.0, psfc=101325.0, uhi=1.5):
    """Outfile.nc-style weather file on a regular lat/lon grid; T2 from t2_fn(lat2d, lon2d, hour_index)."""
    lon2d, lat2d = np.meshgrid(lons, lats)
    with Dataset(path, 'w', format='NETCDF4') as ncf:
        ncf.createDimension('time', len(times_utc)); ncf.createDimension('lat', len(lats)); ncf.createDimension('lon', len(lons))
        tv = ncf.createVariable('time', 'f8', ('time',)); tv.units = "hours since 1970-01-01 00:00:00"; tv.calendar = "gregorian"
        ncf.createVariable('lat', 'f4', ('lat', 'lon'))[:] = lat2d
        ncf.createVariable('lon', 'f4', ('lat', 'lon'))[:] = lon2d
        tv[:] = date2num(times_utc, units=tv.units, calendar=tv.calendar)
        for name, val in (('RH2', rh), ('WIND', wind), ('PSFC', psfc), ('UHI_CYCLE', uhi)):
            v = ncf.createVariable(name, 'f4', ('time', 'lat', 'lon')); v[:] = np.full((len(times_utc),) + lat2d.shape, val, np.float32)
        t2 = ncf.createVariable('T2', 'f4', ('time', 'lat', 'lon'))
        t2[:] = np.stack([t2_fn(lat2d, lon2d, i) for i in range(len(times_utc))]).astype(np.float32)


@pytest.fixture
def workspace():
    d = tempfile.mkdtemp()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _setup(workspace, t2_fn):
    """One 3 km tile (UTM 11N, Los Angeles) and a 0.05 deg weather grid around it covering local 2026-09-09."""
    dem_dir = os.path.join(workspace, "DEM"); os.makedirs(dem_dir)
    _write_raster(os.path.join(dem_dir, "DEM_0_0.tif"), 300, 300, 10.0, 384000.0, 3770000.0, 32611)   # 3 x 3 km
    lats = np.arange(33.6, 34.5, 0.05); lons = np.arange(-118.8, -117.9, 0.05)
    base = datetime.datetime(2026, 9, 9, 0, 0)   # 24 h of 09-09 local = 07Z 09-09 .. 06Z 09-10 (PDT); give a wider UTC window
    times = [base + datetime.timedelta(hours=h) for h in range(0, 48)]
    nc_path = os.path.join(workspace, "Outfile.nc")
    _write_weather(nc_path, lats, lons, times, t2_fn)
    return nc_path, dem_dir


def test_constant_field_matches_scalar(workspace):
    from solweig_gpu.preprocessor import process_metfields
    nc_path, dem_dir = _setup(workspace, lambda lat, lon, i: np.full(lat.shape, 300.0 + i))
    process_metfields(nc_path, dem_dir, "2026-09-09", workspace, field_res_m=100.0, use_uhi=True)
    out = os.path.join(workspace, "metfiles", "metfields_0_0_2026-09-09.npz")
    assert os.path.exists(out)
    z = np.load(out)
    assert z["Ta"].shape[0] == 24 and list(z["hours"]) == list(range(24))
    assert z["Ta"].shape[1:] == (30, 30)                      # 3 km / 100 m
    # local hour h of 09-09 PDT = UTC index h + 7; T2 = 300 + index -> Ta = 27 + h + 7
    for h in (0, 12, 23):
        assert np.allclose(z["Ta"][h], 300.0 + h + 7 - 273.15, atol=1e-3)
    assert np.allclose(z["RH"], 50.0) and np.allclose(z["Ws"], 2.0)
    assert np.allclose(z["P"], 101.325, atol=1e-3) and np.allclose(z["uhii"], 1.5)
    assert np.allclose(z["extent"], [384000.0, 3767000.0, 387000.0, 3770000.0])


def test_gradient_is_interpolated_not_nearest(workspace):
    from solweig_gpu.preprocessor import process_metfields
    nc_path, dem_dir = _setup(workspace, lambda lat, lon, i: 280.0 + 100.0 * (lon + 118.8))   # 1 K per 0.01 deg of longitude
    process_metfields(nc_path, dem_dir, "2026-09-09", workspace, field_res_m=100.0)
    z = np.load(os.path.join(workspace, "metfiles", "metfields_0_0_2026-09-09.npz"))
    ta = z["Ta"][12]
    # smooth, monotonic west -> east, spanning about 3 km / (0.01 deg ~ 0.92 km) ~ 3.2 K
    assert np.all(np.diff(ta, axis=1) > 0)
    assert 2.5 < (ta[:, -1] - ta[:, 0]).mean() < 4.0
    assert np.abs(np.diff(ta, axis=1)).max() < 0.2        # no nearest-cell steps


def test_uhi_off_zeroes_layer(workspace):
    from solweig_gpu.preprocessor import process_metfields
    nc_path, dem_dir = _setup(workspace, lambda lat, lon, i: np.full(lat.shape, 300.0))
    process_metfields(nc_path, dem_dir, "2026-09-09", workspace, use_uhi=False)
    z = np.load(os.path.join(workspace, "metfiles", "metfields_0_0_2026-09-09.npz"))
    assert np.all(z["uhii"] == 0)


def test_run_utci_tiles_requires_fields_when_enabled(workspace):
    """use_met_fields=True must fail clearly when preprocess did not write the fields."""
    from solweig_gpu.solweig_gpu import run_utci_tiles
    for sub in ("Building_DSM", "DEM", "Trees", "walls", "aspect", "metfiles"):
        os.makedirs(os.path.join(workspace, sub))
    for sub in ("Building_DSM", "DEM", "Trees", "walls", "aspect"):
        _write_raster(os.path.join(workspace, sub, f"{sub}_0_0.tif"), 10, 10, 10.0, 384000.0, 3770000.0, 32611)
    with open(os.path.join(workspace, "metfiles", "metfile_0_0_2026-09-09.txt"), "w") as f:
        f.write("iy id it imin Q* QH QE Qs Qf Wind RH Td press rain Kdn snow ldown fcld wuh xsmd lai_hr Kdiff Kdir Wd\n")
        f.write("2026 252 12 0 -999 -999 -999 -999 -999 2.0 50 30 101.3 0 800 0 -999 -999 -999 -999 -999 -999 -999 180\n")
    with pytest.raises(FileNotFoundError, match="use_met_fields"):
        run_utci_tiles(workspace, workspace, "2026-09-09", use_met_fields=True)


def test_ppr_rejects_fields_with_own_met(workspace):
    from solweig_gpu.preprocessor import ppr
    for name in ("Building_DSM.tif", "DEM.tif", "Trees.tif"):
        _write_raster(os.path.join(workspace, name), 20, 20, 10.0, 384000.0, 3770000.0, 32611)
    with pytest.raises(ValueError, match="use_met_fields"):
        ppr(workspace, "Building_DSM.tif", "DEM.tif", "Trees.tif", None, None, 20, 0, "2026-09-09", True,
            own_met_file=os.path.join(workspace, "met.txt"), preprocess_dir=os.path.join(workspace, "pp"), use_met_fields=True)

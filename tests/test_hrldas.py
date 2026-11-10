"""Tests for the HRLDAS (LDASOUT) forcing reader: synthetic 3x3x2 LDASOUT files, no external data."""
import datetime
import numpy as np
import netCDF4
import pytest

from solweig_gpu.preprocessor import process_hrldas_data, _hrldas_latlon, HRLDAS_PATTERN

ATTRS = {'MAP_PROJ': 1, 'LAT1': 28.67683, 'LON1': -99.36203, 'DX': 1000.0, 'DY': 1000.0,
         'TRUELAT1': 30.61, 'TRUELAT2': 30.61, 'STAND_LON': -96.339, 'missing_value': -1.0e33}


def _write_ldasout(path, times, with_urban=True):
    ny, nx = 3, 3
    with netCDF4.Dataset(path, 'w') as nc:
        nc.createDimension('Time', None); nc.createDimension('DateStrLen', 19)
        nc.createDimension('south_north', ny); nc.createDimension('west_east', nx)
        for k, v in ATTRS.items():
            nc.setncattr(k, v)
        tv = nc.createVariable('Times', 'S1', ('Time', 'DateStrLen'))
        def var(name):
            return nc.createVariable(name, 'f4', ('Time', 'south_north', 'west_east'))
        names = ['T2MV', 'T2MB', 'Q2MV', 'Q2MB', 'FVEG', 'FORCPLSM', 'FORCZLSM', 'FORCTLSM', 'FORCWLSM', 'SWFORC', 'LWFORC', 'TRAD']
        if with_urban:
            names += ['T2M', 'Q2M', 'WS10M']
        vs = {n: var(n) for n in names}
        for t, dtm in enumerate(times):
            tv[t, :] = np.array(list(dtm.strftime('%Y-%m-%d_%H:%M:%S')), dtype='S1')
            base = 300.0 + t
            vs['T2MV'][t] = base - 1; vs['T2MB'][t] = base + 1; vs['FVEG'][t] = 0.5
            vs['Q2MV'][t] = 0.010; vs['Q2MB'][t] = 0.012
            vs['FORCPLSM'][t] = 98000.0; vs['FORCZLSM'][t] = 40.0; vs['FORCTLSM'][t] = base
            vs['FORCWLSM'][t] = 5.0; vs['SWFORC'][t] = 500.0; vs['LWFORC'][t] = 400.0; vs['TRAD'][t] = base + 10
            if with_urban:
                vs['T2M'][t] = base; vs['Q2M'][t] = 0.011; vs['WS10M'][t] = 3.0
                arr = np.full((ny, nx), base, np.float32); arr[0, 0] = -9999.0; vs['T2M'][t] = arr     # water-cell sentinel
                arr = np.full((ny, nx), 3.0, np.float32); arr[2, 2] = -1.0e33; vs['WS10M'][t] = arr  # missing_value


def test_filename_pattern():
    assert HRLDAS_PATTERN.match('2020081301.LDASOUT_DOMAIN3')
    assert not HRLDAS_PATTERN.match('2020081301.LDASIN_DOMAIN3')
    assert not HRLDAS_PATTERN.match('wrfout_d03_2020-08-13_12_00_00')


def test_latlon_grid_point_one_is_lat1_lon1():
    lat, lon = _hrldas_latlon(ATTRS, 582, 561)
    assert lat.shape == (582, 561)
    assert abs(lat[0, 0] - ATTRS['LAT1']) < 1e-6 and abs(lon[0, 0] - ATTRS['LON1']) < 1e-6
    assert lat[-1, 0] > lat[0, 0] and lon[0, -1] > lon[0, 0]          # north and east increase
    assert abs(lat[181, 162] - 30.3328) < 1e-3 and abs(lon[181, 162] + 97.7250) < 1e-3   # Austin cell (pyproj reference)


def test_process_hrldas_data_window_and_mapping(tmp_path):
    t0 = datetime.datetime(2020, 8, 13, 1)
    _write_ldasout(tmp_path / '2020081301.LDASOUT_DOMAIN3', [t0 + datetime.timedelta(hours=i) for i in range(3)])
    _write_ldasout(tmp_path / '2020081304.LDASOUT_DOMAIN3', [t0 + datetime.timedelta(hours=i) for i in range(3, 6)])
    out = tmp_path / 'Outfile.nc'
    process_hrldas_data('2020-08-13 02:00:00', '2020-08-13 04:00:00', str(tmp_path), output_file=str(out))
    with netCDF4.Dataset(out) as nc:
        assert nc.dimensions['time'].size == 3                      # 02, 03, 04 across the two files
        times = netCDF4.num2date(nc['time'][:], nc['time'].units, only_use_cftime_datetimes=False)
        assert [t.hour for t in times] == [2, 3, 4]
        t2 = np.ma.filled(nc['T2'][:], np.nan); ws = np.ma.filled(nc['WIND'][:], np.nan)
        assert np.isnan(t2[0, 0, 0]) and np.isnan(ws[0, 2, 2])      # both sentinels -> NaN
        assert abs(t2[0, 1, 1] - 301.0) < 1e-4                      # T2 <- T2M (hour 02 -> base 301)
        assert abs(ws[0, 1, 1] - 3.0) < 1e-6                        # WIND <- WS10M, not FORCWLSM
        psfc = np.ma.filled(nc['PSFC'][:], np.nan)[0, 1, 1]
        assert 98000.0 * 1.003 < psfc < 98000.0 * 1.006              # 40 m reduced to the surface (~ +0.45 %)
        rh = np.ma.filled(nc['RH2'][:], np.nan)[0, 1, 1]
        assert abs(rh - 46.1) < 1.0                                  # q=0.011 (w=0.01112), T=301 K, p=98.44 kPa -> 46.1 %
        assert np.all(np.isnan(np.ma.filled(nc['WDIR'][:], np.nan)))  # no wind components in LDASOUT
        assert abs(nc['lat'][0, 0] - ATTRS['LAT1']) < 1e-5
        assert nc['SWDOWN'][:].max() == 500.0 and nc['GLW'][:].max() == 400.0 and abs(nc['TSK'][0, 1, 1] - 311.0) < 1e-4


def test_process_hrldas_data_fallbacks_without_urban_diagnostics(tmp_path):
    t0 = datetime.datetime(2020, 8, 13, 1)
    _write_ldasout(tmp_path / '2020081301.LDASOUT_DOMAIN3', [t0], with_urban=False)
    out = tmp_path / 'Outfile.nc'
    process_hrldas_data('2020-08-13 01:00:00', '2020-08-13 01:00:00', str(tmp_path), output_file=str(out))
    with netCDF4.Dataset(out) as nc:
        assert abs(nc['T2'][0, 1, 1] - 300.0) < 1e-4                # 0.5*(299) + 0.5*(301)
        assert abs(nc['WIND'][0, 1, 1] - 5.0) < 1e-6                 # FORCWLSM fallback


def test_process_hrldas_data_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        process_hrldas_data('2020-08-13 01:00:00', '2020-08-13 02:00:00', str(tmp_path))
    _write_ldasout(tmp_path / '2020081301.LDASOUT_DOMAIN3', [datetime.datetime(2020, 8, 13, 1)])
    with pytest.raises(ValueError):
        process_hrldas_data('2021-01-01 00:00:00', '2021-01-01 02:00:00', str(tmp_path))

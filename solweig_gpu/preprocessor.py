#SOLWEIG-GPU: GPU-accelerated SOLWEIG model for urban thermal comfort simulation
#Copyright (C) 2022–2025 Harsh Kamath and Naveen Sudharsan

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.

import os
import re
import glob
import datetime
import pytz
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
import shutil
from netCDF4 import Dataset, date2num
import datetime as dt
from datetime import timedelta
from osgeo import gdal, ogr, osr
from shapely.geometry import box, Polygon
from matplotlib.path import Path
from timezonefinder import TimezoneFinder
from scipy.spatial import cKDTree
import math
from tqdm import tqdm
import pytz

gdal.UseExceptions()
RHO_CP = 1.225 * 1005.0  # air density * Cp

WRF_PATTERNS = [
    re.compile(r'^wrfout_d0([1-9])_(\d{4}-\d{2}-\d{2})_(\d{2}_\d{2}_\d{2})$'),  # HH_MM_SS
    re.compile(r'^wrfout_d0([1-9])_(\d{4}-\d{2}-\d{2})_(\d{2}:\d{2}:\d{2})$'),  # HH:MM:SS
    re.compile(r'^wrfout_d0([1-9])_(\d{4}-\d{2}-\d{2})_(\d{2})$'),              # HH
]

def infer_timezone_from_grid(latitudes, longitudes):
    """Infer timezone name from the center of the ERA5 grid."""
    tf = TimezoneFinder()

    if latitudes.ndim == 1 and longitudes.ndim == 1:
        lat_c = float(latitudes[len(latitudes)//2])
        lon_c = float(longitudes[len(longitudes)//2])
    else:
        i = latitudes.shape[0] // 2
        j = latitudes.shape[1] // 2
        lat_c = float(latitudes[i, j])
        lon_c = float(longitudes[i, j])

    tz_name = tf.timezone_at(lat=lat_c, lng=lon_c)
    return tz_name or "UTC"
    
def utc_times_to_local_naive(times_utc, timezone_name):
    """
    Convert UTC timestamps to local naive datetimes for local-day resampling
    and local-night detection.
    """
    t = pd.to_datetime(times_utc, utc=True)
    t_local = t.tz_convert(timezone_name).tz_localize(None)
    return t_local.to_numpy()

######################################### 
# Modified after Andrea Zonato
def _night_sine_profile(sw_values, amp_values, extension_before=0, extension_after=1):
    """
    Build a nighttime UHI profile from SWDOWN in local time:
    - detect night where SWDOWN <= 0
    - optionally extend before/after
    - shape with a sine curve over each night segment
    """
    sw_arr  = np.asarray(sw_values, dtype=float)
    amp_arr = np.asarray(amp_values, dtype=float)

    n = sw_arr.size
    out = np.zeros(n, dtype=float)

    if n == 0 or amp_arr.shape != sw_arr.shape:
        return out

    valid = np.isfinite(sw_arr) & np.isfinite(amp_arr)

    # Night starts exactly when SWDOWN <= 0
    base_night = valid & (sw_arr <= 5.0)

    if not np.any(base_night):
        return out

    idx = np.flatnonzero(base_night)
    splits = np.where(np.diff(idx) > 1)[0] + 1
    raw_segments = [seg for seg in np.split(idx, splits) if seg.size]

    for seg in raw_segments:
        s = int(max(0, seg[0] - extension_before))
        e = int(min(n, seg[-1] + 1 + extension_after))
        seg_ext = np.arange(s, e, dtype=int)

        L = seg_ext.size
        if L < 2:
            continue

        seg_amp = amp_arr[seg_ext]
        seg_amp = seg_amp[np.isfinite(seg_amp)]
        if seg_amp.size == 0:
            continue

        A = float(np.mean(seg_amp))
        if A <= 0:
            continue

        x = np.linspace(0, np.pi, L)
        profile = np.sin(x)
        profile = np.maximum(profile, 0.0)

        out[seg_ext] = A * profile

    return out

def compute_uhi_cycle_from_arrays(times, t2_k, swdown_wm2, u_wind_ms, v_wind_ms):
    """
    Compute UHI_CYCLE from arrays already extracted from ERA5.

    Parameters
    ----------
    times : 1D array-like of datetimes
    t2_k : ndarray (time, lat, lon), air temperature in K
    swdown_wm2 : ndarray (time, lat, lon), shortwave down radiation in W m-2
    u_wind_ms : ndarray (time, lat, lon), zonal wind speed in m s-1
    v_wind_ms : ndarray (time, lat, lon), meridional wind speed in m s-1

    Returns
    -------
    uhi_cycle : ndarray (time, lat, lon), in K
    """
    lat_n = t2_k.shape[1]
    lon_n = t2_k.shape[2]

    coords = {"time": times, "lat": np.arange(lat_n), "lon": np.arange(lon_n)}

    da_t2 = xr.DataArray(t2_k, dims=("time", "lat", "lon"), coords=coords)
    da_sw = xr.DataArray(swdown_wm2, dims=("time", "lat", "lon"), coords=coords)
    da_u = xr.DataArray(u_wind_ms, dims=("time", "lat", "lon"), coords=coords)
    da_v = xr.DataArray(v_wind_ms, dims=("time", "lat", "lon"), coords=coords)

    # Wind speed magnitude
    da_wd = np.sqrt(da_u**2 + da_v**2)

    # Daily diagnostics
    sdown_daily = (da_sw / RHO_CP).resample(time="1D").mean()
    wd_daily = da_wd.resample(time="1D").mean()
    tmax_daily = da_t2.resample(time="1D").max()
    tmin_daily = da_t2.resample(time="1D").min()
    dtr_daily = tmax_daily - tmin_daily

    numerator = (sdown_daily * dtr_daily**3)
    numerator = numerator.where(np.isfinite(numerator), 0).clip(min=0)

    # Avoid divide-by-zero / near-zero wind
    wd_daily_safe = xr.where(wd_daily > 1.0, wd_daily, 1.0)

    uhi_max_daily = (numerator / wd_daily_safe) ** 0.25
    uhi_max_daily = uhi_max_daily.fillna(0)

    # Expand daily amplitude back to hourly timestamps
    uhi_max_ts = uhi_max_daily.reindex(time=da_t2.time, method="ffill").fillna(0)

    # Build nighttime UHI cycle
    uhi_cycle = xr.apply_ufunc(
        _night_sine_profile,
        da_sw,
        uhi_max_ts,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="allowed",
        kwargs={"extension_before": 0, "extension_after": 0},
    )

    uhi_cycle = uhi_cycle.transpose("time", "lat", "lon")
    return uhi_cycle.values.astype(np.float32)

# Modified after Andrea Zonato
######################################### 

def _match_wrfout(base):
    """
    Match WRF output filename against known patterns.
    
    Args:
        base (str): Base filename to match
    
    Returns:
        tuple: (pattern_index, match_object) or (None, None) if no match
    """
    for i, rx in enumerate(WRF_PATTERNS):
        m = rx.match(base)
        if m:
            return i, m
    return None, None

def extract_datetime_strict(filename):
    """
    Return (datetime, domain_int) for strictly valid wrfout names.
    Raises ValueError for any non-matching filename.
    """
    base = os.path.basename(filename)
    idx, m = _match_wrfout(base)
    if m is None:
        raise ValueError(f"Unsupported wrfout filename: {base}")

    dom = int(m.group(1))         
    date = m.group(2)             
    t    = m.group(3)              

    if idx == 0:  # HH_MM_SS
        hh, mm, ss = map(int, t.split('_'))
        dt = datetime.datetime(
            int(date[0:4]), int(date[5:7]), int(date[8:10]), hh, mm, ss
        )
    elif idx == 1:  # HH:MM:SS
        dt = datetime.datetime.strptime(f"{date}_{t}", "%Y-%m-%d_%H:%M:%S")
    else:  # idx == 2, HH only
        dt = datetime.datetime.strptime(f"{date}_{t}", "%Y-%m-%d_%H")

    return dt, dom

# =============================================================================
# Function to check that all raster files have matching dimensions, pixel size, and CRS.
# =============================================================================
def check_rasters(files):
    """
    Check that all provided raster files have matching dimensions, pixel size, and CRS.

    Parameters:
        files (list): List of raster file paths.

    Returns:
        bool: True if all checks pass, raises ValueError/FileNotFoundError otherwise.
    """
    if not files:
        raise ValueError("No raster files provided.")

    ref_file = files[0]
    ds = gdal.Open(ref_file)
    if ds is None:
        raise FileNotFoundError(f"Could not open {ref_file}")
    ref_width = ds.RasterXSize
    ref_height = ds.RasterYSize
    ref_gt = ds.GetGeoTransform()  # (originX, pixelWidth, rot, originY, rot, pixelHeight)
    ref_pixel_width = ref_gt[1]
    ref_pixel_height = ref_gt[5]  # typically negative
    ref_crs = ds.GetProjection()
    ds = None

    for f in files[1:]:
        ds = gdal.Open(f)
        if ds is None:
            raise FileNotFoundError(f"Could not open {f}")
        if ds.RasterXSize != ref_width or ds.RasterYSize != ref_height:
            raise ValueError("Error: Raster dimensions do not match.")
        gt = ds.GetGeoTransform()
        pixel_width = gt[1]
        pixel_height = gt[5]
        if pixel_width != ref_pixel_width or pixel_height != ref_pixel_height:
            raise ValueError("Error: Pixel sizes do not match.")
        if ds.GetProjection() != ref_crs:
            raise ValueError("Error: CRS does not match.")
        ds = None

    return True

# =============================================================================
# Function to tile a raster file into smaller chunks.
# =============================================================================
def _resolve_path(base_path, path_or_name):
    if path_or_name is None:
        return None
    if os.path.isabs(path_or_name):
        return path_or_name
    return os.path.join(base_path, path_or_name)


def find_windcoeff_files(base_path, windcoeff_filename):
    """
    windcoeff_filename can be:
      - None: no wind coefficient
      - folder path: read WindCoeff_dir*.tif inside it
      - single file path: old behavior / fallback
      - glob pattern: e.g. 'WindCoeff_dir*.tif'
    """
    if windcoeff_filename is None:
        return []

    wind_path = _resolve_path(base_path, windcoeff_filename)

    if os.path.isdir(wind_path):
        files = sorted(glob.glob(os.path.join(wind_path, "WindCoeff_dir*.tif")))
    elif any(ch in wind_path for ch in ["*", "?", "["]):
        files = sorted(glob.glob(wind_path))
    elif os.path.isfile(wind_path):
        files = [wind_path]
    else:
        files = []

    if not files:
        raise FileNotFoundError(f"No wind coefficient files found from: {wind_path}")

    dir_files = [
        f for f in files
        if re.search(r"WindCoeff_dir\d{3}\.tif$", os.path.basename(f))
    ]

    if dir_files:
        files = sorted(dir_files)

        found_dirs = []
        for f in files:
            m = re.search(r"WindCoeff_dir(\d{3})\.tif$", os.path.basename(f))
            if m:
                found_dirs.append(int(m.group(1)))

        expected_dirs = list(range(0, 360, 30))
        missing = sorted(set(expected_dirs) - set(found_dirs))

        if missing:
            raise FileNotFoundError(
                "Missing directional wind coefficient rasters: "
                + ", ".join(f"WindCoeff_dir{d:03d}.tif" for d in missing)
            )

    return files


def create_tiles_to_folder(infile, tilesize, overlap, out_folder, tile_prefix, clear_folder=False):
    """
    Tile a raster into a specified folder using a specified filename prefix.

    Example output:
      out_folder/WindCoeff_dir030_0_0.tif
    """
    if overlap < 0 or overlap >= tilesize:
        raise ValueError("overlap must be 0 ≤ overlap < tilesize")

    ds = gdal.Open(infile)
    if ds is None:
        raise FileNotFoundError(f"Could not open {infile}")

    width = ds.RasterXSize
    height = ds.RasterYSize

    if clear_folder and os.path.exists(out_folder):
        shutil.rmtree(out_folder)

    os.makedirs(out_folder, exist_ok=True)

    if tilesize >= width and tilesize >= height:
        outfile = os.path.join(out_folder, f"{tile_prefix}_0_0.tif")
        options = gdal.TranslateOptions(format="GTiff", srcWin=[0, 0, width, height])
        gdal.Translate(outfile, ds, options=options)
        print(f"Created single tile: {outfile}")
        ds = None
        return

    for i in range(0, width, tilesize):
        for j in range(0, height, tilesize):
            tile_width = min(tilesize + overlap, width - i)
            tile_height = min(tilesize + overlap, height - j)

            outfile = os.path.join(out_folder, f"{tile_prefix}_{i}_{j}.tif")
            options = gdal.TranslateOptions(
                format="GTiff",
                srcWin=[i, j, tile_width, tile_height]
            )
            gdal.Translate(outfile, ds, options=options)
            print(f"Created tile: {outfile}")

    ds = None


def create_windcoeff_tiles(windcoeff_files, tilesize, overlap, preprocess_dir):
    """
    Tile all directional wind coefficient rasters into one folder:

      preprocess_dir/WindCoeff/WindCoeff_dir000_i_j.tif
      preprocess_dir/WindCoeff/WindCoeff_dir030_i_j.tif
      ...
    """
    if not windcoeff_files:
        print("No wind coefficient rasters provided. Skipping wind coefficient tiling.")
        return

    out_folder = os.path.join(preprocess_dir, "WindCoeff")

    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder, exist_ok=True)

    for wind_fp in windcoeff_files:
        tile_prefix = os.path.splitext(os.path.basename(wind_fp))[0]

        print(f"Creating wind coefficient tiles for {tile_prefix}...")
        create_tiles_to_folder(
            infile=wind_fp,
            tilesize=tilesize,
            overlap=overlap,
            out_folder=out_folder,
            tile_prefix=tile_prefix,
            clear_folder=False
        )

def create_tiles(infile, tilesize, overlap, tile_type, preprocess_dir):
    """
    Tile a raster file into smaller chunks.

    Normal rasters are written as:
      preprocess_dir/DEM/DEM_i_j.tif
      preprocess_dir/Trees/Trees_i_j.tif
      etc.
    """
    out_folder = os.path.join(preprocess_dir, tile_type)

    create_tiles_to_folder(
        infile=infile,
        tilesize=tilesize,
        overlap=overlap,
        out_folder=out_folder,
        tile_prefix=tile_type,
        clear_folder=True
    )

# =============================================================================
# The function expects two files in folder_path:
#        - 'data_stream-oper_stepType-instant.nc'
#        - "data_stream-oper_stepType-accum.n"
        
# It processes the data to compute temperature (in °C), surface pressure (in kPa), 
# relative humidity (in %), wind speed (in m/s), shortwave and longwave radiation (in W/m^2),
# and writes the results to a new netCDF file. These netCDF files will be used to create the 
# meteogrological forcing for SOLWEIG. Note that RAIN is set to 0.
# =============================================================================

import os
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset, date2num

def _normalize_time_coord(ds: xr.Dataset) -> xr.Dataset:
    """Return a copy with a proper 'time' coord from 'valid_time' or 'time'+'step'."""
    ds = ds.copy()

    # Case 1: valid_time -> time
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        return ds.rename({"valid_time": "time"})

    # Case 2: time + step
    has_time = ("time" in ds.dims) or ("time" in ds.coords)
    has_step = ("step" in ds.dims) or ("step" in ds.coords)
    if has_time and has_step:
        base = ds["time"]
        step = ds["step"]
        # ensure timedelta dtype
        if not np.issubdtype(step.dtype, np.timedelta64):
            try:
                step_td = xr.DataArray(pd.to_timedelta(step.values),
                                       dims=step.dims, coords=step.coords)
            except Exception:
                step_td = xr.DataArray(pd.to_timedelta(step.values, unit="h"),
                                       dims=step.dims, coords=step.coords)
        else:
            step_td = step

        vt = (base + step_td).rename("time")
        if vt.ndim == 2:
            vt_flat = vt.stack(time_flat=("time", "step")).rename("time")
            ds = ds.stack(time_flat=("time", "step")).rename_dims({"time_flat": "time"}).drop_vars(["time", "step"])
            ds = ds.assign_coords(time=vt_flat.values)
        else:
            if "step" in vt.dims and "time" not in vt.dims:
                ds = ds.swap_dims({"step": "time"}).drop_vars(["step"])
            ds = ds.assign_coords(time=vt.values)
        return ds

    # Case 3: plain time
    if has_time:
        try:
            ds = xr.decode_cf(ds)
        except Exception:
            pass
        return ds

    raise KeyError(
        "No usable time coordinate. Expected 'valid_time' or 'time' (+ optional 'step'). "
        f"Found dims={list(ds.dims)}, coords={list(ds.coords)}"
    )

def process_era5_data(start_time, end_time, folder_path, output_file="Outfile.nc"):
    """
    Same numeric outputs as your original function, but time is taken from the files
    and sliced to [start_time, end_time] UTC. Variables written:
      T2 (t2m, Kelvin), PSFC (sp, Pa), RH2 (%), WIND (m/s), SWDOWN (W/m^2 = ssrd/3600).
    """
    # Parse UTC times
    start_dt = dt.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_dt   = dt.datetime.strptime(end_time,   "%Y-%m-%d %H:%M:%S")

    # Open and normalize time coordinates
    instant_file = os.path.join(folder_path, 'data_stream-oper_stepType-instant.nc')
    accum_file   = os.path.join(folder_path, 'data_stream-oper_stepType-accum.nc')

    ds_instant = _normalize_time_coord(xr.open_dataset(instant_file, decode_times=True))
    ds_accum   = _normalize_time_coord(xr.open_dataset(accum_file,   decode_times=True))

    # Slice both to requested window present in files
    ds_instant = ds_instant.sel(time=slice(start_dt, end_dt))
    ds_accum   = ds_accum.sel(time=slice(start_dt, end_dt))

    if ds_instant.time.size == 0:
        raise ValueError("No instantaneous timestamps within requested window.")
    if ds_accum.time.size == 0:
        raise ValueError("No accumulated timestamps within requested window.")

    # Align to exactly overlapping timestamps (hourly)
    common_t = np.intersect1d(ds_instant.time.values, ds_accum.time.values)
    if common_t.size == 0:
        raise ValueError("Instantaneous and accumulated streams have no overlapping timestamps.")
    ds_instant = ds_instant.sel(time=common_t)
    ds_accum   = ds_accum.sel(time=common_t)

    # ---- Compute exactly like your original (keep same numeric values) ----
    def saturation_vapor_pressure(Tc):
        """
        Calculate saturation vapor pressure using Magnus formula.
        
        Args:
            Tc (float or np.ndarray): Temperature in Celsius
        
        Returns:
            float or np.ndarray: Saturation vapor pressure in hPa
        """
        # Tc in °C → hPa
        return 6.112 * np.exp((17.67 * Tc) / (Tc + 243.5))

    temperatures = ds_instant['t2m'].values              # K (kept as-is)
    dew_points   = ds_instant['d2m'].values              # K
    surface_pressures = ds_instant['sp'].values          # Pa
    u10 = ds_instant['u10'].values
    v10 = ds_instant['v10'].values
    wind_speeds = np.sqrt(u10**2 + v10**2)
    wind_dirs = (270.0 - np.degrees(np.arctan2(v10, u10))) % 360.0
    wind_dirs = np.where(wind_speeds > 0.01, wind_dirs, 0.0)

    # Hourly totals (J m^-2 per hour) → W m^-2 by /3600
    shortwave_radiation = (ds_accum['ssrd'].values / 3600.0).astype(np.float32)
    # longwave_radiation  = (ds_accum['strd'].values / 3600.0).astype(np.float32)  # not written

    # RH using Kelvin inputs converted to °C for the formula
    e_temp      = saturation_vapor_pressure(temperatures - 273.15)
    e_dew_point = saturation_vapor_pressure(dew_points   - 273.15)
    relative_humidities = 100.0 * (e_dew_point / e_temp)
    relative_humidities = np.clip(relative_humidities, 0, 100)

    # Lat/Lon grids (same behavior)
    latitudes  = ds_instant['latitude'].values
    longitudes = ds_instant['longitude'].values
    if latitudes.ndim == 1 and longitudes.ndim == 1:
        lon2d, lat2d = np.meshgrid(longitudes, latitudes)
    else:
        lat2d = latitudes
        lon2d = longitudes

    # Use dataset times (no synthetic list)
    times = ds_instant.time.values

    # ---- Write NetCDF (same var set/labels as your code) ----
    with Dataset(output_file, 'w', format='NETCDF4') as nc:
        nc.createDimension('time', times.shape[0])
        nc.createDimension('lat',  lat2d.shape[0])
        nc.createDimension('lon',  lon2d.shape[1])

        time_var = nc.createVariable('time', 'f8', ('time',))
        lat_var  = nc.createVariable('lat',  'f4', ('lat', 'lon'))
        lon_var  = nc.createVariable('lon',  'f4', ('lat', 'lon'))

        t2_var     = nc.createVariable('T2',     'f4', ('time', 'lat', 'lon'), zlib=True)
        psfc_var   = nc.createVariable('PSFC',   'f4', ('time', 'lat', 'lon'), zlib=True)
        rh2_var    = nc.createVariable('RH2',    'f4', ('time', 'lat', 'lon'), zlib=True)
        wind_var   = nc.createVariable('WIND',   'f4', ('time', 'lat', 'lon'), zlib=True)
        wdir_var   = nc.createVariable('WDIR',   'f4', ('time', 'lat', 'lon'), zlib=True)
        swdown_var = nc.createVariable('SWDOWN', 'f4', ('time', 'lat', 'lon'), zlib=True)
        # glw_var  = nc.createVariable('GLW',    'f4', ('time', 'lat', 'lon'), zlib=True)

        # Keep your original unit strings
        time_var.units = "hours since 1970-01-01 00:00:00"
        time_var.calendar = "gregorian"
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"
        t2_var.units = "degC"      # (values are K, kept to match your outputs)
        psfc_var.units = "kPa"     # (values are Pa, kept to match your outputs)
        rh2_var.units = "%"
        wind_var.units = "m/s"
        wdir_var.units = "degrees"
        swdown_var.units = "W/m^2"
        # glw_var.units = "W/m^2"

        # Encode time from dataset timestamps
        py_times = [np.datetime64(t).astype("datetime64[s]").astype(object) for t in times]
        time_var[:] = date2num(py_times, units=time_var.units, calendar=time_var.calendar)

        lat_var[:, :] = lat2d.astype('float32')
        lon_var[:, :] = lon2d.astype('float32')

        # Ensure (time, lat, lon) ordering
        # xarray variables are typically (time, latitude, longitude)
        t2 = ds_instant['t2m'].transpose('time','latitude','longitude').values.astype('float32')
        sp = ds_instant['sp'].transpose('time','latitude','longitude').values.astype('float32')
        rh = xr.DataArray(relative_humidities,
                          dims=('time','latitude','longitude'),
                          coords={'time': times,
                                  'latitude': ds_instant['latitude'],
                                  'longitude': ds_instant['longitude']}
                         ).transpose('time','latitude','longitude').values.astype('float32')
        wind = xr.DataArray(wind_speeds,
                            dims=('time','latitude','longitude'),
                            coords={'time': times,
                                    'latitude': ds_instant['latitude'],
                                    'longitude': ds_instant['longitude']}
                           ).transpose('time','latitude','longitude').values.astype('float32')
        wdir = xr.DataArray(wind_dirs,
                    dims=('time','latitude','longitude'),
                    coords={'time': times,
                            'latitude': ds_instant['latitude'],
                            'longitude': ds_instant['longitude']}
                   ).transpose('time','latitude','longitude').values.astype('float32')        
        swd = xr.DataArray(shortwave_radiation,
                           dims=('time','latitude','longitude'),
                           coords={'time': times,
                                   'latitude': ds_instant['latitude'],
                                   'longitude': ds_instant['longitude']}
                          ).transpose('time','latitude','longitude').values.astype('float32')

        t2_var[:, :, :]     = t2
        psfc_var[:, :, :]   = sp
        rh2_var[:, :, :]    = rh
        wind_var[:, :, :]   = wind
        wdir_var[:, :, :]   = wdir
        swdown_var[:, :, :] = swd
        # glw_var[:, :, :]  = ...

    print(f"ERA5 forcing file created: {output_file}  ({len(times)} steps from {py_times[0]} to {py_times[-1]} UTC)")

# If the nightime UHI intensity needs to be included in air temperature calculation
def process_era5_data_uhi(start_time, end_time, folder_path, output_file="Outfile.nc"):
    """
    Same numeric outputs as your original function, but time is taken from the files
    and sliced to [start_time, end_time] UTC. Variables written:
      T2 (t2m, Kelvin), PSFC (sp, Pa), RH2 (%), WIND (m/s), SWDOWN (W/m^2), UHI_CYCLE (K).
    """
    start_dt = dt.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_dt   = dt.datetime.strptime(end_time,   "%Y-%m-%d %H:%M:%S")

    # Pad the requested window so UHI_CYCLE can be computed on complete nights
    pad_hours = 24
    start_pad = start_dt - dt.timedelta(hours=pad_hours)
    end_pad   = end_dt   + dt.timedelta(hours=pad_hours)

    instant_file = os.path.join(folder_path, 'data_stream-oper_stepType-instant.nc')
    accum_file   = os.path.join(folder_path, 'data_stream-oper_stepType-accum.nc')

    ds_instant = _normalize_time_coord(xr.open_dataset(instant_file, decode_times=True))
    ds_accum   = _normalize_time_coord(xr.open_dataset(accum_file,   decode_times=True))

    # Select padded window for computation
    ds_instant = ds_instant.sel(time=slice(start_pad, end_pad))
    ds_accum   = ds_accum.sel(time=slice(start_pad, end_pad))

    if ds_instant.time.size == 0:
        raise ValueError("No instantaneous timestamps within requested window.")
    if ds_accum.time.size == 0:
        raise ValueError("No accumulated timestamps within requested window.")

    common_t = np.intersect1d(ds_instant.time.values, ds_accum.time.values)
    if common_t.size == 0:
        raise ValueError("Instantaneous and accumulated streams have no overlapping timestamps.")

    ds_instant = ds_instant.sel(time=common_t)
    ds_accum   = ds_accum.sel(time=common_t)

    def saturation_vapor_pressure(Tc):
        return 6.112 * np.exp((17.67 * Tc) / (Tc + 243.5))

    temperatures = ds_instant['t2m'].values
    dew_points   = ds_instant['d2m'].values
    surface_pressures = ds_instant['sp'].values
    u10 = ds_instant['u10'].values
    v10 = ds_instant['v10'].values
    wind_speeds = np.sqrt(u10**2 + v10**2)
    wind_dirs = (270.0 - np.degrees(np.arctan2(v10, u10))) % 360.0
    wind_dirs = np.where(wind_speeds > 0.01, wind_dirs, 0.0).astype(np.float32)

    shortwave_radiation = (ds_accum['ssrd'].values / 3600.0).astype(np.float32)

    e_temp      = saturation_vapor_pressure(temperatures - 273.15)
    e_dew_point = saturation_vapor_pressure(dew_points   - 273.15)
    relative_humidities = 100.0 * (e_dew_point / e_temp)
    relative_humidities = np.clip(relative_humidities, 0, 100)

    latitudes  = ds_instant['latitude'].values
    longitudes = ds_instant['longitude'].values
    if latitudes.ndim == 1 and longitudes.ndim == 1:
        lon2d, lat2d = np.meshgrid(longitudes, latitudes)
    else:
        lat2d = latitudes
        lon2d = longitudes

    times = ds_instant.time.values  # keep original UTC times for writing output

    tz_name = infer_timezone_from_grid(latitudes, longitudes)
    times_local = utc_times_to_local_naive(times, tz_name)

    print(f"Using local timezone for UHI_CYCLE: {tz_name}")

    # Compute UHI in local time so daily grouping and night timing are local
    uhi_cycle = compute_uhi_cycle_from_arrays(times_local, temperatures, shortwave_radiation, u10, v10)
    
    times_pd = pd.to_datetime(times)

    keep = (times_pd >= pd.Timestamp(start_dt)) & (times_pd <= pd.Timestamp(end_dt))
    keep_idx = np.where(keep)[0]

    times = times[keep_idx]
    temperatures = temperatures[keep_idx, :, :]
    dew_points = dew_points[keep_idx, :, :]
    surface_pressures = surface_pressures[keep_idx, :, :]
    u10 = u10[keep_idx, :, :]
    v10 = v10[keep_idx, :, :]
    wind_speeds = wind_speeds[keep_idx, :, :]
    wind_dirs = wind_dirs[keep_idx, :, :]
    shortwave_radiation = shortwave_radiation[keep_idx, :, :]
    relative_humidities = relative_humidities[keep_idx, :, :]
    uhi_cycle = uhi_cycle[keep_idx, :, :]

    with Dataset(output_file, 'w', format='NETCDF4') as nc:
        nc.createDimension('time', times.shape[0])
        nc.createDimension('lat',  lat2d.shape[0])
        nc.createDimension('lon',  lon2d.shape[1])

        time_var = nc.createVariable('time', 'f8', ('time',))
        lat_var  = nc.createVariable('lat',  'f4', ('lat', 'lon'))
        lon_var  = nc.createVariable('lon',  'f4', ('lat', 'lon'))

        t2_var     = nc.createVariable('T2',        'f4', ('time', 'lat', 'lon'), zlib=True)
        psfc_var   = nc.createVariable('PSFC',      'f4', ('time', 'lat', 'lon'), zlib=True)
        rh2_var    = nc.createVariable('RH2',       'f4', ('time', 'lat', 'lon'), zlib=True)
        wind_var   = nc.createVariable('WIND',      'f4', ('time', 'lat', 'lon'), zlib=True)
        wdir_var   = nc.createVariable('WDIR',      'f4', ('time', 'lat', 'lon'), zlib=True)
        swdown_var = nc.createVariable('SWDOWN',    'f4', ('time', 'lat', 'lon'), zlib=True)
        uhi_var    = nc.createVariable('UHI_CYCLE', 'f4', ('time', 'lat', 'lon'), zlib=True)

        time_var.units = "hours since 1970-01-01 00:00:00"
        time_var.calendar = "gregorian"
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"
        t2_var.units = "K"
        psfc_var.units = "Pa"
        rh2_var.units = "%"
        wind_var.units = "m/s"
        wdir_var.units = "degrees"
        swdown_var.units = "W/m^2"
        uhi_var.units = "K"

        py_times = [np.datetime64(t).astype("datetime64[s]").astype(object) for t in times]
        time_var[:] = date2num(py_times, units=time_var.units, calendar=time_var.calendar)

        lat_var[:, :] = lat2d.astype('float32')
        lon_var[:, :] = lon2d.astype('float32')

        t2 = temperatures.astype('float32')
        sp = surface_pressures.astype('float32')
        rh = relative_humidities.astype('float32')
        wind = wind_speeds.astype('float32')
        wdir = wind_dirs.astype('float32')
        swd = shortwave_radiation.astype('float32')
        uhi = uhi_cycle.astype('float32')

        t2_var[:, :, :]     = t2
        psfc_var[:, :, :]   = sp
        rh2_var[:, :, :]    = rh
        wind_var[:, :, :]   = wind
        wdir_var[:, :, :]   = wdir
        swdown_var[:, :, :] = swd
        uhi_var[:, :, :]    = uhi
        
    print(f"ERA5 forcing file created: {output_file}  ({len(times)} steps from {py_times[0]} to {py_times[-1]} UTC)")

    
# =============================================================================
#    The function will:
#        - Populate the list of available WRF output files (names starting with 'wrfout')
#          and sort them based on the datetime string embedded in the filename.
#        - Loop over the sorted files and extract variables:
#            - 2-meter temperature (T2)
#            - Mixing ratio at 2 m (Q2)
#            - Surface pressure (PSFC)
#            - Land surface temperature (TSK)
#            - Downwelling shortwave radiation (SWDOWN)
#            - Downwelling longwave radiation (GLW)
#            - U and V wind components (U10, V10) to compute wind speed
#        - Calculate relative humidity using a helper function.
#        - Generate an hourly time array between start_time and end_time.
#        - Combine the data from all files along the time axis and save to a new NetCDF file.
# =============================================================================

def process_wrfout_data(start_time, end_time, folder_path, output_file="Outfile.nc"):
    """
    Process WRF output files to create meteorological forcing data.

    Parameters:
        start_time (str): Start datetime string in format "%Y-%m-%d %H:%M:%S".
        end_time (str): End datetime string in format "%Y-%m-%d %H:%M:%S".
        folder_path (str): Directory containing wrfout files.
        output_file (str): Output NetCDF file name.
    """
    start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

    def calculate_rh(t2, q2, psfc):
        """
        Calculate relative humidity from temperature, mixing ratio, and pressure.
        
        Args:
            t2 (np.ndarray): 2-meter temperature in Kelvin
            q2 (np.ndarray): Water vapor mixing ratio (kg/kg)
            psfc (np.ndarray): Surface pressure in Pascals
        
        Returns:
            np.ndarray: Relative humidity in percent [0-100]
        """
        # Compute saturation vapor pressure (in hPa) using temperature converted to Celsius.
        e_s = 6.112 * np.exp((17.67 * (t2 - 273.15)) / ((t2 - 273.15) + 243.5))
        e_s = e_s * 100  # convert hPa to Pa
        # Calculate actual vapor pressure using mixing ratio.
        Rd = 287.05  # Gas constant for dry air (J/kg/K)
        Rv = 461.5   # Gas constant for water vapor (J/kg/K)
        eps = Rd / Rv
        e = q2 * psfc / (eps + q2)
        rh = (e / e_s) * 100
        return np.clip(rh, 0, 100)
    
    # List and sort the WRF output files from the folder.
    # Files are assumed to be named like: "wrfout_d03_YYYY-MM-DD_HH:MM:SS"
    all_files = os.listdir(folder_path)
    #wrf_files = [f for f in all_files if f.startswith("wrfout")]
    
    # Define a helper to extract datetime from the filename.
    wrf_files = []
    for f in all_files:
        try:
            extract_datetime_strict(f)   # will raise if not valid
            wrf_files.append(f)
        except ValueError:
            continue

    if not wrf_files:
        raise FileNotFoundError(
            "No wrfout files matching the required patterns were found "
            "(wrfout_d0x_YYYY-MM-DD_HH_MM_SS | HH:MM:SS | HH with x=1..9)."
        )

    # Sort by timestamp, then by domain number for stable ordering
    wrf_files_sorted = sorted(wrf_files, key=lambda f: extract_datetime_strict(f))

    wrf_files_sorted = [f for f in wrf_files_sorted if start_time <= extract_datetime_strict(f)[0] <= end_time]   # extract_datetime_strict returns (datetime, domain): compare the datetime, not the tuple

    if not wrf_files_sorted:
        raise ValueError("No WRF files found within requested time window.")
    
    # Generate the time array for the simulation period (hourly frequency)
    total_hours = int((end_time - start_time).total_seconds() // 3600) + 1
    time_array = [start_time + timedelta(hours=i) for i in range(total_hours)]
    
    t2_list, wind_list, wdir_list, rh2_list, tsk_list = [], [], [], [], []
    swdown_list, glw_list, psfc_list = [], [], []
    lat, lon = None, None
    
    for file in wrf_files_sorted:
        file_path = os.path.join(folder_path, file)
        with xr.open_dataset(file_path) as ds:
            # Extract variables.
            t2 = ds['T2'].values           # 2-meter temperature (K)
            q2 = ds['Q2'].values           # Mixing ratio at 2 m (kg/kg)
            psfc = ds['PSFC'].values       # Surface pressure (Pa)
            
            t2_list.append(t2)
            tsk_list.append(ds['TSK'].values)       # Land surface temperature (K)
            swdown_list.append(ds['SWDOWN'].values)    # Downwelling shortwave radiation (W/m^2)
            #glw_list.append(ds['GLW'].values)          # Downwelling longwave radiation (W/m^2)
            psfc_list.append(psfc)
            
            # Calculate wind speed and wind direction from U10/V10.
            # For WRF projected grids, rotate grid-relative winds to Earth-relative winds first.
            u10 = ds['U10'].values
            v10 = ds['V10'].values
            
            if ('COSALPHA' in ds.variables) and ('SINALPHA' in ds.variables):
                cosalpha = ds['COSALPHA'].values
                sinalpha = ds['SINALPHA'].values
            
                # COSALPHA/SINALPHA may be either 2D or 3D depending on the file.
                # Make them broadcastable to U10/V10 shape: (Time, south_north, west_east)
                if cosalpha.ndim == 2:
                    cosalpha = cosalpha[None, :, :]
                    sinalpha = sinalpha[None, :, :]
            
                u10_earth = u10 * cosalpha - v10 * sinalpha
                v10_earth = v10 * cosalpha + u10 * sinalpha
            else:
                # Fallback: use raw U10/V10 directly.
                # This is okay only if they are already Earth-relative or grid rotation is negligible.
                u10_earth = u10
                v10_earth = v10
            
            wind_speed = np.sqrt(u10_earth**2 + v10_earth**2)
            
            # Meteorological wind direction:
            # degrees clockwise from true north, direction FROM which wind blows.
            wind_dir = (270.0 - np.degrees(np.arctan2(v10_earth, u10_earth))) % 360.0
            wind_dir = np.where(wind_speed > 0.01, wind_dir, 0.0)
            
            wind_list.append(wind_speed)
            wdir_list.append(wind_dir)
            
            # Calculate relative humidity using the helper function.
            rh2 = calculate_rh(t2, q2, psfc)
            rh2_list.append(rh2)
            
            # Extract latitude and longitude (assumed same for all files).
            if lat is None or lon is None:
                lat = ds['XLAT'].values[0, :, :]
                lon = ds['XLONG'].values[0, :, :]
    
    t2_array      = np.concatenate(t2_list, axis=0)
    wind_array    = np.concatenate(wind_list, axis=0)
    wdir_array    = np.concatenate(wdir_list, axis=0)
    rh2_array     = np.concatenate(rh2_list, axis=0)
    tsk_array     = np.concatenate(tsk_list, axis=0)
    swdown_array  = np.concatenate(swdown_list, axis=0)
    #glw_array     = np.concatenate(glw_list, axis=0)
    psfc_array    = np.concatenate(psfc_list, axis=0)
    
    # Create a new NetCDF file and write the combined data.
    with Dataset(output_file, 'w', format='NETCDF4') as nc:
        nc.createDimension('time', len(time_array))
        nc.createDimension('lat', lat.shape[0])
        nc.createDimension('lon', lon.shape[1])
        
        time_var = nc.createVariable('time', 'f8', ('time',))
        lat_var = nc.createVariable('lat', 'f4', ('lat', 'lon'))
        lon_var = nc.createVariable('lon', 'f4', ('lat', 'lon'))
        
        t2_var    = nc.createVariable('T2', 'f4', ('time', 'lat', 'lon'), zlib=True)
        wind_var  = nc.createVariable('WIND', 'f4', ('time', 'lat', 'lon'), zlib=True)
        wdir_var  = nc.createVariable('WDIR', 'f4', ('time', 'lat', 'lon'), zlib=True)
        rh2_var   = nc.createVariable('RH2', 'f4', ('time', 'lat', 'lon'), zlib=True)
        tsk_var   = nc.createVariable('TSK', 'f4', ('time', 'lat', 'lon'), zlib=True)
        swdown_var= nc.createVariable('SWDOWN', 'f4', ('time', 'lat', 'lon'), zlib=True)
     #   glw_var   = nc.createVariable('GLW', 'f4', ('time', 'lat', 'lon'), zlib=True)
        psfc_var  = nc.createVariable('PSFC', 'f4', ('time', 'lat', 'lon'), zlib=True)
        
        time_var.units = "hours since 1970-01-01 00:00:00"
        time_var.calendar = "gregorian"
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"
        
        t2_var.units = "K"
        wind_var.units = "m/s"
        wdir_var.units = "degrees"
        rh2_var.units = "%"
        tsk_var.units = "K"
        swdown_var.units = "W/m^2"
       # glw_var.units = "W/m^2"
        psfc_var.units = "Pa"
        
        time_var[:] = date2num(time_array, units=time_var.units, calendar=time_var.calendar)
        lat_var[:, :] = lat
        lon_var[:, :] = lon
        
        t2_var[:, :, :]    = t2_array
        wind_var[:, :, :]  = wind_array
        wdir_var[:, :, :]  = wdir_array
        rh2_var[:, :, :]   = rh2_array
        tsk_var[:, :, :]   = tsk_array
        swdown_var[:, :, :] = swdown_array
        #glw_var[:, :, :]    = glw_array
        psfc_var[:, :, :]   = psfc_array
    
    print(f"New NetCDF file created: {output_file}")
    
# =============================================================================
# HRLDAS (Noah-MP offline, LDASOUT) forcing
# =============================================================================
HRLDAS_PATTERN = re.compile(r'^(\d{10})\.LDASOUT_DOMAIN(\d)$')   # e.g. 2020081301.LDASOUT_DOMAIN3


def _hrldas_latlon(attrs, ny, nx):
    """
    Rebuild the 2-D latitude/longitude of an HRLDAS/WRF grid from its global attributes
    (LDASOUT files carry no XLAT/XLONG). Implements the WRF/WPS ``module_llxy`` formulas on
    the 6370 km sphere for MAP_PROJ 1 (Lambert conformal), 3 (Mercator) and 6 (regular lat/lon).
    ``LAT1``/``LON1`` are the coordinates of grid point (1, 1) = array index [0, 0].

    Returns:
        tuple: (lat2d, lon2d) float arrays of shape (ny, nx), south_north x west_east.
    """
    rad = math.pi / 180.0
    deg = 180.0 / math.pi
    re_m = 6370000.0
    proj = int(attrs['MAP_PROJ'])
    lat1, lon1 = float(attrs['LAT1']), float(attrs['LON1'])
    dx = float(attrs['DX'])
    stdlon = float(attrs.get('STAND_LON', lon1))
    truelat1 = float(attrs.get('TRUELAT1', lat1))
    truelat2 = float(attrs.get('TRUELAT2', truelat1))
    jj, ii = np.meshgrid(np.arange(1, ny + 1, dtype=float), np.arange(1, nx + 1, dtype=float), indexing='ij')

    if proj == 6:   # regular lat/lon: DX/DY are degrees in LDASOUT geogrid files
        dy = float(attrs.get('DY', dx))
        lat = lat1 + (jj - 1) * dy
        lon = lon1 + (ii - 1) * dx
        return lat, ((lon + 180.0) % 360.0) - 180.0

    rebydx = re_m / dx
    if proj == 3:   # Mercator
        clain = math.cos(rad * truelat1)
        dlon = dx / (re_m * clain)
        rsw = 0.0 if lat1 == 0.0 else math.log(math.tan(0.5 * ((lat1 + 90.0) * rad))) / dlon
        lat = 2.0 * np.arctan(np.exp(dlon * (rsw + jj - 1.0))) * deg - 90.0
        lon = (ii - 1.0) * dlon * deg + lon1
        return lat, ((lon + 180.0) % 360.0) - 180.0

    if proj != 1:
        raise ValueError(f"MAP_PROJ={proj} is not supported (only Lambert conformal, Mercator and lat/lon)")

    # Lambert conformal (WRF set_lc / xyll_lc)
    hemi = 1.0 if truelat1 >= 0.0 else -1.0
    if abs(truelat1 - truelat2) > 0.1:
        cone = (math.log10(math.cos(truelat1 * rad)) - math.log10(math.cos(truelat2 * rad))) / \
               (math.log10(math.tan((45.0 - abs(truelat1) / 2.0) * rad)) - math.log10(math.tan((45.0 - abs(truelat2) / 2.0) * rad)))
    else:
        cone = math.sin(abs(truelat1) * rad)
    deltalon1 = lon1 - stdlon
    if deltalon1 > 180.0: deltalon1 -= 360.0
    if deltalon1 < -180.0: deltalon1 += 360.0
    ctl1r = math.cos(truelat1 * rad)
    rsw = rebydx * ctl1r / cone * (math.tan((90.0 * hemi - lat1) * rad / 2.0) / math.tan((90.0 * hemi - truelat1) * rad / 2.0)) ** cone
    arg = cone * (deltalon1 * rad)
    polei = hemi * 1.0 - hemi * rsw * math.sin(arg)      # grid point (1, 1) is (LAT1, LON1)
    polej = hemi * 1.0 + rsw * math.cos(arg)
    chi1 = (90.0 - hemi * truelat1) * rad
    chi2 = (90.0 - hemi * truelat2) * rad
    xx = hemi * ii - polei
    yy = polej - hemi * jj
    r2 = xx ** 2 + yy ** 2
    r = np.sqrt(r2) / rebydx
    lon = stdlon + deg * np.arctan2(hemi * xx, yy) / cone
    lon = np.mod(lon + 360.0, 360.0)
    if chi1 == chi2:
        chi = 2.0 * np.arctan((r / math.tan(chi1)) ** (1.0 / cone) * math.tan(chi1 * 0.5))
    else:
        chi = 2.0 * np.arctan((r * cone / math.sin(chi1)) ** (1.0 / cone) * math.tan(chi1 * 0.5))
    lat = (90.0 - chi * deg) * hemi
    lat = np.where(r2 == 0.0, hemi * 90.0, lat)
    lon = np.where(r2 == 0.0, stdlon, lon)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    return lat, lon


def _hrldas_file_times(ds):
    """Decode the ``Times`` character variable of an LDASOUT file into naive UTC datetimes."""
    raw = ds['Times'].values
    out = []
    for row in raw:
        s = b''.join(row).decode() if raw.dtype.kind == 'S' else ''.join(row.astype(str))
        out.append(datetime.datetime.strptime(s.strip('\x00 '), "%Y-%m-%d_%H:%M:%S"))
    return out


def process_hrldas_data(start_time, end_time, folder_path, output_file="Outfile.nc"):
    """
    Process HRLDAS (Noah-MP offline) LDASOUT files to create meteorological forcing data.

    Reads every ``YYYYMMDDHH.LDASOUT_DOMAINx`` file in ``folder_path`` (each holds one or more
    hourly steps, dated by its ``Times`` variable, not by the filename), keeps the steps inside
    ``[start_time, end_time]`` and writes the same ``Outfile.nc`` that :func:`process_wrfout_data`
    produces, so the rest of the pipeline (:func:`process_metfiles`) is unchanged.

    Variable mapping (first available wins; a message is printed when a fallback is used):
      T2      <- T2M (urban diagnostics build), else FVEG-weighted T2MV/T2MB, else FORCTLSM
      Q       <- Q2M, else FVEG-weighted Q2MV/Q2MB, else FORCQLSM   (specific humidity -> mixing ratio for RH2)
      PSFC    <- FORCPLSM reduced hydrostatically from FORCZLSM to the surface, else 101325 Pa
      WIND    <- WS10M, else FORCWLSM (lowest model level, not 10 m)
      SWDOWN  <- SWFORC;   GLW <- LWFORC (written, not used downstream)
      TSK     <- TRAD, else TG, else T2
      WDIR    <- not available in LDASOUT (no wind components): written as NaN -> Wd = -999 in the metfiles,
                 which disables the directional wind-coefficient lookup.
    lat/lon come from XLAT/XLONG when present, otherwise from the projection attributes (:func:`_hrldas_latlon`).
    The -1e33 ``missing_value`` and the -9999 sentinel of the 2 m diagnostics (water cells) become NaN.

    Parameters:
        start_time (str): Start datetime (UTC) "%Y-%m-%d %H:%M:%S".
        end_time (str): End datetime (UTC) "%Y-%m-%d %H:%M:%S".
        folder_path (str): Directory containing LDASOUT files.
        output_file (str): Output NetCDF file name.
    """
    start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    G, RD, RV = 9.80665, 287.05, 461.5
    eps = RD / RV

    files = sorted(f for f in os.listdir(folder_path) if HRLDAS_PATTERN.match(f))
    if not files:
        raise FileNotFoundError("No HRLDAS files (YYYYMMDDHH.LDASOUT_DOMAINx) found in " + str(folder_path))

    def clean(arr):
        arr = np.asarray(arr, dtype=np.float32)
        return np.where((arr < -1e30) | (arr <= -9998.0), np.nan, arr)

    def pick(ds, t, *names):
        for n in names:
            if n in ds.variables:
                return n, clean(ds[n].isel(Time=t).values)
        return None, None

    def blended(ds, t, veg, bare):
        """FVEG-weighted vegetated/bare-ground diagnostics."""
        if veg in ds.variables and bare in ds.variables and 'FVEG' in ds.variables:
            f = clean(ds['FVEG'].isel(Time=t).values)
            return f * clean(ds[veg].isel(Time=t).values) + (1.0 - f) * clean(ds[bare].isel(Time=t).values)
        return None

    times, t2_l, w_l, rh_l, tsk_l, sw_l, glw_l, p_l = [], [], [], [], [], [], [], []
    lat = lon = None
    notes = set()
    for f in files:
        with xr.open_dataset(os.path.join(folder_path, f), decode_times=False, concat_characters=False) as ds:
            ftimes = _hrldas_file_times(ds)
            idx = [i for i, ft in enumerate(ftimes) if start_time <= ft <= end_time]
            if not idx:
                continue
            if lat is None:
                ny, nx = ds.sizes['south_north'], ds.sizes['west_east']
                if 'XLAT' in ds.variables and 'XLONG' in ds.variables:
                    lat = np.asarray(ds['XLAT'].values)[0] if ds['XLAT'].ndim == 3 else np.asarray(ds['XLAT'].values)
                    lon = np.asarray(ds['XLONG'].values)[0] if ds['XLONG'].ndim == 3 else np.asarray(ds['XLONG'].values)
                else:
                    lat, lon = _hrldas_latlon(ds.attrs, ny, nx)
            for t in idx:
                name, t2 = pick(ds, t, 'T2M')
                if t2 is None:
                    t2 = blended(ds, t, 'T2MV', 'T2MB'); name = 'FVEG*T2MV+(1-FVEG)*T2MB'
                if t2 is None:
                    name, t2 = pick(ds, t, 'FORCTLSM')
                notes.add(f"T2 <- {name}")
                name, q = pick(ds, t, 'Q2M')
                if q is None:
                    q = blended(ds, t, 'Q2MV', 'Q2MB'); name = 'FVEG*Q2MV+(1-FVEG)*Q2MB'
                if q is None:
                    name, q = pick(ds, t, 'FORCQLSM')
                notes.add(f"Q <- {name}")
                name, p = pick(ds, t, 'FORCPLSM')
                if p is not None and 'FORCZLSM' in ds.variables and 'FORCTLSM' in ds.variables:
                    z = clean(ds['FORCZLSM'].isel(Time=t).values)
                    tf = clean(ds['FORCTLSM'].isel(Time=t).values)
                    p = p * np.exp(G * z / (RD * tf))
                    name = 'FORCPLSM reduced from FORCZLSM'
                if p is None:
                    p = np.full_like(t2, 101325.0); name = 'constant 101325 Pa (no FORCPLSM)'
                notes.add(f"PSFC <- {name}")
                name, ws = pick(ds, t, 'WS10M', 'FORCWLSM')
                notes.add(f"WIND <- {name}")
                _, sw = pick(ds, t, 'SWFORC')
                _, glw = pick(ds, t, 'LWFORC')
                name, tsk = pick(ds, t, 'TRAD', 'TG')
                if tsk is None:
                    tsk, name = t2, 'T2'
                notes.add(f"TSK <- {name}")
                if t2 is None or q is None or ws is None or sw is None:
                    raise KeyError(f"{f}: cannot build T2/Q/WIND/SWDOWN from the available LDASOUT variables")
                w = q / (1.0 - q)                                     # specific humidity -> mixing ratio
                e_s = 6.112 * np.exp((17.67 * (t2 - 273.15)) / ((t2 - 273.15) + 243.5)) * 100.0
                rh = np.clip((w * p / (eps + w)) / e_s * 100.0, 0, 100)
                times.append(ftimes[t]); t2_l.append(t2); w_l.append(ws); rh_l.append(rh); tsk_l.append(tsk)
                sw_l.append(sw); glw_l.append(glw if glw is not None else np.full_like(t2, np.nan)); p_l.append(p)

    if not times:
        raise ValueError("No HRLDAS time steps found within requested time window.")
    order = np.argsort(np.array(times, dtype='datetime64[s]'))
    times = [times[i] for i in order]
    stack = lambda lst: np.stack([lst[i] for i in order], axis=0).astype(np.float32)
    print("[INFO] HRLDAS mapping: " + "; ".join(sorted(notes)))
    gaps = [(times[i], times[i + 1]) for i in range(len(times) - 1) if (times[i + 1] - times[i]) != timedelta(hours=1)]
    if gaps:
        print(f"[WARNING] HRLDAS series is not hourly-continuous at {len(gaps)} place(s), e.g. {gaps[0][0]} -> {gaps[0][1]}")

    with Dataset(output_file, 'w', format='NETCDF4') as ncf:
        ncf.createDimension('time', len(times))
        ncf.createDimension('lat', lat.shape[0])
        ncf.createDimension('lon', lon.shape[1])
        time_var = ncf.createVariable('time', 'f8', ('time',))
        lat_var = ncf.createVariable('lat', 'f4', ('lat', 'lon'))
        lon_var = ncf.createVariable('lon', 'f4', ('lat', 'lon'))
        out_vars = {}
        for name, units in [('T2', 'K'), ('WIND', 'm/s'), ('WDIR', 'degrees'), ('RH2', '%'), ('TSK', 'K'),
                            ('SWDOWN', 'W/m^2'), ('GLW', 'W/m^2'), ('PSFC', 'Pa')]:
            out_vars[name] = ncf.createVariable(name, 'f4', ('time', 'lat', 'lon'), zlib=True)
            out_vars[name].units = units
        time_var.units = "hours since 1970-01-01 00:00:00"
        time_var.calendar = "gregorian"
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"
        time_var[:] = date2num(times, units=time_var.units, calendar=time_var.calendar)
        lat_var[:, :] = lat
        lon_var[:, :] = lon
        out_vars['T2'][:] = stack(t2_l)
        out_vars['WIND'][:] = stack(w_l)
        out_vars['WDIR'][:] = np.full((len(times),) + lat.shape, np.nan, dtype=np.float32)   # no wind components in LDASOUT
        out_vars['RH2'][:] = stack(rh_l)
        out_vars['TSK'][:] = stack(tsk_l)
        out_vars['SWDOWN'][:] = stack(sw_l)
        out_vars['GLW'][:] = stack(glw_l)
        out_vars['PSFC'][:] = stack(p_l)
        ncf.source = "HRLDAS LDASOUT files from " + str(folder_path)

    print(f"New NetCDF file created: {output_file} ({len(times)} hourly steps, {times[0]} .. {times[-1]} UTC)")

# =============================================================================
# Functions to process the NetCDF file and create metfiles based on a set of raster tiles.
# =============================================================================
def _haversine_m(lat1, lon1, lat2, lon2):
    """
    Calculate great circle distance between two points on Earth using Haversine formula.
    
    Args:
        lat1 (float): Latitude of first point (degrees)
        lon1 (float): Longitude of first point (degrees)
        lat2 (float): Latitude of second point (degrees)
        lon2 (float): Longitude of second point (degrees)
    
    Returns:
        float: Distance in meters
    """
    # distance in meters
    R = 6371000.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _local_cell_size_m(lon2d, lat2d, cx, cy, tree):
    """
    Estimate local grid cell size in meters at a given coordinate.
    
    Args:
        lon2d (np.ndarray): 2D array of longitudes
        lat2d (np.ndarray): 2D array of latitudes
        cx (float): Center longitude
        cy (float): Center latitude
        tree (cKDTree): KDTree for nearest neighbor lookup
    
    Returns:
        tuple: (cell_width_m, cell_height_m) in meters
    """
    ny, nx = lat2d.shape
    _, idx = tree.query([cx, cy], k=1)
    i, j = np.unravel_index(idx, (ny, nx))
    ew, ns = [], []
    def dist(i1, j1, i2, j2):
        """Calculate haversine distance between two grid cells."""
        return _haversine_m(lat2d[i1,j1], lon2d[i1,j1], lat2d[i2,j2], lon2d[i2,j2])
    if j-1 >= 0: ew.append(dist(i,j, i, j-1))
    if j+1 < nx: ew.append(dist(i,j, i, j+1))
    if i-1 >= 0: ns.append(dist(i,j, i-1, j))
    if i+1 < ny: ns.append(dist(i,j, i+1, j))
    anyd = ew + ns
    if not anyd:
        return 1e30, 1e30
    cell_w = np.median(ew) if ew else np.median(anyd)
    cell_h = np.median(ns) if ns else np.median(anyd)
    return cell_w, cell_h

def _tile_size_m(poly):
    """
    Calculate tile size in meters from polygon bounds.
    
    Args:
        poly (Polygon): Shapely polygon representing tile extent
    
    Returns:
        tuple: (width_m, height_m) tile dimensions in meters
    """
    minx, miny, maxx, maxy = poly.bounds
    cx, cy = (minx+maxx)/2.0, (miny+maxy)/2.0
    w = _haversine_m(cy, minx, cy, maxx) 
    h = _haversine_m(miny, cx, maxy, cx) 
    return w, h

def process_metfiles(netcdf_file, raster_folder, base_path, selected_date_str, preprocess_dir, use_uhi=True):

    metfiles_folder = os.path.join(preprocess_dir, "metfiles")
    os.makedirs(metfiles_folder, exist_ok=True)
    
    tf = TimezoneFinder()
    dataset = nc.Dataset(netcdf_file, "r")
    
    tif_files = glob.glob(os.path.join(raster_folder, "*.tif"))
    if not tif_files:
        print(f"No TIFF files found in {raster_folder}.")
        dataset.close()
        return

    var_map = {
        "Wind": "WIND",
        "RH": "RH2",
        "Td": "T2",           # K -> °C
        "press": "PSFC",      # Pa -> kPa
        "Kdn": "SWDOWN",
        "Wd": "WDIR",         # meteorological wind direction, wind FROM direction
        "uhii": "UHI_CYCLE"
    }

    fixed_values = {
        "Q*": -999, "QH": -999, "QE": -999, "Qs": -999, "Qf": -999,
        "snow": -999, "ldown": -999, "fcld": -999, "wuh": -999, "xsmd": -999, "lai_hr": -999,
        "Kdiff": -999, "Kdir": -999, "Wd": -999,
        "rain": 0
    }
    
    time_var = dataset.variables["time"][:]
    time_units = dataset.variables["time"].units
    time_base_date = nc.num2date(time_var, units=time_units, only_use_cftime_datetimes=False)
    selected_local_date = datetime.datetime.strptime(selected_date_str, "%Y-%m-%d").date()

    lat2d = np.array(dataset.variables["lat"][:], dtype=float)
    lon2d = np.array(dataset.variables["lon"][:], dtype=float)
    ny, nx = lat2d.shape
    pts_flat = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    tree = cKDTree(pts_flat)

    columns = [
        'iy', 'id', 'it', 'imin',
        'Q*', 'QH', 'QE', 'Qs', 'Qf',
        'Wind', 'RH', 'Td', 'press',
        'Kdn', 'rain', 'snow', 'ldown',
        'fcld', 'wuh', 'xsmd', 'lai_hr',
        'Kdiff', 'Kdir', 'Wd', 'uhii'
    ]

    columns_out = [
        "iy", "id", "it", "imin",
        "Q*", "QH", "QE", "Qs", "Qf",
        "Wind", "RH", "Td", "press",
        "rain",
        "Kdn",
        "snow",
        "ldown",
        "fcld",
        "wuh",
        "xsmd",
        "lai_hr",
        "Kdiff",
        "Kdir",
        "Wd",
        "uhii"
    ]
    
    for tif_file in tif_files:
        ds_tif = gdal.Open(tif_file)
        if ds_tif is None:
            print(f"Could not open {tif_file}. Skipping.")
            continue

        gt_tif = ds_tif.GetGeoTransform()
        xsize = ds_tif.RasterXSize
        ysize = ds_tif.RasterYSize

        proj_tif = ds_tif.GetProjection()
        srs_tif = osr.SpatialReference()
        srs_tif.ImportFromWkt(proj_tif)
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(4326)
        srs_tif.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osr.CoordinateTransformation(srs_tif, target_srs)

        left = gt_tif[0]
        top = gt_tif[3]
        right = left + gt_tif[1] * xsize
        bottom = top + gt_tif[5] * ysize
        corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
        try:
            lonlat_corners = [transform.TransformPoint(x, y) for x, y in corners]
            lons = [pt[0] for pt in lonlat_corners]
            lats = [pt[1] for pt in lonlat_corners]
        except Exception as e:
            print(f"Warning: CRS transform failed for {tif_file}. Assuming EPSG:4326. Error: {e}")
            lons = [p[0] for p in corners]
            lats = [p[1] for p in corners]

        min_lon_tif, max_lon_tif = min(lons), max(lons)
        min_lat_tif, max_lat_tif = min(lats), max(lats)
        shape = box(min_lon_tif, min_lat_tif, max_lon_tif, max_lat_tif)
        shape = Polygon([(x, y) for (x, y) in shape.exterior.coords])

        shape_name = os.path.splitext(os.path.basename(tif_file))[0]
        shape_name_clean = re.sub(r'\W+', '_', shape_name).replace("DEM", "metfile", 1)
        output_text_file = os.path.join(metfiles_folder, f"{shape_name_clean}_{selected_date_str}.txt")
        
        lat_center, lon_center = shape.centroid.y, shape.centroid.x
        timezone_name = tf.timezone_at(lng=lon_center, lat=lat_center) or "UTC"
        local_tz = pytz.timezone(timezone_name)
        
        local_start = local_tz.localize(datetime.datetime.combine(selected_local_date, datetime.time(0, 0)))
        local_end = local_tz.localize(datetime.datetime.combine(selected_local_date, datetime.time(23, 59)))
        utc_start = local_start.astimezone(pytz.utc)
        utc_end = local_end.astimezone(pytz.utc)
        
        time_indices = [
            idx for idx, dt in enumerate(time_base_date)
            if utc_start <= dt.replace(tzinfo=pytz.utc) <= utc_end
        ]
        if not time_indices:
            print(f"No UTC data found for local date {selected_date_str} in {tif_file}.")
            ds_tif = None
            continue

        print(f"Processing {len(time_indices)} time steps for {shape_name_clean}")

        tile_w_m, tile_h_m = _tile_size_m(shape)
        cell_w_m, cell_h_m = _local_cell_size_m(lon2d, lat2d, lon_center, lat_center, tree)
        use_nn = (cell_w_m > tile_w_m) and (cell_h_m > tile_h_m)

        inside_mask = None
        if not use_nn:
            path = Path(np.asarray(shape.exterior.coords)[:, :2])
            inside_mask = path.contains_points(np.column_stack([lon2d.ravel(), lat2d.ravel()])).reshape(lat2d.shape)
            if not np.any(inside_mask):
                use_nn = True
                
        met_new = []
        for t in time_indices:
            utc_time = time_base_date[t].replace(tzinfo=pytz.utc)
            local_time = utc_time.astimezone(local_tz)
            year = local_time.year
            doy = local_time.timetuple().tm_yday
            hour = local_time.hour
            minute = local_time.minute

            row = [year, doy, hour, minute]
            row.extend([fixed_values[key] for key in ["Q*", "QH", "QE", "Qs", "Qf"]])
            
            for key in ["Wind", "RH", "Td", "press", "Kdn"]:
                var_name = var_map[key]
                if var_name in dataset.variables:
                    try:
                        data_array = dataset.variables[var_name][t, :, :]
                        data_array = np.asanyarray(data_array)
                        if np.ma.isMaskedArray(data_array):
                            data_array = np.where(data_array.mask, np.nan, data_array.data)

                        if use_nn:
                            _, idx_nn = tree.query([lon_center, lat_center], k=1)
                            ii, jj = np.unravel_index(idx_nn, (ny, nx))
                            mean_value = float(data_array[ii, jj])
                        else:
                            masked_data = np.where(inside_mask, data_array, np.nan)
                            mean_value = float(np.nanmean(masked_data)) if np.any(~np.isnan(masked_data)) else np.nan
                            if not np.isfinite(mean_value):
                                _, idx_nn = tree.query([lon_center, lat_center], k=1)
                                ii, jj = np.unravel_index(idx_nn, (ny, nx))
                                mean_value = float(data_array[ii, jj])

                        if key == "Td" and np.isfinite(mean_value):
                            mean_value -= 273.15
                        if key == "press" and np.isfinite(mean_value):
                            mean_value /= 1000.0

                        if not np.isfinite(mean_value):
                            mean_value = -999

                        row.append(mean_value)

                    except Exception as e:
                        print(f"Sampling error for {var_name} at time {t}: {e}")
                        row.append(-999)
                else:
                    row.append(-999)

            row.append(fixed_values["rain"])
            row.extend([fixed_values[key] for key in ["snow", "ldown", "fcld", "wuh", "xsmd", "lai_hr", "Kdiff", "Kdir"]])
            
            wd_value = -999.0
            var_name = var_map["Wd"]

            if var_name in dataset.variables:
                try:
                    data_array = dataset.variables[var_name][t, :, :]
                    data_array = np.asanyarray(data_array)

                    if np.ma.isMaskedArray(data_array):
                        data_array = np.where(data_array.mask, np.nan, data_array.data)

                    if use_nn:
                        _, idx_nn = tree.query([lon_center, lat_center], k=1)
                        ii, jj = np.unravel_index(idx_nn, (ny, nx))
                        wd_value = float(data_array[ii, jj])
                    else:
                        masked_data = np.where(inside_mask, data_array, np.nan)
                        wd_value = float(np.nanmean(masked_data)) if np.any(~np.isnan(masked_data)) else np.nan

                        if not np.isfinite(wd_value):
                            _, idx_nn = tree.query([lon_center, lat_center], k=1)
                            ii, jj = np.unravel_index(idx_nn, (ny, nx))
                            wd_value = float(data_array[ii, jj])

                    if np.isfinite(wd_value):
                        wd_value = wd_value % 360.0
                    else:
                        wd_value = -999.0

                except Exception as e:
                    print(f"Sampling error for {var_name} at time {t}: {e}")
                    wd_value = -999.0

            row.append(wd_value)

            # Add uhii at the end
            uhii_value = 0.0
            if use_uhi and (var_map["uhii"] in dataset.variables):
                try:
                    data_array = dataset.variables[var_map["uhii"]][t, :, :]
                    data_array = np.asanyarray(data_array)
                    if np.ma.isMaskedArray(data_array):
                        data_array = np.where(data_array.mask, np.nan, data_array.data)

                    if use_nn:
                        _, idx_nn = tree.query([lon_center, lat_center], k=1)
                        ii, jj = np.unravel_index(idx_nn, (ny, nx))
                        uhii_value = float(data_array[ii, jj])
                    else:
                        masked_data = np.where(inside_mask, data_array, np.nan)
                        uhii_value = float(np.nanmean(masked_data)) if np.any(~np.isnan(masked_data)) else np.nan
                        if not np.isfinite(uhii_value):
                            _, idx_nn = tree.query([lon_center, lat_center], k=1)
                            ii, jj = np.unravel_index(idx_nn, (ny, nx))
                            uhii_value = float(data_array[ii, jj])

                    if not np.isfinite(uhii_value):
                        uhii_value = 0.0

                except Exception as e:
                    print(f"Sampling error for UHI_CYCLE at time {t}: {e}")
                    uhii_value = 0.0

            row.append(uhii_value)
            met_new.append(row)

        df = pd.DataFrame(met_new, columns=columns)
        df = df[columns_out]

        with open(output_text_file, "w") as f:
            f.write(" ".join(df.columns) + "\n")
            for _, row in df.iterrows():
                f.write(
                    "{:d} {:d} {:d} {:d} "
                    "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} "
                    "{:.5f} {:.2f} {:.2f} {:.2f} "
                    "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n".format(
                        int(row["iy"]), int(row["id"]), int(row["it"]), int(row["imin"]),
                        row["Q*"], row["QH"], row["QE"], row["Qs"], row["Qf"],
                        row["Wind"], row["RH"], row["Td"], row["press"],
                        row["rain"], row["Kdn"], row["snow"], row["ldown"], row["fcld"],
                        row["wuh"], row["xsmd"], row["lai_hr"], row["Kdiff"], row["Kdir"],
                        row["Wd"], row["uhii"]
                    )
                )

        print(f"Metfile saved: {output_text_file}")
        ds_tif = None

    dataset.close()
    print(f"All raster extents processed and metfiles saved in {metfiles_folder}")
    
# =============================================================================
# Function to process own met file: copies the source met file into new files
# renaming each copy based on the numeric suffix extracted from .tif files.
# =============================================================================
def create_met_files(base_path, source_met_file, preprocess_dir):
    """
    Copy a given met file to multiple outputs based on the raster tile filenames.

    Parameters:
        base_path (str): Base directory containing input rasters.
        source_met_file (str): Path to user-provided met file.
        preprocess_dir (str): Directory for preprocessing outputs (pre_processing_outputs).
    """
    raster_folder = os.path.join(preprocess_dir, 'Building_DSM')
    target_folder = os.path.join(preprocess_dir, 'metfiles')

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    else:
        shutil.rmtree(target_folder)
        os.makedirs(target_folder)
    
    for file in os.listdir(raster_folder):
        if file.lower().endswith('.tif'):
            name_without_ext = os.path.splitext(file)[0]
            prefix = 'Building_DSM_'
            if name_without_ext.startswith(prefix):
                digits = name_without_ext[len(prefix):]
                new_filename = f'metfile_{digits}.txt'
                target_met_file = os.path.join(target_folder, new_filename)
                shutil.copy(source_met_file, target_met_file)
                print(f"Copied to {target_met_file}")

# =============================================================================
# Main function: checks rasters, creates tiles, and creates metfiles using either a
# user-supplied met file or a netCDF file. Only the parameters required for the chosen
# method need to be provided.
# =============================================================================
def ppr(base_path, building_dsm_filename, dem_filename, trees_filename,
         landcover_filename, windcoeff_filename,
         tile_size, overlap, selected_date_str, use_own_met,
         start_time=None, end_time=None, data_source_type=None, data_folder=None,
         own_met_file=None, preprocess_dir=None, use_uhi=True):
    """
    Preprocessing routine to validate raster files, generate tiles, and prepare metfiles for SOLWEIG.

    Parameters:
        base_path (str): Base working directory containing input rasters.
        building_dsm_filename (str): Filename of building DSM raster.
        dem_filename (str): Filename of DEM raster.
        trees_filename (str): Filename of trees raster.
        landcover_filename (str): Filename of landcover raster or None.
        windcoeff_filename (str): Filename of wind coefficient raster or None.
        tile_size (int): Tile size in pixels.
        overlap (int): Overlap between tiles in pixels.
        selected_date_str (str): Selected date (YYYY-MM-DD).
        use_own_met (bool): Whether to use a user-provided met file.
        start_time (str): Start datetime (required if not using own met file).
        end_time (str): End datetime (required if not using own met file).
        data_source_type (str): 'ERA5', 'wrfout' or 'hrldas' (HRLDAS/Noah-MP LDASOUT files).
        data_folder (str): Folder containing input NetCDF files.
        own_met_file (str): Path to user-provided met file (used if use_own_met is True).
        preprocess_dir (str): Directory for preprocessing outputs.
        use_uhi (bool): Whether to use UHI-aware ERA5 preprocessing.
    """
    if preprocess_dir is None:
        preprocess_dir = os.path.join(base_path, "processed_inputs")
    os.makedirs(preprocess_dir, exist_ok=True)

    building_dsm_path = os.path.join(base_path, building_dsm_filename)
    dem_path = os.path.join(base_path, dem_filename)
    trees_path = os.path.join(base_path, trees_filename)

    landcover_path = None
    landcover_path = None
    windcoeff_files = []

    if landcover_filename is not None:
        landcover_path = _resolve_path(base_path, landcover_filename)

    windcoeff_files = find_windcoeff_files(base_path, windcoeff_filename)

    # Check that all rasters have matching dimensions, pixel size, and CRS.
    try:
        raster_list = [building_dsm_path, dem_path, trees_path]

        if landcover_path is not None:
            raster_list.append(landcover_path)

        # Check all directional wind coefficient rasters too
        if windcoeff_files:
            raster_list.extend(windcoeff_files)

        check_rasters(raster_list)

    except ValueError as error:
        print(error)
        exit(1)

    rasters = {
        "Building_DSM": building_dsm_path,
        "DEM": dem_path,
        "Trees": trees_path
    }

    if landcover_path is not None:
        rasters["Landcover"] = landcover_path

    for tile_type, raster in rasters.items():
        print(f"Creating tiles for {tile_type}...")
        create_tiles(raster, tile_size, overlap, tile_type, preprocess_dir)

    # Directional wind coefficient rasters are handled separately
    # so the 12 files do not overwrite each other.
    if windcoeff_files:
        create_windcoeff_tiles(windcoeff_files, tile_size, overlap, preprocess_dir)
    else:
        print("Wind coefficient not used; skipping WindCoeff tiles.")

    # For metfiles processing, we use the DEM tiles folder.
    dem_tiles_folder = os.path.join(preprocess_dir, "DEM")

    # Choose between own met file or processed NetCDF file.
    if use_own_met:
        if own_met_file is None:
            print("Error: Please provide the path to your own met file.")
            exit(1)
        create_met_files(base_path, own_met_file, preprocess_dir)
    else:
        # Ensure all additional required parameters are provided.
        if data_folder is None or data_source_type is None or start_time is None or end_time is None:
            print("Error: When not using your own met file, please provide data_folder, data_source_type, start_time, and end_time.")
            exit(1)

        # Define the name (and path) for the processed NetCDF output.
        processed_nc_file = os.path.join(preprocess_dir, "Outfile.nc")

        if data_source_type.lower() == "era5":
            if use_uhi:
                print("[INFO] Using ERA5 processing with UHI correction")
                process_era5_data_uhi(start_time, end_time, data_folder, output_file=processed_nc_file)
            else:
                print("[INFO] Using standard ERA5 processing (no UHI)")
                process_era5_data(start_time, end_time, data_folder, output_file=processed_nc_file)

        elif data_source_type.lower() == "wrfout":
            process_wrfout_data(start_time, end_time, data_folder, output_file=processed_nc_file)

        elif data_source_type.lower() == "hrldas":
            process_hrldas_data(start_time, end_time, data_folder, output_file=processed_nc_file)

        else:
            print("Error: data_source_type must be 'ERA5', 'wrfout' or 'hrldas'.")
            exit(1)

        # Process the generated NetCDF file to create metfiles.
        process_metfiles(processed_nc_file, dem_tiles_folder, base_path, selected_date_str, preprocess_dir, use_uhi=use_uhi)









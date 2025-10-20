"""
Preprocessing utilities for SOLWEIG inputs.

Builds meteorological forcing (metfiles) from ERA5 or COSMO sources and
prepares per-tile rasters (e.g., HGT − DEM diffs).
All comments are in English for clarity.
"""

import os
import glob
import json
import shutil
from datetime import datetime, time, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytz
from netCDF4 import Dataset, date2num, num2date
from timezonefinder import TimezoneFinder
from osgeo import gdal, osr


# ---- Small shared helpers -------------------------------------------------

def saturation_vapor_pressure_C(T_c: np.ndarray | float) -> np.ndarray | float:
    """Saturation vapor pressure (hPa) for temperature in Celsius.
    Tetens formula used consistently across inputs.
    """
    return 6.112 * np.exp((17.67 * T_c) / (T_c + 243.5))


def rh_from_t_q_p(t2_k: np.ndarray, q2: np.ndarray, psfc_pa: np.ndarray) -> np.ndarray:
    """Relative humidity (%) from temperature (K), specific humidity (kg/kg), and surface pressure (Pa)."""
    e_s_pa = saturation_vapor_pressure_C(t2_k - 273.15) * 100.0
    Rd, Rv = 287.05, 461.5
    eps = Rd / Rv
    e_pa = q2 * psfc_pa / (eps + q2)
    rh = (e_pa / e_s_pa) * 100.0
    return np.clip(rh, 0, 100)


NC = Dataset


def create_hgt_dem_diff_tile(nc_path, dem_tile_array, center_lon, center_lat,
                               r_start, r_end, c_start, c_end,
                               base_path):
    """
    Create a tile GeoTIFF: HGT(nearest to tile center) - DEM(tile window).
    - Reads HGT and lat/lon from processed NetCDF (nc_path)
    - Uses DEM.tif in base_path to fetch full geotransform/projection for tile georef
    - Writes HGT_minus_DEM_{r0}_{r1}_{c0}_{c1}.tif to base_path
    """
    if not os.path.exists(nc_path):
        print(f"[HGT-DEM] NetCDF not found: {nc_path}")
        return False

    # Open NC and extract HGT and lat/lon
    with NC(nc_path, 'r') as nc:
        if 'HGT' not in nc.variables:
            print("[HGT-DEM] HGT not found in NetCDF; skipping diff tile.")
            return False
        hgt = nc.variables['HGT'][:]
        # Try common latitude/longitude variable names
        lat_name = 'lat' if 'lat' in nc.variables else ('latitude' if 'latitude' in nc.variables else None)
        lon_name = 'lon' if 'lon' in nc.variables else ('longitude' if 'longitude' in nc.variables else None)
        if lat_name is None or lon_name is None:
            print("[HGT-DEM] lat/lon variables not found in NetCDF; expected 'lat'/'lon' or 'latitude'/'longitude'.")
            return False
        latv  = nc.variables[lat_name][:]
        lonv  = nc.variables[lon_name][:]

    # Make sure lat/lon are 2D
    if latv.ndim == 1 and lonv.ndim == 1:
        lonv, latv = np.meshgrid(lonv, latv)

    # If lat/lon 2D shapes do not match HGT, try transposed alignment
    if latv.ndim == 2 and hgt.ndim == 2 and latv.shape != hgt.shape:
        if latv.T.shape == hgt.shape:
            latv = latv.T
            lonv = lonv.T

    # Nearest gridpoint to (center_lat, center_lon)
    d2 = (latv - center_lat) ** 2 + (lonv - center_lon) ** 2
    iy, ix = np.unravel_index(np.argmin(d2), d2.shape)
    h_val = float(hgt[iy, ix])

    # Build output array: HGT_nearest - DEM_tile
    dem_tile = np.asarray(dem_tile_array, dtype=np.float32)
    out_arr = (h_val - dem_tile).astype(np.float32)

    # Open full DEM to get georef and compute tile geotransform
    dem_path = os.path.join(base_path, 'DEM.tif')
    ds_dem = gdal.Open(dem_path, gdal.GA_ReadOnly)
    if ds_dem is None:
        print(f"[HGT-DEM] Cannot open DEM at {dem_path}")
        return False
    gt = ds_dem.GetGeoTransform()
    proj = ds_dem.GetProjection()

    tile_gt = (
        gt[0] + float(c_start) * gt[1] + float(r_start) * gt[2],
        gt[1],
        gt[2],
        gt[3] + float(c_start) * gt[4] + float(r_start) * gt[5],
        gt[4],
        gt[5]
    )

    tile_rows = int(r_end) - int(r_start)
    tile_cols = int(c_end) - int(c_start)

    out_name = f"HGT_minus_DEM_{int(r_start)}_{int(r_end)}_{int(c_start)}_{int(c_end)}.tif"
    out_path = os.path.join(base_path, out_name)

    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(
        out_path,
        tile_cols,
        tile_rows,
        1,
        gdal.GDT_Float32,
        options=[
            'TILED=YES',
            'BIGTIFF=YES',
            'BLOCKXSIZE=256',
            'BLOCKYSIZE=256',
            'NUM_THREADS=ALL_CPUS'
        ]
    )
    if ds_out is None:
        print("[HGT-DEM] Could not create output GeoTIFF.")
        return False
    ds_out.SetGeoTransform(tile_gt)
    ds_out.SetProjection(proj)
    rb = ds_out.GetRasterBand(1)
    rb.WriteArray(out_arr)
    rb.SetNoDataValue(-9999.0)
    try:
        ds_out.FlushCache()
    except Exception:
        pass
    ds_out = None
    print(f"[HGT-DEM] Saved: {out_path}")
    return True

def mosaic_hgt_dem_diffs(folder_path, log_fn=None, out_name="HGT_minus_DEM.tif", cleanup=True):
    """Mosaic HGT_minus_DEM_* tiles into one raster exactly matching DEM.tif shape/georef.

    Strategy:
    - Parse tile indices from filenames (HGT_minus_DEM_r0_r1_c0_c1.tif).
    - Pre-allocate a full-size array (rows, cols) from DEM.tif and fill slices directly.
    - Write GeoTIFF with DEM geotransform/projection. No resampling/warping, no rounding errors.
    - If anything goes wrong, fall back to the VRT+Warp approach used previously.
    """
    patt = os.path.join(folder_path, "HGT_minus_DEM_*.tif")
    files = sorted(glob.glob(patt))
    if not files:
        if log_fn:
            log_fn("No HGT_minus_DEM_*.tif tiles found to mosaic.\n")
        return None

    dem_path = os.path.join(folder_path, "DEM.tif")
    dem_ds = gdal.Open(dem_path, gdal.GA_ReadOnly)
    if dem_ds is None:
        if log_fn:
            log_fn("DEM.tif not found or unreadable; cannot align mosaic.\n")
        return None

    dem_gt = dem_ds.GetGeoTransform()
    dem_proj = dem_ds.GetProjection()
    dem_cols = dem_ds.RasterXSize
    dem_rows = dem_ds.RasterYSize

    # First try: direct placement by indices
    try:
        if log_fn:
            log_fn(f"🧩 Mosaicking {len(files)} HGT-DEM tiles by direct placement...\n")
        else:
            print(f"🧩 Mosaicking {len(files)} HGT-DEM tiles by direct placement...")

        full = np.full((dem_rows, dem_cols), -9999.0, dtype=np.float32)
        for f in files:
            # Expect pattern ..._r0_r1_c0_c1.tif
            base = os.path.basename(f)
            try:
                parts = os.path.splitext(base)[0].split('_')
                r0, r1, c0, c1 = map(int, parts[-4:])
            except Exception:
                raise RuntimeError(f"Unexpected tile name: {base}")
            ds_t = gdal.Open(f, gdal.GA_ReadOnly)
            arr = ds_t.GetRasterBand(1).ReadAsArray().astype(np.float32)
            tr, tc = arr.shape
            if tr != (r1 - r0) or tc != (c1 - c0):
                raise RuntimeError(f"Tile size mismatch for {base}: got {arr.shape}, expected {(r1-r0, c1-c0)}")
            full[r0:r1, c0:c1] = arr

        out_path = os.path.join(folder_path, out_name)
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            out_path, dem_cols, dem_rows, 1, gdal.GDT_Float32,
            options=['TILED=YES', 'BIGTIFF=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512', 'NUM_THREADS=ALL_CPUS']
        )
        out_ds.SetGeoTransform(dem_gt)
        out_ds.SetProjection(dem_proj)
        rb = out_ds.GetRasterBand(1)
        rb.SetNoDataValue(-9999.0)
        rb.WriteArray(full)
        try:
            out_ds.FlushCache()
        except Exception:
            pass
        out_ds = None

        if cleanup:
            for f in files:
                try:
                    os.remove(f)
                except Exception:
                    pass
        return out_path

    except Exception as e:
        # Fallback to VRT+Warp approach
        if log_fn:
            log_fn(f"Direct placement failed ({e}); falling back to VRT+Warp.\n")
        else:
            print(f"Direct placement failed ({e}); falling back to VRT+Warp.")

        dem_gt = dem_ds.GetGeoTransform()
        dem_proj = dem_ds.GetProjection()
        dem_cols = dem_ds.RasterXSize
        dem_rows = dem_ds.RasterYSize
        minX = dem_gt[0]
        maxY = dem_gt[3]
        maxX = dem_gt[0] + dem_cols * dem_gt[1] + dem_rows * dem_gt[2]
        minY = dem_gt[3] + dem_cols * dem_gt[4] + dem_rows * dem_gt[5]

        vrt_path = os.path.join(folder_path, "_hgt_dem_diff.vrt")
        vrt = gdal.BuildVRT(
            vrt_path, files,
            options=gdal.BuildVRTOptions(srcNodata=-9999.0, VRTNodata=-9999.0, resolution='highest')
        )
        if vrt is None:
            if log_fn:
                log_fn("Failed to build VRT for HGT-DEM mosaic.\n")
            return None

        out_path = os.path.join(folder_path, out_name)
        warp_opts = gdal.WarpOptions(
            dstSRS=dem_proj,
            xRes=abs(dem_gt[1]), yRes=abs(dem_gt[5]),
            outputBounds=(minX, minY, maxX, maxY),
            targetAlignedPixels=True,
            srcNodata=-9999.0, dstNodata=-9999.0,
            resampleAlg='near',
            creationOptions=[
                'TILED=YES',
                'BIGTIFF=YES',
                'BLOCKXSIZE=512',
                'BLOCKYSIZE=512',
                'NUM_THREADS=ALL_CPUS'
            ]
        )
        out_ds = gdal.Warp(out_path, vrt, options=warp_opts)
        try:
            vrt = None
        except Exception:
            pass
        if out_ds is None:
            if log_fn:
                log_fn("Failed to write HGT-DEM mosaic GeoTIFF.\n")
            return None
        try:
            out_ds.FlushCache()
        except Exception:
            pass
        out_ds = None

        if cleanup:
            try:
                for f in files:
                    try:
                        os.remove(f)
                    except Exception:
                        pass
                try:
                    if os.path.exists(vrt_path):
                        os.remove(vrt_path)
                except Exception:
                    pass
                if log_fn:
                    log_fn("🧹 Removed temporary  HGT_minus_DEM_* tiles and VRT.\n")
                else:
                    print("🧹 Removed temporary  HGT_minus_DEM_* tiles and VRT.")
            except Exception:
                pass
        return out_path

## moved to top of file



"""Preprocessing utilities: build metfiles from various sources and save tile rasters.
All comments and logs are in English.
"""

def process_era5_data(start_time: str, end_time: str, folder_path: str, output_file: str):
    """
    Slice ERA5 hourly data between start_time and end_time, compute derived variables,
    and write a NetCDF identical to the COSMO schema.

    Expected (invented) ERA5 input variable names:
      - ERA5_T2M (K), ERA5_RH2M (%) già in umidità relativa, ERA5_PMSL (Pa)
      - ERA5_U10M (m s-1), ERA5_V10M (m s-1)
      - ERA5_SW (W m-2), ERA5_LW (W m-2)
      - ERA5_HGT (m) as a 2D (lat, lon) field without time
    """

    # Parse time range
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time   = datetime.strptime(end_time,   "%Y-%m-%d %H:%M:%S")

    # Find an ERA5 NetCDF in the folder (single file expected; fallback to any .nc)
    files = [f for f in os.listdir(folder_path) if f.lower().startswith("era5") and f.endswith('.nc')]
    if not files:
        files = [f for f in os.listdir(folder_path) if f.endswith('.nc')]
    if not files:
        print("No NetCDF files found in the folder.")
        return
    file_path = os.path.join(folder_path, files[0])
    print(f"Found NetCDF file: {files[0]}")

    # Read source data and slice in time
    with Dataset(file_path) as ds:
        # Time variable
        time_var  = ds["time"]
        time_units = time_var.units
        calendar   = getattr(time_var, "calendar", "standard")
        time_vals  = time_var[:]
        time_all   = num2date(time_vals, units=time_units, calendar=calendar)

        
        # Indices in the requested time range
        time_mask = [(t >= start_time) and (t <= end_time) for t in time_all]
        idx_sel = np.where(time_mask)[0]
        if idx_sel.size == 0:
            print("No temporal data found in the selected range.")
            return
        sel_times = [time_all[i] for i in idx_sel]
        print(f"Found {idx_sel.size} timesteps in range {start_time} → {end_time}")

        # Coordinates; keep input as-is, make 2D for output only
        lat = ds.variables["lat"][:]
        lon = ds.variables["lon"][:]
        if np.ndim(lat) == 1 and np.ndim(lon) == 1:
            lon2d, lat2d = np.meshgrid(lon, lat)
        else:
            lat2d = np.array(lat)
            lon2d = np.array(lon)
            if lat2d.shape != lon2d.shape and lon2d.T.shape == lat2d.shape:
                lon2d = lon2d.T

        # Required variables for selected timesteps (ERA5 placeholder names)
        t2   = ds.variables["T2M"][idx_sel, :, :]
        rh2  = ds.variables["RH2M"][idx_sel, :, :]  # RH already provided (%)
        psfc = ds.variables["PSFC"][idx_sel, :, :]

        # Derived variables
        u10  = ds.variables["U10M"][idx_sel, :, :]
        v10  = ds.variables["V10M"][idx_sel, :, :]
        wind = np.sqrt(u10**2 + v10**2)

        swdown = ds.variables["SWDOWN"][idx_sel, :, :]
        glw    = ds.variables["LWDOWN"][idx_sel, :, :]

        uhi_cycle = ds.variables["UHI_CYCLE"][idx_sel, :, :] if "UHI_CYCLE" in ds.variables else np.zeros_like(t2, dtype=np.float32)
        if "UHI_CYCLE" not in ds.variables:
            print("[WARN] ERA5 source missing UHI_CYCLE; writing zeros")

        # HGT: 2D field (no time)
        hgt = ds.variables["HGT"][:]
        hgt = np.squeeze(hgt)

    # Write processed NetCDF (same structure as process_cosmo_data)
    with Dataset(output_file, "w", format="NETCDF4") as nc_out:
        # Dimensions
        nc_out.createDimension("time", len(sel_times))
        nc_out.createDimension("lat", lat2d.shape[0])
        nc_out.createDimension("lon", lat2d.shape[1])

        # Coordinate variables
        time_var = nc_out.createVariable("time", "f8", ("time",))
        lat_var  = nc_out.createVariable("lat",  "f4", ("lat", "lon"))
        lon_var  = nc_out.createVariable("lon",  "f4", ("lat", "lon"))

        # Data variables
        var_map = {
            "T2":     (t2,    "K"),
            "WIND":   (wind,  "m s-1"),
            "RH2":    (rh2,   "%"),
            "SWDOWN": (swdown,"W m-2"),
            "GLW":    (glw,   "W m-2"),
            "PSFC":   (psfc,  "Pa"),
            "UHI_CYCLE": (uhi_cycle, "K"),
        }
        for name, (data, unit) in var_map.items():
            v = nc_out.createVariable(name, "f4", ("time", "lat", "lon"), zlib=True)
            v.units = unit
            v[:, :, :] = data

        # HGT (2D)
        vhs = nc_out.createVariable("HGT", "f4", ("lat", "lon"), zlib=True)
        vhs.units = "m"
        vhs[:, :] = hgt

        # Coordinates content/metadata
        lat_var[:, :] = lat2d
        lon_var[:, :] = lon2d
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"

        # Encode time to a fixed epoch
        time_var.units = "hours since 1900-01-01"
        time_var.calendar = "gregorian"
        time_var[:] = date2num(sel_times, units=time_var.units, calendar=time_var.calendar)

    print(f"New NetCDF created: {output_file}")

def process_cosmo_data(start_time: str, end_time: str, folder_path: str, output_file: str):
    """
    Slice COSMO hourly data between start_time and end_time, compute derived variables,
    and write a new NetCDF containing:
      - time (selected range), lat, lon
      - T2 (K), WIND (m/s), RH2 (%), SWDOWN (W/m^2), GLW (W/m^2), PSFC (Pa)
      - HGT (m) as a 2D (lat, lon) field without time

    Notes:
      - HGT is read directly from the source file and written without any transpose/shape checks.
      - Input lat/lon are written as 2D variables (lat, lon) to match the source grid.
    """
    # Using module-level imports (os, numpy as np, Dataset, num2date, date2num, datetime)

    # Parse time range
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time   = datetime.strptime(end_time,   "%Y-%m-%d %H:%M:%S")

    # RH computed via shared helper rh_from_t_q_p

    # Find a COSMO NetCDF in the folder
    cosmo_files = [f for f in os.listdir(folder_path) if f.startswith("cosmo")]
    if not cosmo_files:
        print("No NetCDF files found in the folder.")
        return
    file_path = os.path.join(folder_path, cosmo_files[0])
    print(f"Found NetCDF file: {cosmo_files[0]}")

    # Read source data
    with Dataset(file_path) as ds:
        # Time and its conversion
        time_units = ds["time"].units
        calendar   = getattr(ds["time"], "calendar", "standard")
        time_vals  = ds["time"][:]
        time_all   = num2date(time_vals, units=time_units, calendar=calendar)

        # Select indices within the requested time range
        time_mask = [(t >= start_time) and (t <= end_time) for t in time_all]
        idx_sel = np.where(time_mask)[0]
        if idx_sel.size == 0:
            print("No temporal data found in the selected range.")
            return
        sel_times = [time_all[i] for i in idx_sel]
        print(f"Found {idx_sel.size} timesteps in range {start_time} → {end_time}")

        # Read coordinates; keep input as-is, make 2D for output only
        lat = ds.variables["lat"][:]
        lon = ds.variables["lon"][:]
        if np.ndim(lat) == 1 and np.ndim(lon) == 1:
            lon2d, lat2d = np.meshgrid(lon, lat)
        else:
            lat2d = np.array(lat)
            lon2d = np.array(lon)
            if lat2d.shape != lon2d.shape and lon2d.T.shape == lat2d.shape:
                lon2d = lon2d.T

        # Extract required variables for selected timesteps
        t2   = ds.variables["T2M"][idx_sel, :, :]
        rh2   = ds.variables["RH2M"][idx_sel, :, :]
        psfc = ds.variables["PSFC"][idx_sel, :, :]   # using sea-level pressure as in your code

        # Derived variables
        u10  = ds.variables["U10M"][idx_sel, :, :]
        v10  = ds.variables["V10M"][idx_sel, :, :]
        wind = np.sqrt(u10**2 + v10**2)

        swdown = ds.variables["SWDOWN"][idx_sel, :, :]
        glw    = ds.variables["LWDOWN"][idx_sel, :, :]

        # HGT: read directly (2D, no time); squeeze in case it's (1, y, x)
        hgt = ds.variables["HGT"][:]
        hgt = np.squeeze(hgt)

    # Write processed NetCDF
    with Dataset(output_file, "w", format="NETCDF4") as nc_out:
        # Dimensions
        nc_out.createDimension("time", len(sel_times))
        nc_out.createDimension("lat", lat2d.shape[0])
        nc_out.createDimension("lon", lat2d.shape[1])

        # Coordinate variables
        time_var = nc_out.createVariable("time", "f8", ("time",))
        lat_var  = nc_out.createVariable("lat",  "f4", ("lat", "lon"))
        lon_var  = nc_out.createVariable("lon",  "f4", ("lat", "lon"))

        # Data variables (3D with time) — same naming you used
        var_map = {
            "T2":     (t2,    "K"),
            "WIND":   (wind,  "m s-1"),
            "RH2":    (rh2,   "%"),
            "SWDOWN": (swdown,"W m-2"),
            "GLW":    (glw,   "W m-2"),
            "PSFC":   (psfc,  "Pa"),
        }
        for name, (data, unit) in var_map.items():
            v = nc_out.createVariable(name, "f4", ("time", "lat", "lon"), zlib=True)
            v.units = unit
            v[:, :, :] = data

        # HGT (2D, no time)
        vhs = nc_out.createVariable("HGT", "f4", ("lat", "lon"), zlib=True)
        vhs.units = "m"
        vhs[:, :] = hgt

        # Set coordinates content/metadata (always 2D)
        lat_var[:, :] = lat2d
        lon_var[:, :] = lon2d
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"

        # Encode time to a fixed epoch
        time_var.units = "hours since 1900-01-01"
        time_var.calendar = "gregorian"
        time_var[:] = date2num(sel_times, units=time_var.units, calendar=time_var.calendar)

def process_metfiles(rstart: int, rend: int, cstart: int, cend: int,
                     netcdf_file: str, center_lon: float, center_lat: float,
                     base_path: str, start_time: str | None = None, end_time: str | None = None):
    """Create one metfile text for the tile window by sampling the nearest gridpoint in a NetCDF."""

    tf = TimezoneFinder()
    dataset = Dataset(netcdf_file, "r")

    var_map = {
        "Wind": "WIND",
        "RH": "RH2",
        "Td": "T2",
        "press": "PSFC",
        "Kdn": "SWDOWN",
        "Kdiff": "KDIFF",
        "Kdir": "KDIR",
        "UHI_CYCLE": "UHI_CYCLE",
    }

    # Determine folder tag: prefer run's start_time for consistency with thermal_comfort()
    def _parse_dt(dt_str: str | None) -> Optional[datetime]:
        if not dt_str:
            return None
        try:
            return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    run_dt = _parse_dt(start_time)
    run_end_dt = _parse_dt(end_time)

    time_var = dataset.variables["time"][:]
    time_units = dataset.variables["time"].units
    time_base_date = num2date(time_var, units=time_units, only_use_cftime_datetimes=False)
    start_date = time_base_date[0].date()
    end_date = time_base_date[-1].date()
    start_date_str = start_date.strftime("%Y-%m-%d")

    first_dt = time_base_date[0]
    # Use run's start date if provided; fallback to first date in NetCDF
    tag_dt = run_dt or first_dt
    folder_tag = f"{tag_dt.year:04d}_{tag_dt.month:02d}_{tag_dt.day:02d}"
    metfiles_folder = os.path.join(base_path, f"metfiles_{folder_tag}")
    os.makedirs(metfiles_folder, exist_ok=True)
 
    # --- Columns to export ---
    columns = [
        "iy", "id", "it", "imin",
        "Wind", "RH", "Td", "press",
        "Kdn", "Kdiff", "Kdir", "UHI_CYCLE"
    ]

    # --- Cache variables needed downstream ---
    cached_vars = {}
    missing_nc_vars = []
    for nc_var in set(var_map.values()):
        if nc_var in dataset.variables:
            cached_vars[nc_var] = dataset.variables[nc_var][:]
        else:
            cached_vars[nc_var] = None
            missing_nc_vars.append(nc_var)
    if missing_nc_vars:
        print(
            "[WARN] Missing variables in NetCDF: " + ", ".join(sorted(missing_nc_vars)) +
            ". Using fallback values where possible."
        )

    # --- Read latitude/longitude (support 1D or 2D) ---
    lat_arr = np.asarray(dataset.variables["lat"][:])
    lon_arr = np.asarray(dataset.variables["lon"][:])

    # --- Select nearest NC cell for the provided tile center ---
    print(f"Tile {rstart}_{rend}_{cstart}_{cend} center lat={center_lat:.6f} lon={center_lon:.6f}")

    if lat_arr.ndim == 2 and lon_arr.ndim == 2:
    # Curvilinear grid: search in 2D
        dist2 = (lat_arr - center_lat)**2 + (lon_arr - center_lon)**2
        flat_idx = np.argmin(dist2)
        tile_lat_idx, tile_lon_idx = np.unravel_index(flat_idx, dist2.shape)
        nc_lat = float(lat_arr[tile_lat_idx, tile_lon_idx])
        nc_lon = float(lon_arr[tile_lat_idx, tile_lon_idx])

    elif lat_arr.ndim == 1 and lon_arr.ndim == 1:
    # Rectilinear grid: independent 1D axes
        tile_lat_idx = int(np.argmin(np.abs(lat_arr - center_lat)))
        tile_lon_idx = int(np.argmin(np.abs(lon_arr - center_lon)))
        nc_lat = float(lat_arr[tile_lat_idx])
        nc_lon = float(lon_arr[tile_lon_idx])

    else:
    # Mixed shape (e.g., variables defined as lat(lat,lon) but effectively rectilinear)
        lat_axis = lat_arr[:, 0] if lat_arr.ndim == 2 else lat_arr
        lon_axis = lon_arr[0, :] if lon_arr.ndim == 2 else lon_arr
        tile_lat_idx = int(np.argmin(np.abs(lat_axis - center_lat)))
        tile_lon_idx = int(np.argmin(np.abs(lon_axis - center_lon)))
    
    # Use full 2D fields when available to report exact cell center
    if lat_arr.ndim == 2 and lon_arr.ndim == 2:
        nc_lat = float(lat_arr[tile_lat_idx, tile_lon_idx])
        nc_lon = float(lon_arr[tile_lat_idx, tile_lon_idx])
    else:
        nc_lat = float(lat_axis[tile_lat_idx])
        nc_lon = float(lon_axis[tile_lon_idx])

    print(f"Nearest NC lat={nc_lat:.6f} lon={nc_lon:.6f}")
    
    # Accumulate selected indices for final subsetting
    sidecar_path = os.path.join(base_path, ".selected_indices.json")
    
    # Read existing selections (backward compatible)
    sel = {"lat": [], "lon": []}
    if os.path.exists(sidecar_path):
        try:
            with open(sidecar_path, "r") as f:
                sel = json.load(f)
        except Exception:
            sel = {"lat": [], "lon": []}

    # Ensure keys exist
    sel.setdefault("lat", [])
    sel.setdefault("lon", [])
    sel.setdefault("cells", [])  # for 2D grids

    lat_arr = np.asarray(dataset.variables["lat"][:])
    lon_arr = np.asarray(dataset.variables["lon"][:])

    if lat_arr.ndim == 1 and lon_arr.ndim == 1:
    # Rectilinear grid → store separate indices
        sel["lat"].append(int(tile_lat_idx))
        sel["lon"].append(int(tile_lon_idx))
        sel["lat"] = sorted(set(int(i) for i in sel["lat"]))
        sel["lon"] = sorted(set(int(j) for j in sel["lon"]))

        with open(sidecar_path, "w") as f:
            json.dump(sel, f, separators=(",", ":"))

        nc_lat = float(lat_arr[tile_lat_idx])
        nc_lon = float(lon_arr[tile_lon_idx])
        print(
            f"Selected idx lat_idx={tile_lat_idx}, lon_idx={tile_lon_idx} "
            f"(lat={nc_lat:.6f}, lon={nc_lon:.6f}) "
            f"(uniq lat={len(sel['lat'])}, uniq lon={len(sel['lon'])})"
        )

    else:
    # Curvilinear grid → store (iy, ix) pairs
        sel["cells"].append([int(tile_lat_idx), int(tile_lon_idx)])
        cells_unique = sorted({(i, j) for i, j in sel["cells"]})
        sel["cells"] = [[i, j] for (i, j) in cells_unique]

        with open(sidecar_path, "w") as f:
            json.dump(sel, f, separators=(",", ":"))

        nc_lat = float(lat_arr[tile_lat_idx, tile_lon_idx])
        nc_lon = float(lon_arr[tile_lat_idx, tile_lon_idx])
        print(
            f"Selected cell iy={tile_lat_idx}, ix={tile_lon_idx} "
            f"(lat={nc_lat:.6f}, lon={nc_lon:.6f}) "
            f"(uniq cells={len(sel['cells'])})"
        )
    
    
    output_text_file = os.path.join(metfiles_folder, f"metfile_{rstart}_{rend}_{cstart}_{cend}.txt")

    timezone_name = tf.timezone_at(lng=center_lon, lat=center_lat) or "UTC"
    local_tz = pytz.timezone(timezone_name)
    def _localize(dt: datetime | None, default: datetime) -> datetime:
        target = dt or default
        if target.tzinfo is None:
            try:
                return local_tz.localize(target)
            except Exception:
                return local_tz.localize(target, is_dst=None)
        return target.astimezone(local_tz)

    default_start_local = local_tz.localize(datetime.combine(start_date, time(0, 0)))
    default_end_local = local_tz.localize(datetime.combine(end_date, time(23, 59)))
    requested_start_local = _localize(run_dt, default_start_local)
    requested_end_local = _localize(run_end_dt, default_end_local)
    if requested_end_local < requested_start_local:
        requested_end_local = requested_start_local

    utc_start = requested_start_local.astimezone(pytz.utc)
    utc_end = requested_end_local.astimezone(pytz.utc)

    time_indices = [
        i for i, dt in enumerate(time_base_date)
        if utc_start <= dt.replace(tzinfo=pytz.utc) <= utc_end
    ]
    if not time_indices:
        print(f"No UTC data for {start_date_str} for tile {rstart}_{rend}_{cstart}_{cend}")
        dataset.close()
        return

    print(f"Metfile {rstart}_{rend}_{cstart}_{cend}: {len(time_indices)} steps TZ={timezone_name}")

        
    data = []
    for t in time_indices:
        utc_time = time_base_date[t].replace(tzinfo=pytz.utc)
        local_time = utc_time.astimezone(local_tz)
        year = local_time.year
        doy = local_time.timetuple().tm_yday
        hour = local_time.hour
        minute = local_time.minute
        month  = local_time.month
        day   = local_time.day

        row = [year, doy, hour, minute]
        extracted_values = {}

        for key, ncvar in var_map.items():
            arr = cached_vars.get(ncvar)
            if arr is None:
                if key == "UHI_CYCLE":
                    val = 0.0
                else:
                    val = -999.0
            else:
                try:
                    slice_arr = arr[t]
                    val = slice_arr[tile_lat_idx, tile_lon_idx]
                except Exception:
                    val = float(arr[t]) if arr.ndim == 1 else -999.0
            if key == "Td" and val != -999:
                val -= 273.15
            if key == "press" and val != -999:
                val /= 1000.0
            extracted_values[key] = float(val)

        row.append(extracted_values)
        data.append(row)

    # Format for DataFrame
    formatted_data = []
    for row in data:
        fixed = row[:4]
        values = row[4]
        full_row = {col: -999 for col in columns}
        full_row["UHI_CYCLE"] = 0.0
        full_row.update(dict(zip(["iy", "id", "it", "imin"], fixed)))
        for k, v in values.items():
            full_row[k] = v
        formatted_data.append(full_row)

    df = pd.DataFrame(formatted_data, columns=columns)
    with open(output_text_file, "w") as f:
        f.write(" ".join(df.columns) + "\n")
        fmt = ['%d', '%d', '%d', '%d'] + ['%.5f'] * (len(columns) - 4)
        np.savetxt(f, df.values, fmt=fmt)
    print(f"Saved {output_text_file}")

    dataset.close()
    print(f"Metfile done {rstart}_{rend}_{cstart}_{cend}")



def ppr(r_start: int, r_end: int, c_start: int, c_end: int,
        input_data_path: str, DEM,
        data_source_type: str,
        start_time: str, end_time: str,
        projection: str, center_x: float, center_y: float):
    """Per-tile preprocessing runner: generates metfiles from gridded sources, and HGT−DEM diffs."""

    # Use the provided tile center coordinates (from tile_centers_from_full)
    x_c = float(center_x)
    y_c = float(center_y)

    # Reproject center to EPSG:4326 using OSR (robust on clusters without PROJ db)
    srs_src = osr.SpatialReference(); srs_src.ImportFromWkt(projection)
    srs_dst = osr.SpatialReference(); srs_dst.ImportFromEPSG(4326)
    ct = osr.CoordinateTransformation(srs_src, srs_dst)
    lonlat = ct.TransformPoint(x_c, y_c)
    try:
        # GDAL 3 axis order handling
        if str(gdal.__version__).startswith('3'):
            center_lon, center_lat = lonlat[1], lonlat[0]
        else:
            center_lon, center_lat = lonlat[0], lonlat[1]
    except Exception:
        center_lon, center_lat = lonlat[0], lonlat[1]
    
    
    # Name processed NetCDF with the run start date (e.g., Outfile_2013_05_01.nc)
    try:
        start_dt_for_nc = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        date_tag_nc = f"{start_dt_for_nc.year:04d}_{start_dt_for_nc.month:02d}_{start_dt_for_nc.day:02d}"
    except Exception:
        # Fallback to the first time in the source NetCDF later if parsing fails
        date_tag_nc = None
    processed_nc_file = os.path.join(input_data_path, f"Outfile_{date_tag_nc}.nc" if date_tag_nc else "Outfile.nc")

    # Generate metfiles from processed NetCDF sources.
    if not os.path.exists(processed_nc_file):
        if data_source_type and data_source_type.lower() == "era5":
            process_era5_data(start_time, end_time, input_data_path, processed_nc_file)
        elif data_source_type and data_source_type.lower() == "cosmo":
            process_cosmo_data(start_time, end_time, input_data_path, processed_nc_file)

    process_metfiles(
        r_start,
        r_end,
        c_start,
        c_end,
        processed_nc_file,
        center_lon,
        center_lat,
        input_data_path,
        start_time=start_time,
        end_time=end_time,
    )

def finalize_outfile_subset(base_path, outfile_name="Outfile.nc", output_name="Outfile_selected.nc"):
    """Create a cropped NetCDF from Outfile.nc using indices stored in .selected_indices.json.

    Supports sidecar with:
      - {"lat":[...], "lon":[...]}  (rectilinear or curvilinear; produces cartesian product)
      - {"cells":[[iy, ix], ...]}   (curvilinear-friendly; expanded to unique rows/cols, cartesian)
    Preserves source grid structure:
      - If source lat/lon are 1D → write 1D lat/lon.
      - If source lat/lon are 2D → write 2D lat/lon.
    """
    import json
    from netCDF4 import Dataset
    import numpy as np
    import os

    sidecar_path = os.path.join(base_path, ".selected_indices.json")
    src_path = os.path.join(base_path, outfile_name)
    dst_path = os.path.join(base_path, output_name)

    if not os.path.exists(src_path):
        print(f"Source NetCDF missing: {src_path}")
        return False
    if not os.path.exists(sidecar_path):
        print("No selection; skip subset")
        return False
    if os.path.exists(dst_path):
        print(f"Subset NetCDF already exists: {dst_path}, skipping creation.")
        return True

    # --- Load selection and normalize to unique row/col index sets
    with open(sidecar_path, "r") as f:
        sel = json.load(f)

    lat_idx_set = set(int(i) for i in sel.get("lat", []))
    lon_idx_set = set(int(j) for j in sel.get("lon", []))

    cells = sel.get("cells", [])
    if isinstance(cells, list) and cells:
        # Expand pairs to unique row/col sets (cartesian product subset)
        for item in cells:
            try:
                iy, ix = int(item[0]), int(item[1])
                lat_idx_set.add(iy)
                lon_idx_set.add(ix)
            except Exception:
                pass

    sel_lat_idx = np.array(sorted(lat_idx_set), dtype=int)
    sel_lon_idx = np.array(sorted(lon_idx_set), dtype=int)

    if sel_lat_idx.size == 0 or sel_lon_idx.size == 0:
        print("Empty selection; skip subset")
        return False

    # --- Open source and figure out coordinate dimension names and shapes
    with Dataset(src_path, "r") as src:
        # Detect coordinate var names
        lat_name = "lat" if "lat" in src.variables else ("latitude" if "latitude" in src.variables else None)
        lon_name = "lon" if "lon" in src.variables else ("longitude" if "longitude" in src.variables else None)
        if lat_name is None or lon_name is None:
            print("Missing lat/lon variables in source.")
            return False

        time_var = src.variables["time"]
        time = time_var[:]

        lat_var = src.variables[lat_name][:]
        lon_var = src.variables[lon_name][:]

        # Determine rectilinear (1D) vs curvilinear (2D)
        lat_is_2d = (lat_var.ndim == 2)
        lon_is_2d = (lon_var.ndim == 2)

        # We require consistent structure
        if lat_is_2d != lon_is_2d:
            print("Inconsistent lat/lon dimensionality (one 1D, the other 2D).")
            return False

        if lat_is_2d:
            ny, nx = lat_var.shape
        else:
            ny = lat_var.shape[0]
            nx = lon_var.shape[0]

        # Bounds check
        sel_lat_idx = sel_lat_idx[(sel_lat_idx >= 0) & (sel_lat_idx < ny)]
        sel_lon_idx = sel_lon_idx[(sel_lon_idx >= 0) & (sel_lon_idx < nx)]
        if sel_lat_idx.size == 0 or sel_lon_idx.size == 0:
            print("No valid indices after bounds check")
            return False

        # --- Create destination file
        with Dataset(dst_path, "w", format="NETCDF4") as dst:
            # Create dimensions
            dst.createDimension("time", len(time))
            dst.createDimension(lat_name, len(sel_lat_idx))
            dst.createDimension(lon_name, len(sel_lon_idx))

            # time
            t_dst = dst.createVariable("time", time_var.dtype.str if hasattr(time_var, "dtype") else "f8", ("time",))
            t_dst[:] = time
            if hasattr(time_var, "units"):    t_dst.units = time_var.units
            if hasattr(time_var, "calendar"): t_dst.calendar = time_var.calendar

            # lat/lon variables (preserve structure; use float64 for precision)
            if lat_is_2d:
                lat_dst = dst.createVariable(lat_name, "f8", (lat_name, lon_name), zlib=True)
                lon_dst = dst.createVariable(lon_name, "f8", (lat_name, lon_name), zlib=True)
                lat_dst[:, :] = lat_var[sel_lat_idx[:, None], sel_lon_idx]
                lon_dst[:, :] = lon_var[sel_lat_idx[:, None], sel_lon_idx]
            else:
                lat_dst = dst.createVariable(lat_name, "f8", (lat_name,), zlib=True)
                lon_dst = dst.createVariable(lon_name, "f8", (lon_name,), zlib=True)
                lat_dst[:] = lat_var[sel_lat_idx]
                lon_dst[:] = lon_var[sel_lon_idx]

            lat_dst.units = getattr(src.variables[lat_name], "units", "degrees_north")
            lon_dst.units = getattr(src.variables[lon_name], "units", "degrees_east")

            # --- Copy data variables (time, lat, lon already handled)
            # Accept both ('time','lat','lon') and ('time','latitude','longitude')
            valid_dim_sets = {
                ("time", lat_name, lon_name),
            }

            for vname, var in src.variables.items():
                if vname in ("time", lat_name, lon_name):
                    continue

                # Only subset variables that match the 3D space-time grid
                if tuple(var.dimensions) in valid_dim_sets and var.ndim == 3:
                    vdst = dst.createVariable(vname, "f4", ("time", lat_name, lon_name), zlib=True)
                    # copy units if any
                    if hasattr(var, "units"):
                        vdst.units = var.units

                    # subset
                    data_sub = var[:]                                   # (time, y, x)
                    data_sub = np.take(data_sub, sel_lat_idx, axis=1)   # lat
                    data_sub = np.take(data_sub, sel_lon_idx, axis=2)   # lon
                    vdst[:, :, :] = np.ascontiguousarray(data_sub, dtype=np.float32)

            # HGT or similar static 2D fields (("lat","lon") or ("latitude","longitude"))
            for static_name in "HGT":
                if static_name in src.variables:
                    svar = src.variables[static_name]
                    if tuple(svar.dimensions) == (lat_name, lon_name):
                        sdst = dst.createVariable(static_name, "f4", (lat_name, lon_name), zlib=True)
                        if hasattr(svar, "units"):
                            sdst.units = svar.units
                        ssub = svar[:]                                  # (y, x)
                        ssub = np.take(ssub, sel_lat_idx, axis=0)       # lat
                        ssub = np.take(ssub, sel_lon_idx, axis=1)       # lon
                        sdst[:, :] = np.ascontiguousarray(ssub, dtype=np.float32)

            # Copy global attributes
            try:
                for attr in src.ncattrs():
                    if attr.lower() != "_fillvalue":
                        dst.setncattr(attr, src.getncattr(attr))
            except Exception:
                pass

    print(
        f"Created subset NetCDF: {dst_path} "
        f"(lat_idx={sel_lat_idx.size}, lon_idx={sel_lon_idx.size}, time={len(time)})"
    )
    return True
## module docstring moved to top

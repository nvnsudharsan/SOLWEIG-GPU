
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("Developed by Andrea Zonato, andrea.zonato@cimafoundation.org")

"""Build input datasets for SOLWEIG.

Author: Andrea Zonato

What this script does
- Given a target location (lat, lon), it computes geographic bounding boxes
  and a UTM working CRS, then prepares a common analysis grid for SOLWEIG.
- It collects static inputs: landcover (ESA WorldCover), canopy height/trees
  (multiple Earth Engine sources), and a DEM (TINITALY if in Italy, otherwise
  MERIT/DEM via Earth Engine). It also derives urban layers from OSM (vegetation,
  impervious, water) and buildings from a public GBA WFS (LOD1), with an OSM fallback.
- It rasterizes vector layers (vegetation, water, buildings, urban surfaces),
  resamples everything to the common grid, applies stacking logic (water >
  vegetation > urban), and builds a Building DSM = DEM + building heights.
- It downloads and prepares meteorological forcing. If the area is in Italy,
  it uses the DDS downscaled dataset; elsewhere it samples ERA5 via Earth Engine.
  In both cases, it standardizes names/dimensions and derives relative
  humidity and surface altitude to save a single, consistent NetCDF file.

"""

import argparse, math, shutil, subprocess, sys, time, io
from pathlib import Path

import importlib, importlib.util
from dataclasses import dataclass
from typing import Any, List

# Bootstrap: ensure third-party packages are installed before importing
def _ensure(mod, pip_name=None, required=True):
    """Ensure module is installable; avoid importing until actually needed."""
    if importlib.util.find_spec(mod) is None:
        pkg = pip_name or mod
        try:
            subprocess.check_call([sys.executable, "-m", "conda", "install", pkg])
        except Exception as e:
            if required:
                raise
            else:
                print(f"[deps] Optional package '{pkg}' not installed: {e}")

for _mod, _pip in [
    ("ee", "earthengine-api"), ("geemap", None), ("geopandas", None), ("numpy", None),
    ("pandas", None), ("rasterio", None), ("requests", None), ("osmnx", None),
    ("scipy", None), ("geopy", None), ("pyproj", None), ("shapely", None), ("xarray", None),
    ("cftime", None), ("netCDF4", None), ("ddsapi", None)
]:
    _ensure(_mod, _pip)

import ee
import geemap
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import osmnx as ox
from geopy.geocoders import Nominatim
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from shapely.geometry import box
import xarray as xr
from scipy.ndimage import uniform_filter, distance_transform_edt, maximum_filter, gaussian_filter

# Force GeoPandas to use Fiona engine for IO to avoid pyogrio/PROJ data issues
try:
    gpd.options.io_engine = "fiona"
except Exception:
    pass

# ------------------------------ Types & Models ------------------------------ #

@dataclass
class BBoxes:
    """Holds bounding boxes in different CRSs for processing."""
    bbox4326: List[float]              # [min_lon, min_lat, max_lon, max_lat]
    bbox_osm: List[float]              # [min_lat, min_lon, max_lat, max_lon]
    bbox_utm: Any                      # shapely Polygon in UTM CRS
    bbox_utm_solweig: Any              # shrunk polygon for SOLWEIG domain
    crs_utm: str                       # EPSG code for UTM zone

@dataclass
class GridSpec:
    """Raster grid specification used to resample/align rasters."""
    transform: Any
    width: int
    height: int

@dataclass
class Paths:
    """Convenience container for commonly used paths for a city."""
    data_dir: Path
    out_dir: Path
    esa_tif: Path
    tree_tif: Path
    tree_tiles: Path
    dem_tif: Path
    veg_fp: Path
    wat_fp: Path
    bld_fp: Path
    imprv_fp: Path
    veg_ras: Path
    wat_ras: Path
    bld_ras: Path
    landuse_ras: Path
    tree_ras: Path
    dem_ras: Path
    dsm_plus_ras: Path
    lcz_tif: Path
    lcz_ras: Path
    wind_coeff_ras: Path

def build_paths(base: Path, city: str) -> Paths:
    """Construct standard input/output file paths for the selected city.

    Notes on tree tiles:
    - Temporary canopy tiles are stored under `<city>/tree_dsm_tiles` (in the
      per-city data folder) and then mosaicked into a single DSM written to
      `<city>/tree_dsm_1m_merged.tif`. The tile folder may be removed later in
      the pipeline to keep the workspace tidy.
    """
    data_dir = base / city
    out_dir = base / f"{city}_for_solweig"
    return Paths(
        data_dir=data_dir,
        out_dir=out_dir,
        esa_tif=data_dir / "ESA_WorldCover_2020.tif",
        tree_tif=data_dir / "tree_dsm_1m_merged.tif",
        tree_tiles=data_dir / "tree_dsm_tiles",
        dem_tif=data_dir / "downloaded_dem.tif",
        lcz_tif=data_dir / "LCZ_raw.tif",
        veg_fp=data_dir / "vegetation.geojson",
        wat_fp=data_dir / "water.geojson",
        bld_fp=data_dir / "building.geojson",
        imprv_fp=data_dir / "impervious.geojson",
        veg_ras=out_dir / "Vegetation.tif",
        wat_ras=out_dir / "Water.tif",
        bld_ras=out_dir / "Buildings.tif",
        landuse_ras=out_dir / "Landuse.tif",
        tree_ras=out_dir / "Trees.tif",
        dem_ras=out_dir / "DEM.tif",
        dsm_plus_ras=out_dir / "Building_DSM.tif",
        lcz_ras=out_dir / "LCZ.tif",
        wind_coeff_ras=out_dir / "WindCoeff.tif",
    )

# ------------------------------- Config ------------------------------------- #

_DEFAULT_BASE_CANDIDATE = Path("/Users/andrea/Desktop/prova_tutto_insieme")
# Default workspace root; fall back to this script's folder if the preferred base is absent.
if _DEFAULT_BASE_CANDIDATE.exists():
    DEFAULT_BASE = str(_DEFAULT_BASE_CANDIDATE)
else:
    DEFAULT_BASE = str(Path(__file__).resolve().parent)
G0 = 9.80665  # standard gravity (m s-2)
# Air density × specific heat capacity (rho * Cp) to convert fluxes to kinematic units
RHO_CP = 1.225 * 1005.0

# Variable standardization (common to DDS and ERA5)
VAR_MAP = {
    "t2m": "T2M", "air_temperature": "T2M",
    "T_2M": "T2M",
    "relative_humidity": "RH2M",
    "ssrd": "SWDOWN", "surface_net_downward_shortwave_flux": "SWDOWN",
    "ASOB_S": "SWDOWN",
    "strd": "LWDOWN", "surface_net_downward_longwave_flux": "LWDOWN",
    "ATHB_S": "LWDOWN",
    "sp": "PSFC", "air_pressure_at_sea_level": "PSFC",
    "PMSL": "PSFC",
    "u10": "U10M", "grid_eastward_wind": "U10M",
    "U_10M": "U10M",
    "v10": "V10M", "grid_northward_wind": "V10M",
    "V_10M": "V10M",
    "surface_altitude": "HGT", "HSURF": "HGT",
    # Roughness length from ERA5: forecast_surface_roughness (and common aliases)
    "z0": "Z0", "forecast_surface_roughness": "Z0", "surface_roughness": "Z0", "roughness_length": "Z0",
}
DIMS_MAP = {"latitude": "lat", "longitude": "lon"}
WANTED_VARS = ["T2M", "RH2M", "SWDOWN", "HGT", "LWDOWN", "PSFC", "U10M", "V10M", "Z0", "UHI_CYCLE", "T2M_UHI"]



def _night_sine_profile(sw_values, amp_values, temp_values,
                        extension_before=4,
                        extension_after=2):
    """
    Return a simple sine-shaped UHI profile:
    - starts 3h before sunset, ends 3h after sunrise
    - amplitude = nightly UHI maximum
    - zero in daytime
    """
    import numpy as np

    sw_arr   = np.asarray(sw_values, dtype=float)
    amp_arr  = np.asarray(amp_values, dtype=float)
    _ = temp_values

    n = sw_arr.size
    out = np.zeros_like(sw_arr, dtype=float)
    if n == 0 or amp_arr.shape != sw_arr.shape:
        return out

    valid = np.isfinite(sw_arr) & np.isfinite(amp_arr)
    base_night = valid & (sw_arr <= 0)
    if not np.any(base_night):
        return out

    # Find contiguous night segments
    idx = np.flatnonzero(base_night)
    splits = np.where(np.diff(idx) > 1)[0] + 1
    raw_segments = [seg for seg in np.split(idx, splits) if seg.size]

    # Extend each segment
    expanded = []
    for seg in raw_segments:
        s = int(max(0, seg[0] - extension_before))
        e = int(min(n, seg[-1] + 1 + extension_after))
        expanded.append(np.arange(s, e, dtype=int))

    for seg in expanded:
        L = seg.size
        if L == 0:
            continue
        seg_amp = amp_arr[seg]
        seg_amp = seg_amp[np.isfinite(seg_amp)]
        if seg_amp.size == 0:
            continue
        A = float(np.mean(seg_amp))
        if A <= 0:
            continue

        # sine wave from 0 to π across the night
        x = np.linspace(0, np.pi, L)
        profile = np.sin(x)
        profile = np.maximum(profile, 0.0)
        out[seg] = A * profile

    return out
def add_uhi_indicator(ds):
    """Add UHI-based metrics derived from meteo variables to the dataset."""

    needed = {"SWDOWN", "T2M", "U10M"}
    if not needed <= set(ds.data_vars):
        return ds

    sw = ds["SWDOWN"].astype(float)
    t2m = ds["T2M"].astype(float)

    # Convert SWDOWN to kinematic units and aggregate daily diagnostics
    sdown_daily = (sw / RHO_CP).resample(time="1D").mean()
    tmax_daily = t2m.resample(time="1D").max()
    tmin_daily = t2m.resample(time="1D").min()
    dtr_daily = tmax_daily - tmin_daily
    numerator = (sdown_daily * dtr_daily**3).where(np.isfinite(sdown_daily * dtr_daily**3), 0).clip(min=0)
    uhi_max_daily = numerator ** 0.25
    uhi_max_daily = uhi_max_daily.fillna(0)

    uhi_max_ts = uhi_max_daily.reindex(time=ds.time, method="ffill").fillna(0)
    uhi_cycle = xr.apply_ufunc(
        _night_sine_profile,
        sw,
        uhi_max_ts,
        t2m,
        input_core_dims=[["time"], ["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="allowed",
        kwargs={
            "extension_before": 4,
            "extension_after": 2,
        }
    )
    uhi_cycle = uhi_cycle.fillna(0)

    uhi_cycle = uhi_cycle.transpose("time", "lat", "lon", ...) if "lat" in uhi_cycle.dims else uhi_cycle

    uhi_cycle = uhi_cycle.rename("UHI_CYCLE")

    uhi_max_ts.attrs.update({"units": "K", "long_name": "Daily maximum urban heat island intensity"})
    uhi_cycle.attrs.update({
        "units": "K",
        "long_name": "Nighttime urban heat island intensity (asymmetric SWDOWN-based)",
        "standard_name": "urban_heat_island_intensity",
    })

    ds["UHI_CYCLE"] = uhi_cycle
    # Also export an auxiliary absolute temperature that includes the UHI increment
    if "T2M" in ds:
        t2m_uhi = (ds["T2M"] + ds["UHI_CYCLE"])
        # Keep units in Kelvin to be consistent with ERA5 base temperature
        t2m_uhi.attrs.update({
            "units": "K",
            "long_name": "2 m air temperature plus UHI (absolute)",
            "standard_name": "air_temperature_plus_uhi"
        })
        ds["T2M_UHI"] = t2m_uhi
    return ds

# ------------------------------- OSM Tags ----------------------------------- #

OSM_TAGS_GREEN = {
    "leisure": ["park", "garden", "golf_course", "common", "nature_reserve"],
    "landuse": ["grass", "forest", "recreation_ground", "village_green", "meadow"],
    "natural": ["wood", "scrub", "heath", "grassland"],
    "boundary": ["national_park"],
}

OSM_TAGS_WATER = {
    "natural": ["water", "sea", "bay", "strait", "fjord"],
    "water": True,
    "waterway": ["riverbank", "reservoir", "canal", "dock", "basin"],
    "landuse": ["reservoir", "basin", "salt_pond", "fish_pond"],
    "place": ["sea", "ocean", "bay"],
}

OSM_TAGS_IMPERVIOUS_ASPHALT = {"surface": "asphalt"}
OSM_TAGS_IMPERVIOUS_PARKING = {"amenity": "parking"}

# ==========================
# Default city presets
# ==========================
# --- INNSBRUCK (reference) ---

# --- PESCARA ---
# DEFAULT_LAT = 42.4643
# DEFAULT_LON = 14.1942
# DEFAULT_KM_BUFFER = 6
# DEFAULT_KM_REDUCED_LAT = 2
# DEFAULT_KM_REDUCED_LON = 2
# DEFAULT_YEAR_START = 2003
# DEFAULT_YEAR_END = 2023
# DEFAULT_RES_M = 3 
# --- TORINO ---
# DEFAULT_LAT = 45.0803
# DEFAULT_LON = 7.6669
# DEFAULT_KM_BUFFER = 6
# DEFAULT_KM_REDUCED_LAT = 2
# DEFAULT_KM_REDUCED_LON = 3
# DEFAULT_YEAR_START = 2003
# DEFAULT_YEAR_END = 2023
# DEFAULT_RES_M = 3
# # --- NAPOLI ---
# DEFAULT_LAT = 40.8522
# DEFAULT_LON = 14.2681
# DEFAULT_KM_BUFFER = 6
# DEFAULT_KM_REDUCED_LAT = 3
# DEFAULT_KM_REDUCED_LON = 2
# DEFAULT_YEAR_START = 2003
# DEFAULT_YEAR_END = 2023
# DEFAULT_RES_M = 3
# --- LECCE ---
# DEFAULT_LAT = 40.3548
# DEFAULT_LON = 18.1724
# DEFAULT_KM_BUFFER = 3
# DEFAULT_KM_REDUCED_LAT = 1
# DEFAULT_KM_REDUCED_LON = 1
# DEFAULT_YEAR_START = 2021
# DEFAULT_YEAR_END = 2023
# DEFAULT_RES_M = 3
# --- PALERMO ---
# DEFAULT_LAT = 38.1157
# DEFAULT_LON = 13.3615
# DEFAULT_KM_BUFFER = 6
# DEFAULT_KM_REDUCED_LAT = 2
# DEFAULT_KM_REDUCED_LON = 2
# DEFAULT_YEAR_START = 2003
# DEFAULT_YEAR_END = 2004
# DEFAULT_RES_M = 3  # reference grid resolution (meters)

# --- AMSTERDAM ---
# DEFAULT_LAT = 52.3676
# DEFAULT_LON = 4.9041
# DEFAULT_KM_BUFFER = 12
# DEFAULT_KM_REDUCED_LAT = 2
# DEFAULT_KM_REDUCED_LON = 2
# DEFAULT_YEAR_START = 2013
# DEFAULT_YEAR_END = 2023
# DEFAULT_RES_M = 3  # reference grid resolution (meters)

# --- GHENT ---
# DEFAULT_LAT = 51.0543
# DEFAULT_LON = 3.7174
# DEFAULT_KM_BUFFER = 6
# DEFAULT_KM_REDUCED_LAT = 1
# DEFAULT_KM_REDUCED_LON = 1
# DEFAULT_YEAR_START = 2013
# DEFAULT_YEAR_END = 2014
# DEFAULT_RES_M = 3  # reference grid resolution (meters)

# --- DORTMUND ---
# DEFAULT_LAT = 51.5090
# DEFAULT_LON = 7.4733
# DEFAULT_KM_BUFFER = 5
# DEFAULT_KM_REDUCED_LAT = 1
# DEFAULT_KM_REDUCED_LON = 2
# DEFAULT_YEAR_START = 2025
# DEFAULT_YEAR_END = 2025
# DEFAULT_RES_M = 2  # reference grid resolution (meters)
DEFAULT_LAT = 51.510
DEFAULT_LON = 7.4733
DEFAULT_KM_BUFFER = 8
DEFAULT_KM_REDUCED_LAT = 3
DEFAULT_KM_REDUCED_LON = 1
DEFAULT_YEAR_START = 2024
DEFAULT_YEAR_END = 2025
DEFAULT_RES_M = 2  # reference grid resolution (meters)

# --- TEMPE ---
# DEFAULT_LAT = 33.425
# DEFAULT_LON = -111.94
# DEFAULT_KM_BUFFER = 6
# DEFAULT_KM_REDUCED_LAT = 1
# DEFAULT_KM_REDUCED_LON = 1
# DEFAULT_YEAR_START = 2016
# DEFAULT_YEAR_END = 2021
# DEFAULT_RES_M = 3  # reference grid resolution (meters)
# --- JODHPUR ---
# DEFAULT_LAT = 26.2389
# DEFAULT_LON = 73.0243
# DEFAULT_KM_BUFFER = 6
# DEFAULT_KM_REDUCED_LAT = 1
# DEFAULT_KM_REDUCED_LON = 1
# DEFAULT_YEAR_START = 2013
# DEFAULT_YEAR_END = 2014
# DEFAULT_RES_M = 2  # reference grid resolution (meters)

# --- ROME ---
# DEFAULT_LAT = 41.9028
# DEFAULT_LON = 12.4964
# DEFAULT_KM_BUFFER = 7        
# DEFAULT_KM_REDUCED_LAT = 2
# DEFAULT_KM_REDUCED_LON = 2
# DEFAULT_YEAR_START = 2013
# DEFAULT_YEAR_END = 2023
# DEFAULT_RES_M = 3               # reference grid resolution (meters)

# Common time lists
HOURS = [f"{h:02d}:00" for h in range(24)]
DAYS  = [f"{d:02d}" for d in range(1, 32)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]


# ------------------------------- Utils -------------------------------------- #

def tlog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def timed(label):
    def deco(f):
        def w(*a, **k):
            t0 = time.time(); tlog(f"{label} ...")
            r = f(*a, **k)
            tlog(f"{label} done in {time.time()-t0:.2f}s\n"); return r
        return w
    return deco

def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)

def fail(msg):
    tlog(f"ERROR: {msg}")
    sys.exit(1)

def clear_osmnx_cache():
    """Best-effort: empty OSMnx on-disk HTTP cache without removing the folder.
    OSMnx typically caches in ox.settings.cache_folder (e.g., ~/.cache/osmnx).
    This function silently ignores errors and logs what it removes.
    """
    try:
        cache_dir = None
        try:
            cache_dir = getattr(ox.settings, "cache_folder", None)
        except Exception:
            cache_dir = None
        if cache_dir is None:
            # fallback to common default
            cache_dir = str(Path.home() / ".cache" / "osmnx")
        cache_path = Path(cache_dir)
        if cache_path.exists() and cache_path.is_dir():
            removed = 0
            for p in cache_path.iterdir():
                try:
                    if p.is_file() or p.is_symlink():
                        p.unlink()
                        removed += 1
                    elif p.is_dir():
                        shutil.rmtree(p)
                        removed += 1
                except Exception:
                    pass
            tlog(f"OSMnx cache cleared: {removed} entries from {cache_path}")
    except Exception as e:
        tlog(f"WARNING: could not clear OSMnx cache: {e}")


# ----------------------------- BBoxes / CRS --------------------------------- #

def utm_epsg_from_latlon(lat, lon):
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"

def compute_bounding_boxes(lat, lon, km_buffer, km_reduced_lat, km_reduced_lon):
    """Compute geographic/UTM bounding boxes and a reduced SOLWEIG domain.

    Returns a BBoxes dataclass with:
    - bbox4326: [min_lon, min_lat, max_lon, max_lat] for downloads
    - bbox_osm: [min_lat, min_lon, max_lat, max_lon] for OSM/CDS area
    - bbox_utm: shapely polygon of the full domain in UTM
    - bbox_utm_solweig: shapely polygon shrunk by km_reduced_* used for SOLWEIG grid
    - crs_utm: target UTM CRS as EPSG string
    """
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * math.cos(math.radians(lat))
    dlat, dlon = km_buffer / km_per_deg_lat, km_buffer / km_per_deg_lon
    min_lat, max_lat, min_lon, max_lon = lat - dlat, lat + dlat, lon - dlon, lon + dlon
    bbox4326 = [min_lon, min_lat, max_lon, max_lat]
    # bbox in lat-lon order for OSM/ERA5 API areas (with +1 km margin)
    extra = 1.0
    bbox_osm = [
        min_lat - extra / km_per_deg_lat,
        min_lon - extra / km_per_deg_lon,
        max_lat + extra / km_per_deg_lat,
        max_lon + extra / km_per_deg_lon,
    ]
    crs_wgs84, crs_utm = "EPSG:4326", utm_epsg_from_latlon(lat, lon)
    bbox_utm = gpd.GeoSeries([box(*bbox4326)], crs=crs_wgs84).to_crs(crs_utm).geometry[0]
    # Shrink for SOLWEIG domain
    sx, sy = km_reduced_lon * 1000.0, km_reduced_lat * 1000.0
    minx, miny, maxx, maxy = bbox_utm.bounds
    bbox_utm_solweig = box(minx + sx, miny + sy, maxx - sx, maxy - sy)
    return BBoxes(bbox4326, bbox_osm, bbox_utm, bbox_utm_solweig, crs_utm)

# -------------------------- Country Helpers (Italy) ------------------------- #

ITALY_REGIONS = {
    "Abruzzo","Basilicata","Calabria","Campania","Emilia-Romagna","Friuli-Venezia Giulia",
    "Lazio","Liguria","Lombardia","Marche","Molise","Piemonte","Puglia","Sardegna",
    "Sicilia","Toscana","Trentino-Alto Adige/Südtirol","Umbria","Valle d'Aosta/Vallée d'Aoste","Veneto"
}

# ---------------------------- Reverse Geocode -------------------------------- #

@timed("Reverse geocoding")
def reverse_geocode(lat, lon):
    """Return (city, state, country_code) using Nominatim; fallback to Unknown."""
    geolocator = Nominatim(user_agent="solweig_builder")
    try:
        loc = geolocator.reverse((lat, lon), exactly_one=True, timeout=15)
        a = (loc.raw.get("address", {}) if loc else {})
        city = a.get("city") or a.get("town") or a.get("village") or a.get("municipality") or "Unknown"
        state = a.get("state", "Unknown")
        cc = (a.get("country_code") or "").lower()
        return city, state, cc
    except Exception:
        return "Unknown", "Unknown", ""

# ----------------------------- Earth Engine ---------------------------------- #
@timed("Initializing Earth Engine")
def safe_initialize_ee():
    """Initialize Earth Engine with fallbacks (local creds or interactive)."""
    import os
    try:
        ee.Initialize(project=os.environ.get("EE_PROJECT")); return
    except Exception:
        try:
            ee.Authenticate(); ee.Initialize(project=os.environ.get("EE_PROJECT")); return
        except Exception as e:
            raise RuntimeError("Earth Engine not initialized. Run ee.Authenticate() or set service account env.") from e

# ------------------------------ OSM / GBA ------------------------------------ #

# Public GBA service providing global LOD1 building footprints/heights via WFS
GBA_URL = "https://tubvsig-so2sat-vm1.srv.mwn.de/geoserver/wfs"
GBA_LAYER = "global3D:lod1_global"
GBA_EE_TILE_ROOT = "projects/sat-io/open-datasets/GLOBAL_BUILDING_ATLAS"

def _bbox_to_poly(b):
    minx, miny, maxx, maxy = b
    return box(minx, miny, maxx, maxy)

def _get_osm_polygons(bbox4326, tags):
    """Fetch OSM features using OSMnx polygon API (requires osmnx>=2)."""
    poly = box(*bbox4326)
    try:
        gdf = ox.features_from_polygon(poly, tags=tags)
    except Exception as e:
        tlog(f"OSM fetch failed: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    gdf = gdf.reset_index(drop=False)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])].copy()


def _format_gba_lon(val: float) -> str:
    prefix = "e" if val >= 0 else "w"
    return f"{prefix}{abs(int(round(val))):03d}"


def _format_gba_lat(val: float) -> str:
    prefix = "n" if val >= 0 else "s"
    return f"{prefix}{abs(int(round(val))):02d}"


def _gba_tile_name(bbox4326, step=5.0) -> str:
    """Return the 5°×5° GBA tile identifier covering the bbox centroid."""

    min_lon, min_lat, max_lon, max_lat = bbox4326
    lon = (min_lon + max_lon) / 2.0
    lat = (min_lat + max_lat) / 2.0

    lon_min = math.floor(lon / step) * step
    lon_max = lon_min + step
    lon_min = max(-180.0, lon_min)
    lon_max = min(180.0, lon_max)

    lat_max = math.ceil(lat / step) * step
    lat_max = min(90.0, lat_max)
    lat_min = lat_max - step
    lat_min = max(-90.0, lat_min)

    return (
        f"{_format_gba_lon(lon_min)}_{_format_gba_lat(lat_max)}_"
        f"{_format_gba_lon(lon_max)}_{_format_gba_lat(lat_min)}"
    )


def _fetch_gba_buildings_tiled(bbox4326, tile_deg=0.1):
    """Fetch buildings from the GBA WFS by tiling the bbox to avoid server limits.

    Returns a GeoDataFrame in EPSG:4326 with only polygonal geometries.
    De-duplicates features across tiles via geometry WKB.
    """
    frames = []
    seen = set()
    to_3857 = Transformer.from_crs(4326, 3857, always_xy=True).transform
    # small overlap to avoid boundary gaps between tiles
    overlap_deg = 0.001
    bb_minx, bb_miny, bb_maxx, bb_maxy = bbox4326
    for (xmin, ymin, xmax, ymax) in _generate_tiles(bbox4326, size_deg=tile_deg):
        try:
            exmin = max(bb_minx, xmin - overlap_deg)
            exymin = max(bb_miny, ymin - overlap_deg)
            exmax = min(bb_maxx, xmax + overlap_deg)
            exymax = min(bb_maxy, ymax + overlap_deg)
            minx, miny = to_3857(exmin, exymin)
            maxx, maxy = to_3857(exmax, exymax)
            bbox_3857_str = f"{minx},{miny},{maxx},{maxy},EPSG:3857"
            params = {
                "service": "WFS", "version": "2.0.0", "request": "GetFeature",
                "typeNames": GBA_LAYER, "outputFormat": "application/json", "bbox": bbox_3857_str,
            }
            r = requests.get(GBA_URL, params=params, timeout=120)
            if not r.ok:
                reason = r.reason or "unknown"
                detail = ""
                try:
                    snippet = r.text.strip().replace("\n", " ")
                    if snippet:
                        detail = f" | {snippet[:180]}"
                except Exception:
                    pass
                tlog(
                    f"GBA request failed ({r.status_code} {reason}) for tile "
                    f"{xmin:.4f},{ymin:.4f},{xmax:.4f},{ymax:.4f}{detail}"
                )
                continue
            tmp = gpd.read_file(io.BytesIO(r.content))
            if tmp is None or tmp.empty:
                continue
            if tmp.crs is None:
                tmp = tmp.set_crs("EPSG:3857")
            tmp = tmp.to_crs(4326)
            tmp = tmp[tmp.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
            # dedupe by geometry wkb
            tmp["_wkb"] = tmp.geometry.apply(lambda g: g.wkb if g is not None else None)
            tmp = tmp[~tmp["_wkb"].isin(seen)]
            seen.update(tmp["_wkb"].dropna().tolist())
            frames.append(tmp.drop(columns=["_wkb"]))
        except Exception as e:
            tlog(f"GBA tile fetch failed: {e}")
            continue
    if not frames:
        tlog("GBA building download produced no data for the requested area")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    return gdf


def _clip_polygons(gdf: gpd.GeoDataFrame, clip_poly_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clip polygons with a polygon clipper; returns input if clipping fails."""
    if gdf is None or gdf.empty:
        return gdf
    try:
        return gpd.overlay(gdf, clip_poly_gdf, how="intersection")
    except Exception:
        try:
            return gdf.clip(clip_poly_gdf.geometry.iloc[0])
        except Exception:
            return gdf


def _grid_cells(bbox4326, cols, rows):
    xmin, ymin, xmax, ymax = bbox4326
    dx = (xmax - xmin) / cols
    dy = (ymax - ymin) / rows
    cells = []
    for r in range(rows):
        for c in range(cols):
            x0 = xmin + c * dx
            x1 = xmin + (c + 1) * dx
            y0 = ymin + r * dy
            y1 = ymin + (r + 1) * dy
            cells.append([x0, y0, x1, y1])
    return cells


def _ee_split_config(bbox4326):
    """Determine EE split heuristics based on AOI dimensions (degrees)."""

    min_lon, min_lat, max_lon, max_lat = bbox4326
    width = max(max_lon - min_lon, 1e-6)
    height = max(max_lat - min_lat, 1e-6)
    area = width * height

    if area <= 0.02:
        return 120000, 1, 1
    if area <= 0.08:
        return 90000, 2, 2
    if area <= 0.2:
        return 70000, 3, 3
    if area <= 0.5:
        return 50000, 3, 3
    if area <= 1.5:
        return 40000, 4, 4
    return 30000, 4, 4


def _fetch_gba_buildings_ee_subset(bbox4326, cache_dir: Path) -> gpd.GeoDataFrame:
    """Download a GBA subset via Earth Engine if available."""

    tile_name = _gba_tile_name(bbox4326)
    asset = f"{GBA_EE_TILE_ROOT}/{tile_name}"
    tlog(f"Attempting GBA EE tile '{asset}'")

    try:
        fc_tile = ee.FeatureCollection(asset)
    except Exception as e:
        tlog(f"GBA EE tile '{asset}' unavailable: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    roi = ee.Geometry.Rectangle(bbox4326, proj=None, geodesic=False)
    subset = fc_tile.filterBounds(roi)
    try:
        count = subset.size().getInfo()
    except Exception as e:
        tlog(f"Failed to count features in GBA EE tile '{asset}': {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    if not count:
        tlog(f"GBA EE tile '{asset}' returned no features for the AOI")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    temp_dir = Path(cache_dir) / "gba_ee_temp"
    ensure_dir(temp_dir)
    exported_paths = []

    def _cleanup_temp():
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

    def _export(fc_obj, suffix):
        out_path = temp_dir / f"gba_subset_{suffix}.geojson"
        try:
            geemap.ee_export_vector(fc_obj, filename=str(out_path))
            if out_path.exists() and out_path.stat().st_size > 0:
                exported_paths.append(out_path)
            else:
                tlog(f"GBA EE export produced empty file: {out_path}")
        except Exception as err:
            tlog(f"GBA EE export failed ({suffix}): {err}")

    split_threshold, grid_cols, grid_rows = _ee_split_config(bbox4326)

    if count <= split_threshold:
        _export(subset, "full")
    else:
        tlog(
            "GBA EE tile contains a large number of features; splitting the AOI "
            f"into {grid_cols}x{grid_rows} cells"
        )
        for idx, cell in enumerate(_grid_cells(bbox4326, grid_cols, grid_rows), start=1):
            cell_geom = ee.Geometry.Rectangle(cell, proj=None, geodesic=False)
            part = subset.filterBounds(cell_geom)
            try:
                n_part = part.size().getInfo()
            except Exception:
                n_part = None
            if not n_part:
                continue
            _export(part, f"part{idx:02d}")

    if not exported_paths:
        tlog("GBA EE export produced no files")
        _cleanup_temp()
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdfs = []
    for fp in exported_paths:
        try:
            df = gpd.read_file(fp)
            if df is None or df.empty:
                continue
            if df.crs is None:
                df = df.set_crs("EPSG:4326")
            else:
                df = df.to_crs("EPSG:4326")
            gdfs.append(df[df.geometry.notnull()])
        except Exception as e:
            tlog(f"Failed to read EE export {fp}: {e}")

    if not gdfs:
        tlog("GBA EE exports could not be read")
        _cleanup_temp()
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    clip_poly = box(*bbox4326)
    gdf = gdf[gdf.geometry.intersects(clip_poly)]
    if gdf.empty:
        tlog("GBA EE subset contains no polygons intersecting the AOI")
        _cleanup_temp()
        return gdf

    tlog(f"GBA EE Buildings (tile {tile_name}): {len(gdf)} polygons")
    result = gdf.reset_index(drop=True)
    _cleanup_temp()
    return result


@timed("Build OSM/GBA vectors")
def build_vectors_from_osm_gba(paths: Paths, bbox4326):
    """Create vegetation.geojson, water.geojson, impervious.geojson, building.geojson using OSM + GBA.
    Simpler, readable flow with polygon-only features for water.
    """
    data_dir = paths.data_dir
    ensure_dir(data_dir)
    poly_clip = gpd.GeoDataFrame(geometry=[_bbox_to_poly(bbox4326)], crs="EPSG:4326")

    def _load_cached(path: Path, label: str):
        if path.exists() and path.stat().st_size > 0:
            try:
                gdf = gpd.read_file(path)
                if gdf is not None and not gdf.empty:
                    tlog(f"{label}: using cached file {path}")
                    return gdf
            except Exception as e:
                tlog(f"{label}: cannot read cached file ({e})")
        return None

    def osm_polygons(tags):
        gdf = _get_osm_polygons(bbox4326, tags)
        return _clip_polygons(gdf, poly_clip)

    # Vegetation
    gdf_green = _load_cached(paths.veg_fp, "OSM Vegetation")
    if gdf_green is None or gdf_green.empty:
        gdf_green = osm_polygons(OSM_TAGS_GREEN)
        if gdf_green is not None and not gdf_green.empty:
            gdf_green.to_file(paths.veg_fp, driver="GeoJSON")
            tlog(f"OSM Vegetation: {len(gdf_green)} polys")

    # Water (polygon-only)
    gdf_water = _load_cached(paths.wat_fp, "OSM Water")
    if gdf_water is None or gdf_water.empty:
        try:
            gdfw = _get_osm_polygons(bbox4326, OSM_TAGS_WATER)
            if gdfw is not None and not gdfw.empty:
                if gdfw.crs is None:
                    gdfw = gdfw.set_crs("EPSG:4326")
                gdfw = gdfw[gdfw.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
                gdf_water = _clip_polygons(gdfw, poly_clip)
                if gdf_water is not None and not gdf_water.empty:
                    gdf_water.to_file(paths.wat_fp, driver="GeoJSON")
                    tlog(f"OSM Water: {len(gdf_water)} polys")
        except Exception as e:
            tlog(f"OSM water download failed: {e}")

    # Impervious
    imprv = _load_cached(paths.imprv_fp, "OSM Impervious")
    if imprv is None or imprv.empty:
        imprv_frames = []
        for tg in (OSM_TAGS_IMPERVIOUS_ASPHALT, OSM_TAGS_IMPERVIOUS_PARKING):
            g = osm_polygons(tg)
            if g is not None and not g.empty:
                imprv_frames.append(g[["geometry"]])
        if imprv_frames:
            imprv = gpd.GeoDataFrame(pd.concat(imprv_frames, ignore_index=True), crs="EPSG:4326")
            imprv.to_file(paths.imprv_fp, driver="GeoJSON")
            tlog(f"OSM Impervious: {len(imprv)} polys")

    # Buildings: prefer GBA WFS tiles, fallback to OSM buildings
    gdf_bld = _load_cached(paths.bld_fp, "Buildings")
    if gdf_bld is None or gdf_bld.empty:
        gdf_bld = _fetch_gba_buildings_tiled(bbox4326, tile_deg=0.1)
        if gdf_bld is not None and not gdf_bld.empty:
            gdf_bld = _clip_polygons(gdf_bld, poly_clip)
            tlog(f"GBA Buildings (tiled): {len(gdf_bld)} polys")
        else:
            tlog("GBA WFS failed; attempting GBA EE tile download")
            gdf_bld = _fetch_gba_buildings_ee_subset(bbox4326, paths.data_dir)
            if gdf_bld is not None and not gdf_bld.empty:
                gdf_bld = _clip_polygons(gdf_bld, poly_clip)
            else:
                tlog("GBA EE download failed; falling back to OSM buildings")
                gdf_bld = osm_polygons({"building": True})
                if gdf_bld is not None and not gdf_bld.empty:
                    gdf_bld = gdf_bld.copy()
                    tlog(f"OSM Buildings fallback: {len(gdf_bld)} polys")
                else:
                    tlog("OSM building fallback produced no data")

    if gdf_bld is not None and not gdf_bld.empty:
        h = pd.to_numeric(gdf_bld.get("height"), errors="coerce")
        gdf_bld = gdf_bld.copy(); gdf_bld["HEIGHT_ROOF"] = h.fillna(10.0).astype(float)
        gdf_bld.to_file(paths.bld_fp, driver="GeoJSON")
        tlog("Saved Buildings with HEIGHT_ROOF (height or 10m default)")

# ------------------------- Downloads (WC, Trees, DEM) ------------------------ #

def _generate_tiles(bbox4326, size_deg=0.05):
    """Yield lon/lat sub-tiles covering the given bbox.

    Parameters
    - bbox4326: [min_lon, min_lat, max_lon, max_lat]
    - size_deg: tile side length in degrees

    Yields
    - Tuples (xmin, ymin, xmax, ymax) in degrees
    """
    minx, miny, maxx, maxy = bbox4326
    x = minx
    while x < maxx:
        nx, y = min(x + size_deg, maxx), miny
        while y < maxy:
            ny = min(y + size_deg, maxy); yield (x, y, nx, ny); y = ny
        x = nx

@timed("WorldCover download")
def download_worldcover(out_tif, bbox4326):
    """Download ESA WorldCover (v200) clipped to bbox.

    Tries a single export first, falling back to a tiled export+mosaic if the
    area is too large for a single Earth Engine request.
    """
    if out_tif.exists():
        tlog(f"WorldCover exists: {out_tif}")
        return
    region = ee.Geometry.BBox(*bbox4326)
    image = ee.ImageCollection("ESA/WorldCover/v200").first()
    # Try single export; if it fails (e.g., >50 MB), fall back to tiling and mosaic
    try:
        geemap.ee_export_image(image, filename=str(out_tif), scale=10, region=region, file_per_band=False)
        if out_tif.exists() and out_tif.stat().st_size > 0:
            return
    except Exception as e:
        tlog(f"WorldCover single export failed: {e}; falling back to tiling")

    tiles_dir = out_tif.parent / "worldcover_tiles"
    ensure_dir(tiles_dir)
    saved = []
    for i, (xmin, ymin, xmax, ymax) in enumerate(_generate_tiles(bbox4326, size_deg=0.2)):
        try:
            tile_region = ee.Geometry.BBox(xmin, ymin, xmax, ymax)
            out_tile = tiles_dir / f"tile_{i}.tif"
            geemap.ee_export_image(image.clip(tile_region), filename=str(out_tile), scale=10, region=tile_region, file_per_band=False)
            if out_tile.exists() and out_tile.stat().st_size > 10_000:
                saved.append(out_tile)
        except Exception as te:
            tlog(f"WorldCover tile {i} failed: {te}")

    if not saved:
        fail("No WorldCover tiles downloaded.")

    srcs = [rasterio.open(fp) for fp in saved]
    try:
        mosaic, transform = merge(srcs)
        meta = srcs[0].meta.copy()
        meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform)
        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(mosaic)
    finally:
        for s in srcs:
            try:
                s.close()
            except Exception:
                pass

@timed("LCZ download")
def download_lcz(out_tif, bbox4326):
    """Download the Global LCZ map (LCZ_Filter band) clipped to bbox.

    Exports a GeoTIFF at ~100 m resolution; later resampled to target grid.
    """
    if out_tif.exists():
        tlog(f"LCZ exists: {out_tif}")
        return
    region = ee.Geometry.BBox(*bbox4326)
    img = ee.ImageCollection("RUB/RUBCLIM/LCZ/global_lcz_map/latest").mosaic().select("LCZ_Filter")
    geemap.ee_export_image(img, filename=str(out_tif), scale=100, region=region, file_per_band=False)

@timed("Tree canopy DSM download/mosaic")
def download_tree_dsm(out_tif: Path, tiles_dir: Path, bbox4326: List[float]):
    """Download a canopy height DSM by trying multiple sources, then mosaic.

    Strategy (kept simple and robust):
    - Try each source in order (meta, eth, umd).
    - For 'eth' and 'umd', attempt a single full export; if it fails or is not
      valid, fall back to tiled export and mosaic.
    - For 'meta', directly use tiled export with smaller tiles (0.05°) to stay
      under Earth Engine request size limits, then mosaic.

    Tiles are written to `tiles_dir` (e.g., `<city>/tree_dsm_tiles`) before
    being mosaicked into `out_tif`.
    """
    if out_tif.exists():
        tlog(f"Tree DSM exists: {out_tif}")
        return

    ensure_dir(tiles_dir)

    region = ee.Geometry.BBox(*bbox4326)

    def _raster_is_valid(fp: Path) -> bool:
        try:
            if not fp.exists() or fp.stat().st_size == 0:
                return False
            with rasterio.open(fp) as src:
                return src.count >= 1 and src.width >= 8 and src.height >= 8
        except Exception:
            return False

    # Keep the original three sources, try in order; no retries, simple fallback to tiling
    sources = [
        ("meta", ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight").mosaic(), 2),
        ("eth", ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1"), 10),
        ("umd", ee.ImageCollection("users/potapovpeter/GEDI_V27").mosaic(), 25),
    ]

    for name, img, scale in sources:
        tlog(f"Trying Tree DSM from source: {name} (scale {scale} m)")
        tile_deg = 0.05 if name == "meta" else 0.2

        # For 'meta' skip single export (requests often exceed EE limits)
        if name != "meta":
            try:
                geemap.ee_export_image(img.clip(region), filename=str(out_tif), scale=scale, region=region, file_per_band=False)
                if _raster_is_valid(out_tif):
                    tlog(f"Tree DSM saved from source '{name}' (single export)")
                    return
            except Exception as e:
                tlog(f"Single export failed for source '{name}': {e}")

        # Tile fallback for this source (writes to <city>/tree_dsm_tiles)
        saved = []
        for i, (xmin, ymin, xmax, ymax) in enumerate(_generate_tiles(bbox4326, size_deg=tile_deg)):
            try:
                tile_region = ee.Geometry.BBox(xmin, ymin, xmax, ymax)
                out_tile = tiles_dir / f"tile_{name}_{i}.tif"
                geemap.ee_export_image(img.clip(tile_region), filename=str(out_tile), scale=scale, region=tile_region, file_per_band=False)
                if _raster_is_valid(out_tile):
                    saved.append(out_tile)
            except Exception as te:
                tlog(f"Tree DSM tile {i} failed for source '{name}': {te}")

        if saved:
            # Mosaic all successfully downloaded tiles into the final DSM
            srcs = [rasterio.open(fp) for fp in saved]
            try:
                mosaic, transform = merge(srcs)
                meta = srcs[0].meta.copy()
                meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform, dtype="float32", nodata=None)
                with rasterio.open(out_tif, "w", **meta) as dst:
                    dst.write(mosaic)
                tlog(f"Tree DSM saved from source '{name}' (tiled mosaic)")
                return
            finally:
                for s in srcs:
                    try:
                        s.close()
                    except Exception:
                        pass
        else:
            tlog(f"No tiles produced for source '{name}', trying next source...")

    fail("No canopy tiles downloaded from any source.")

@timed("DEM download (TINITALY if Italy else MERIT/DEM)")
def download_dem(out_tif, bbox_utm, crs_utm, bbox4326, in_italy):
    if out_tif.exists():
        tlog(f"DEM exists: {out_tif}")
        return

    if in_italy:
        # Use TINITALY via WCS
        tlog("Using TINITALY DEM (WCS)")
        wcs_url = "http://tinitaly.pi.ingv.it/TINItaly_1_1/wcs"
        params = {
            "service": "WCS",
            "version": "1.0.0",
            "request": "GetCoverage",
            "coverage": "TINItaly_1_1:tinitaly_dem",
            "CRS": crs_utm,
            "BBOX": f"{bbox_utm.bounds[0]},{bbox_utm.bounds[1]},{bbox_utm.bounds[2]},{bbox_utm.bounds[3]}",
            "WIDTH": 1000,
            "HEIGHT": 1000,
            "FORMAT": "image/tiff",
        }
        r = requests.get(wcs_url, params=params, timeout=60)
        r.raise_for_status()
        out_tif.write_bytes(r.content)

    else:
        # Use MERIT/DEM via Earth Engine
        tlog("Using MERIT/DEM (Earth Engine)")
        region = ee.Geometry.BBox(*bbox4326)
        image = ee.Image("MERIT/DEM/v1_0_3")
        geemap.ee_export_image(
            image, filename=str(out_tif),
            scale=90, region=region, file_per_band=False
        )

# ------------------------------- Processing ---------------------------------- #

@timed("Reclassify WorldCover → Landuse")
def reclassify_esa_worldcover_inplace(tif_path):
    """Reclassify WorldCover-coded raster into simplified land-use classes.

    Note: this function now targets the Landuse output (not the original
    ESA_WorldCover file). Call it on `paths.landuse_ras` after resampling.

    Mapping used:
    - 1: urban (WorldCover 50)
    - 2: vegetation (default for others)
    - 3: water (WorldCover in WORLD_COVER_WATER)
    - 4: other (WorldCover 60)
    """
    WORLD_COVER_URBAN = [50]
    WORLD_COVER_WATER = [80,70,90]  
    WORLD_COVER_OTHER = [60]

    with rasterio.open(tif_path, "r+") as src:
        wc = src.read(1)
        out = np.full(wc.shape, 2, dtype=np.uint8)  # default vegetation
        out[np.isin(wc, WORLD_COVER_URBAN)] = 1
        out[np.isin(wc, WORLD_COVER_WATER)] = 3
        out[np.isin(wc, WORLD_COVER_OTHER)] = 4
        src.write(out.astype(src.meta["dtype"]), 1)

def compute_reference_grid(bbox_utm, resolution_m):
    """Create a regular grid spec covering bbox_utm with given resolution."""
    minx, miny, maxx, maxy = bbox_utm.bounds
    width = int(np.ceil((maxx - minx) / resolution_m))
    height = int(np.ceil((maxy - miny) / resolution_m))
    transform = from_origin(minx, maxy, resolution_m, resolution_m)
    return GridSpec(transform=transform, width=width, height=height)

def _resample_to_grid(src_path, dst_path, dst_crs, transform, width, height, method):
    """Reproject/resample a single-band raster to the target grid."""
    with rasterio.open(src_path) as src:
        meta = src.meta.copy()
        meta.update(crs=dst_crs, transform=transform, width=width, height=height, compress="lzw", nodata=None)
        with rasterio.open(dst_path, "w", **meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=method,
            )

def _write_all_zeros_raster(out_fp, crs, transform, width, height, dtype="uint8", nodata=0):
    with rasterio.open(
        out_fp, "w", driver="GTiff", height=height, width=width, count=1,
        dtype=dtype, crs=crs, transform=transform, compress="lzw", nodata=nodata
    ) as dst:
        dst.write(np.zeros((height, width), dtype=dtype), 1)

# =================== Save GeoTIFF =================== #
def _save_array_like(reference_fp: Path, out_fp: Path, array: np.ndarray, nodata=np.nan):
    """
    Save a single-band float32 GeoTIFF using metadata from a reference raster.
    """
    import rasterio
    try:
        with rasterio.open(reference_fp) as src:
            meta = src.meta.copy()
        meta.update(dtype="float32", count=1, nodata=nodata)
        with rasterio.open(out_fp, "w", **meta) as dst:
            dst.write(array.astype(np.float32), 1)
    except Exception:
        pass


# =========================================================
# Helpers
# =========================================================
def _safe_tag(tag):
    """Format tag for diagnostics."""
    return f"{str(tag).upper()}-ABOVE"

def _safe_log_ratio_at_z(z, d, z0, tag="trees"):
    """
    Safe logarithmic ratio used in *above-canopy* branches:
        ln( max(z - d, 1.01*z0_safe) / z0_safe )

    - Floors z0 to a small positive number.
    - Clamps numerator so the ratio is >= 1.01.
    - Diagnostics suppressed.
    """
    z0_floor = 1e-4  # avoid zeros and denormals
    z0_safe  = np.maximum(z0, z0_floor)

    num_arg  = np.maximum(z - d, 1.01 * z0_safe)  # ensure > z0_safe
    den_arg  = z0_safe
    ratio    = num_arg / den_arg

    # Diagnostics suppressed

    # Guarantee strictly positive argument for log
    ratio_safe = np.maximum(ratio, 1.01)  # keep it > 1 so log > 0
    log_ratio  = np.log(ratio_safe)

    # Secondary diagnostics suppressed

    return log_ratio


# =========================================================
# Window stats (unchanged except for logging)
# =========================================================
def _win_stats(mask, heights, window_px):
    """Compute moving-window plan fraction (λp) and windowed mean height H_mean
    using a *rectangular* (box) footprint of side `window_px`.

    Parameters
    ----------
    mask : 2D array-like of bool/0-1 (True where object exists)
    heights : 2D array-like of float (per-pixel heights in meters)
    window_px : int window *side* in pixels (>=1). It will be made odd.

    Returns
    -------
    lp : 2D float32 array in [0,1]  -- moving window plan fraction
    H_mean : 2D float32 array (m)   -- moving window **mean** height over masked area
    """
    import numpy as np
    from scipy.ndimage import uniform_filter

    if window_px < 1:
        window_px = 1
    # ensure odd size for symmetry
    if window_px % 2 == 0:
        window_px += 1

    m = mask.astype(np.float32)
    # zero heights outside mask so mean_h/mean_m = sum_h/sum_m
    h = np.where(mask, heights, 0.0).astype(np.float32)

    # box means in the window
    mean_m = uniform_filter(m, size=window_px, mode="nearest")
    mean_h = uniform_filter(h, size=window_px, mode="nearest")

    # plan fraction = mean of mask in the window
    lp = mean_m

    # conditional mean height over masked area
    with np.errstate(invalid="ignore", divide="ignore"):
        H_mean = np.where(mean_m > 0, mean_h / mean_m, 0.0)

    return lp.astype(np.float32), H_mean.astype(np.float32)


# =========================================================
# λf from edges (height-weighted, true vertical area sum)
# =========================================================
# =========================================================
# λf from edges (directional; building-side, cylindrical average)
# =========================================================

def _lambdaf_edges(mask, heights, window_px, pixel_size, clip=(0.0, 1.0)):
    """
    Frontal area index λf via **directional building-side edges** to reduce
    raster stair-stepping artifacts. We:
      1) detect building boundary edges in the 4 cardinal directions
         (W, E, N, S) where a building pixel borders free space;
      2) assign each edge's **vertical area** = height(built-side) × edge length (px)
         to the **building pixel** only (no double-side counting);
      3) aggregate vertical areas in a moving *box* window of side `window_px`;
      4) convert to frontal area density per ground area for each direction;
      5) sum the 4 directions to get the total lateral vertical area and scale by 1/π (Cauchy mean projection for isotropic wind).

    Parameters
    ----------
    mask : 2D bool/0-1 array (True where building exists)
    heights : 2D float array (building height in meters)
    window_px : int, box window *side* in pixels (>=1). Made odd if even.
    pixel_size : float, pixel size in meters (assumed square pixels)
    clip : tuple(float,float), clamp range for λf

    Returns
    -------
    lf : 2D float32 λf (m² vertical wall / m² ground), isotropic mean (1/π of lateral area)
    """
    import numpy as np
    from scipy.ndimage import uniform_filter

    if window_px < 1:
        window_px = 1
    if window_px % 2 == 0:
        window_px += 1

    m = mask.astype(bool)
    H = np.where(m, heights, 0.0).astype(np.float32)
    px = float(pixel_size)

    # Pad arrays to simplify neighbor tests at borders
    # We'll work on original grid by slicing back after computing edge maps
    m_pad = np.pad(m, ((1,1),(1,1)), mode="edge")
    H_pad = np.pad(H, ((1,1),(1,1)), mode="edge")

    # Neighbor masks relative to **interior** (center) pixel being building
    # West edge: building pixel with left neighbor = False
    W_edge = m_pad[1:-1,1:-1] & (~m_pad[1:-1,0:-2])
    E_edge = m_pad[1:-1,1:-1] & (~m_pad[1:-1,2:  ])
    N_edge = m_pad[1:-1,1:-1] & (~m_pad[0:-2,1:-1])
    S_edge = m_pad[1:-1,1:-1] & (~m_pad[2:  ,1:-1])

    # Directional vertical wall areas (assign to the **building pixel itself**)
    A_W = np.where(W_edge, H, 0.0).astype(np.float32) * px
    A_E = np.where(E_edge, H, 0.0).astype(np.float32) * px
    A_N = np.where(N_edge, H, 0.0).astype(np.float32) * px
    A_S = np.where(S_edge, H, 0.0).astype(np.float32) * px

    # Aggregate each directional area within the moving window
    mean_A_W = uniform_filter(A_W, size=window_px, mode="nearest")
    mean_A_E = uniform_filter(A_E, size=window_px, mode="nearest")
    mean_A_N = uniform_filter(A_N, size=window_px, mode="nearest")
    mean_A_S = uniform_filter(A_S, size=window_px, mode="nearest")

    # Convert means back to sums in the window
    Npix = float(window_px * window_px)
    sum_A_W = mean_A_W * Npix
    sum_A_E = mean_A_E * Npix
    sum_A_N = mean_A_N * Npix
    sum_A_S = mean_A_S * Npix

    # Ground area of the window (m²)
    window_area_m2 = max(Npix * (px * px), 1.0)

    # Directional frontal area densities (m² wall / m² ground)
    Af_W = sum_A_W / window_area_m2
    Af_E = sum_A_E / window_area_m2
    Af_N = sum_A_N / window_area_m2
    Af_S = sum_A_S / window_area_m2

    # Isotropic mean frontal area using Cauchy projection: Ā = (1/π) * h * P
    # Here Af_total is the lateral area per ground area (h * perimeter / window_area)
    Af_total = (Af_W + Af_E + Af_N + Af_S)
    lf = (1.0 / np.pi) * Af_total

    lf = lf.astype(np.float32)
    lf[~np.isfinite(lf)] = 0.0
    lf = np.clip(lf, *clip)
    return lf



# =========================================================
# Buildings: coefficient at z (z_ref == z_eval)
# =========================================================
def _coeff_at_z_buildings(mean_heights, lambdaf,
                          z_eval=10.0, zref=10.0, z0_ref=0.7,
                          lp_min_open=0.0,
                          clamp=(0.01, 1.5),
                          H_raw_local=None):
    """Window-based building coefficient derived from neighbourhood stats."""
    H = np.asarray(mean_heights, dtype=np.float32)
    lam_eff = np.clip(np.asarray(lambdaf, dtype=np.float32), 0.0, 1.0)
    z = float(z_eval)

    clamp_min, clamp_max = clamp
    C = np.ones_like(H, dtype=np.float32)

    active = np.isfinite(H) & np.isfinite(lam_eff) & np.isfinite(lambdaf) & (lam_eff > lp_min_open)
    if not np.any(active):
        return C

    z0u = np.maximum(0.1*H.astype(np.float32), 0.01)
    d = np.maximum(0.7*H.astype(np.float32), 0.0)

    den_ref = math.log(max(z, 1.01 * z0_ref) / max(z0_ref, 1e-6))
    if den_ref == 0:
        den_ref = 1e-6

    inside = active & (H > z)
    outside = active & (~inside)

    if np.any(inside):
        H_i = np.maximum(H[inside], 1e-6)
        z0_i = z0u[inside]
        d_i = d[inside]
        a_i = 9.6 * lam_eff[inside]
        ratio_top = np.maximum((H_i - d_i) / z0_i, 1.01)
        corr = np.log(ratio_top) / den_ref
        atten = np.exp(a_i * (z / H_i - 1.0))
        C_in = atten * corr
        C[inside] = C_in.astype(np.float32)

    if np.any(outside):
        H_o = np.maximum(H[outside], 1e-6)
        z0_o = z0u[outside]
        d_o = d[outside]
        ratio = (z - d_o) / z0_o
        ratio = np.where(ratio <= 1.0, 1.01, ratio)
        num = np.log(ratio)
        C_out = num / den_ref
        C[outside] = C_out.astype(np.float32)

    return np.clip(C, None, clamp_max)

# =========================================================
# Trees: coefficient at z (z_ref == z_eval)
# =========================================================
def _coeff_at_z_trees(H_raw, H_mean, lp_t, *,
                      z_eval=10.0, zref=10.0, z0_ref=0.7,
                      LAI=4.0, a0=0.5, a1=0.2,
                      alpha_min=0.6, alpha_max=3.5,
                      hmin=1.0, lp_min_open=0.0,
                      clamp=(0.01, 1.5)):
    """Window-based vegetation coefficient without pixel overrides."""
    if H_mean is not None:
        Hm = np.asarray(H_mean, dtype=np.float32)
    else:
        Hm = np.asarray(H_raw, dtype=np.float32)
    lam_eff = np.clip(np.asarray(lp_t, dtype=np.float32), 0.0, 1.0)
    z = float(z_eval)

    clamp_min, clamp_max = clamp
    C = np.ones_like(Hm, dtype=np.float32)

    active = np.isfinite(Hm) & np.isfinite(lam_eff) & (lam_eff > lp_min_open)
    if not np.any(active):
        return C

    den_ref = math.log(max(z, 1.01 * z0_ref) / max(z0_ref, 1e-6))
    if den_ref == 0:
        den_ref = 1e-6

    lai_eff = LAI * lam_eff
    alpha = np.clip(a0 + a1 * lai_eff, alpha_min, alpha_max).astype(np.float32)

    inside = active & (Hm > z)
    outside = active & (~inside)

    if np.any(inside):
        H_i = np.maximum(Hm[inside], 1e-6)
        z0_i = np.maximum(0.1 * H_i, 0.05)
        d_i = 0.7 * H_i
        ratio_top = np.maximum((H_i - d_i) / z0_i, 1.01)
        corr = np.log(ratio_top) / den_ref
        frac = np.maximum(0.0, 1.0 - (z / H_i))
        atten = np.exp(-alpha[inside] * frac)
        C_in = atten * corr
        C[inside] = C_in.astype(np.float32)

    if np.any(outside):
        H_o = np.maximum(Hm[outside], 1e-6)
        z0_o = np.maximum(0.1 * H_o, 0.05)
        d_o = 0.7 * H_o
        ratio = (z - d_o) / z0_o
        ratio = np.where(ratio <= 1.0, 1.01, ratio)
        num = np.log(ratio)
        C_out = num / den_ref
        C[outside] = C_out.astype(np.float32)

    return np.clip(C, clamp_min, clamp_max)

# ------------------- Fill missing values ------------------- #
def _fill_nearest(arr, fill_value=1.0):
    """
    Fill NaNs with the nearest valid value (Euclidean Distance Transform).
    If the whole array is NaN, fill with `fill_value`.
    """
    out = arr.astype(np.float32, copy=True)
    valid = np.isfinite(out)
    if not np.any(~valid):
        return out
    if not np.any(valid):
        out[:] = fill_value
        return out
    idx = distance_transform_edt(~valid, return_indices=True)
    filled = out[tuple(idx)]
    result = np.where(valid, out, filled)
    return result

# =================== Main computation (z = 10 m) =================== #
def compute_wind_coeff(paths, win_m=100.0, smooth_m=250.0, hmin_b=1.0, hmin_t=1.0,
                       z_eval=10.0, zref=10.0, z0_ref=0.7,
                       # Vegetation alpha parameters (kept for _coeff_at_z_trees)
                       LAI_t=2.0, a0_t=0.5, a1_t=0.4,
                       alpha_min_t=0.2, alpha_max_t=2.5,
                       coeff_min=0.01, coeff_max=1.00, lp_min_open=0.02):
    """
    Build wind reduction coefficients at height `z_eval` using *moving* windows:
      • λp and H_mean computed with sliding uniform windows of ~100 m;
      • C_buildings and C_trees computed separately;
      • each set to 1 where its own λp ≤ 0.02;
      • total coefficient = C_buildings × C_trees;
      • each component is smoothed separately with a *moving* Gaussian window equivalent to ~200 m (configurable via smooth_m; ±3σ ≈ smooth_m), then multiplied to obtain the total.
    """
    import numpy as np
    import rasterio

    # 1) Load aligned rasters
    if not paths.bld_ras.exists() or not paths.tree_ras.exists():
        return

    with rasterio.open(paths.bld_ras) as src_b:
        B = src_b.read(1, masked=True).filled(np.nan).astype(np.float32)
        transform = src_b.transform
        meta_ref = src_b.profile
    with rasterio.open(paths.tree_ras) as src_t:
        T = src_t.read(1, masked=True).filled(np.nan).astype(np.float32)
        if src_t.shape != B.shape:
            raise ValueError("Trees raster shape differs from Buildings raster")

    # pixel size → window (odd integer)
    px = abs(transform.a)
    ws = max(1, int(round(win_m / max(px, 1e-6))))
    if ws % 2 == 0:
        ws += 1

    # 2) Basic masks and cutoff by minimal heights
    Hb = np.where(np.isfinite(B) & (B >= hmin_b), B, 0.0)
    Ht = np.where(np.isfinite(T) & (T >= hmin_t), T, 0.0)
    Mb = Hb > 0
    Mt = Ht > 0

    # 3) Moving-window stats (λp, H_mean) for buildings and trees
    lp_b, Hb_mean = _win_stats(Mb, Hb, ws)
    lp_t, Ht_mean = _win_stats(Mt, Ht, ws)

    # 4) Frontal index for buildings (λf) from edges
    lf_b = _lambdaf_edges(Mb, Hb_mean, ws, px)

    # 5) Coefficients at z (inside/outside handled inside the functions)
    Cb = _coeff_at_z_buildings(
        Hb_mean,lf_b,
        z_eval=z_eval, zref=zref, z0_ref=z0_ref,
        lp_min_open=lp_min_open, clamp=(coeff_min, coeff_max), H_raw_local=Hb,
    ).astype(np.float32)

    Ct = _coeff_at_z_trees(
        H_raw=Ht, H_mean=Ht_mean, lp_t=lp_t,
        z_eval=z_eval, zref=zref, z0_ref=z0_ref,
        LAI=LAI_t, a0=a0_t, a1=a1_t,
        alpha_min=alpha_min_t, alpha_max=alpha_max_t,
        hmin=hmin_t, lp_min_open=lp_min_open,
        clamp=(coeff_min, coeff_max),
    ).astype(np.float32)

    # 6) Set C=1 where λp ≤ 0.02 (individually for each layer)
    Cb = np.where(lp_b > lp_min_open, Cb, 1.0).astype(np.float32)
    Ct = np.where(lp_t > lp_min_open, Ct, 1.0).astype(np.float32)

    # 6.a) Local override for trees with a window of ~10 m
    # Calculate λp and H_mean on a small moving window (≈10 m) and recombine Ct on this scale.
    ws10 = max(1, int(round(10.0 / max(px, 1e-6))))
    if ws10 % 2 == 0:
        ws10 += 1

    lp_t_10, Ht_mean_10 = _win_stats(Mt, Ht, ws10)
    Ct_10 = _coeff_at_z_trees(
        H_raw=Ht, H_mean=Ht_mean_10, lp_t=lp_t_10,
        z_eval=z_eval, zref=zref, z0_ref=z0_ref,
        LAI=LAI_t, a0=a0_t, a1=a1_t,
        alpha_min=alpha_min_t, alpha_max=alpha_max_t,
        hmin=hmin_t, lp_min_open=lp_min_open,
        clamp=(coeff_min, coeff_max),
    ).astype(np.float32)
    # Also apply here the rule lp<=threshold -> 1.0
    Ct_10 = np.where(lp_t_10 > lp_min_open, Ct_10, 1.0).astype(np.float32)

    # Where the 10 m value is < 1, replace it in the main field
    Ct = np.where(Ct_10 < Ct, Ct_10, Ct).astype(np.float32)

    

    # 8) Gaussian smoothing
    #    Apply *anchor-preserving* smoothing ONLY to the trees coefficient (Ct),
    #    while smoothing buildings (Cb) with a standard Gaussian (no anchors).
    #    Then, smooth the total coefficient again at the end.
    #    Window equivalence: ±3σ ≈ smooth_m ⇒ σ_px = (smooth_m/px)/6.
    sigma_px = max(1, (smooth_m / max(px, 1e-6)) / 6.0)

    def _gauss_preserve(arr, anchor_mask):
        """Gaussian smoothing with exact preservation at anchor pixels.
        Implementation: filter (values*weights) and (weights) separately, with large
        weights (e.g., 1e6) on anchors. After filtering, enforce original values on anchors.
        """
        valid = np.isfinite(arr)
        if not np.any(valid):
            return arr
        w = valid.astype(np.float32)
        w_anchor = (anchor_mask & valid).astype(np.float32) * 1e6
        w = w + w_anchor
        num0 = np.where(valid, arr, 0.0).astype(np.float32) * w
        num = gaussian_filter(num0, sigma=0.5*sigma_px, mode="nearest")
        den = gaussian_filter(w,    sigma=0.5*sigma_px, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(den > 1e-6, num / den, arr)
        out = np.where(anchor_mask, arr, out)
        return np.clip(out.astype(np.float32), coeff_min, coeff_max)

    
    # Buildings: standard Gaussian smoothing (no anchors)
    Cb_s =  _gauss_preserve(Cb, Mb)

    # Trees: anchor-preserving smoothing
    Ct_s = _gauss_preserve(Ct, Mt)

    # 8.b) Total coefficient = product of smoothed components
    Ctot = np.clip(Cb_s * Ct_s, coeff_min, coeff_max).astype(np.float32)

    # 8.c) Smooth the total again (light additional smoothing with same σ)
    Ctot = gaussian_filter(Ctot.astype(np.float32), sigma=2*sigma_px, mode="nearest")
    Ctot = np.clip(Ctot.astype(np.float32), coeff_min, coeff_max)

    # 9) Saving: components and total
    def _save_like(ref_meta, out_fp, arr):
        m = ref_meta.copy(); m.update(dtype="float32", count=1, compress="lzw", nodata=np.nan)
        with rasterio.open(out_fp, "w", **m) as dst:
            dst.write(arr.astype(np.float32), 1)

    ensure_dir(paths.out_dir)
    _save_like(meta_ref, paths.out_dir / "WindCoeff_Urban.tif", Cb)
    _save_like(meta_ref, paths.out_dir / "WindCoeff_Vegetation.tif", Ct)
    _save_like(meta_ref, paths.wind_coeff_ras, Ctot)

def _read_vector_or_warn(vector_fp, friendly_name):
    if not vector_fp.exists():
        # removed logging
        return None
    try:
        gdf = gpd.read_file(vector_fp)
    except Exception as e:
        # removed logging
        return None
    if gdf.empty:
        # removed logging
        return None
    if gdf.crs is None:
        # removed logging
        return None
    return gdf

def rasterize_vector_checked(vector_fp, out_fp, value_field, transform, width, height, crs, friendly_name, dtype="uint8"):
    gdf = _read_vector_or_warn(vector_fp, friendly_name)
    if gdf is None:
        _write_all_zeros_raster(out_fp, crs, transform, width, height, dtype=dtype, nodata=0)
        return

    if gdf.crs.to_string() != crs:
        try:
            gdf = gdf.to_crs(crs)
        except Exception as e:
            # removed logging
            _write_all_zeros_raster(out_fp, crs, transform, width, height, dtype=dtype, nodata=0)
            return

    if isinstance(value_field, str):
        if value_field not in gdf.columns:
            # removed logging
            _write_all_zeros_raster(out_fp, crs, transform, width, height, dtype=dtype, nodata=0)
            return
        shapes = ((geom, val) for geom, val in zip(gdf.geometry, gdf[value_field]))
    else:
        shapes = ((geom, value_field) for geom in gdf.geometry)

    arr = rasterize(shapes=shapes, out_shape=(height, width), transform=transform, fill=0, dtype=dtype)
    with rasterio.open(
        out_fp, "w", driver="GTiff", height=height, width=width, count=1,
        dtype=dtype, crs=crs, transform=transform, compress="lzw", nodata=0
    ) as dst:
        dst.write(arr, 1)

def rasterize_polygons_to_array(gdf: gpd.GeoDataFrame, transform, width: int, height: int, crs: str, value: int = 1, dtype="uint8"):
    """Rasterize a polygon GeoDataFrame to an in-memory array."""
    if gdf is None or gdf.empty:
        return np.zeros((height, width), dtype=dtype)
    if gdf.crs is None or gdf.crs.to_string() != crs:
        try:
            gdf = gdf.to_crs(crs)
        except Exception:
            return np.zeros((height, width), dtype=dtype)
    shapes = ((geom, value) for geom in gdf.geometry)
    return rasterize(shapes=shapes, out_shape=(height, width), transform=transform, fill=0, dtype=dtype)

@timed("Apply raster logic")
def apply_raster_logic(building_fp, tree_fp, landuse_fp, water_fp, vegetation_fp, dem_fp, dem_plus_building_fp, urban_arr=None):
    """Apply layer precedence and compute the Building DSM.

    - Trees are zeroed wherever buildings exist (canopies cannot overlap roofs).
    - Landuse precedence: water(3) > vegetation(2) > urban(1). Urban can come
      from rasterized OSM impervious∪buildings (urban_arr) if provided.
    - Building DSM is computed as DEM + building heights (meter units).
    """
    with rasterio.open(building_fp) as b, \
         rasterio.open(tree_fp, "r+") as t, \
         rasterio.open(landuse_fp, "r+") as l, \
         rasterio.open(water_fp) as w, \
         rasterio.open(vegetation_fp) as v, \
         rasterio.open(dem_fp) as d:

        b_arr, t_arr, l_arr = b.read(1), t.read(1), l.read(1)
        w_arr, v_arr = w.read(1), v.read(1)
        if urban_arr is None:
            # no urban override available
            import numpy as np
            u_arr = np.zeros_like(l_arr, dtype=np.uint8)
        else:
            u_arr = urban_arr.astype(l_arr.dtype)
        dem_arr = d.read(1)

        # Trees: zero over buildings
        t_arr[b_arr > 0] = 0
        # Landuse stacking: urban(1) → vegetation(2) overrides urban → water(3) overrides all
        l_arr[w_arr > 0] = 3
        l_arr[v_arr > 0] = 2
        l_arr[u_arr > 0] = 1


        t.write(t_arr, 1)
        l.write(l_arr, 1)

        dem_plus = dem_arr.astype("float32") + b_arr.astype("float32")
        prof = d.profile
        prof.update(dtype="float32", compress="lzw", nodata=None)
        with rasterio.open(dem_plus_building_fp, "w", **prof) as dst:
            dst.write(dem_plus, 1)

def check_raster_alignment(paths):
    """Ensure a group of rasters share CRS, resolution, bounds and shape.

    Raises a ValueError if any mismatch is detected to avoid mixing misaligned
    layers in downstream processing.

    Parameters
    - paths: iterable of Path objects to GeoTIFF rasters to check
    """
    from rasterio.errors import RasterioIOError
    refs = []
    for p in paths:
        try:
            with rasterio.open(p) as src:
                refs.append({
                    "name": p.name,
                    "crs": src.crs.to_string(),
                    "res": src.res,
                    "bounds": tuple(map(lambda v: round(v, 6), (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top))),
                    "shape": (src.height, src.width),
                })
        except RasterioIOError:
            fail(f"Cannot open raster: {p}")
    ref0 = refs[0]
    for r in refs[1:]:
        for k in ("crs", "res", "bounds", "shape"):
            if r[k] != ref0[k]:
                raise ValueError(
                    f"Mismatch in '{k}' between {ref0['name']} and {r['name']}:\n"
                    f"  {ref0['name']} → {ref0[k]}\n  {r['name']} → {r[k]}"
                )

def final_report_table(paths):
    """Print a compact table with basic metadata for final rasters.

    Columns include EPSG code, width/height, pixel size and bounds.
    Non-existent or non-TIFF paths are ignored.
    """
    rows = []
    for p in paths:
        try:
            if not p.exists() or p.suffix.lower() != ".tif":
                continue
            with rasterio.open(p) as src:
                rows.append({
                    "name": p.name,
                    "epsg": src.crs.to_epsg() if src.crs else None,
                    "w": src.width, "h": src.height, "rx": src.res[0], "ry": src.res[1],
                    "minx": round(src.bounds.left, 3), "miny": round(src.bounds.bottom, 3),
                    "maxx": round(src.bounds.right, 3), "maxy": round(src.bounds.top, 3),
                })
        except Exception:
            continue
    print(pd.DataFrame(rows).to_string(index=False))

@timed("ERA5 hourly → NetCDF")
def download_and_embed_era5(out_nc, bbox_osm, year_start, year_end, data_dir, in_italy):
    """Download+build meteo forcing via DDS (Italy) or ERA5 (CDS) with a single, standardized output.
    - DDS: hourly GRIB + const NetCDF (HSURF). Compute RH, standardize names, keep only wanted variables.
    - ERA5: yearly GRIB ZIPs → unzip → grib_to_netcdf → concat years, compute HGT (z/g), RH, divide SSRD/STRD by 3600, standardize names, keep only wanted vars.
    """
    import os
    import xarray as xr

    out_nc = Path(out_nc)
    data_dir = Path(data_dir)
    ensure_dir(data_dir)

    if out_nc.exists():
        tlog(f"ERA5 hourly+geopotential already present: {out_nc}")
        return

    # Determine source area from bbox_osm
    min_lat, min_lon, max_lat, max_lon = bbox_osm

    if in_italy:
        # ------------------- ITALY (DDSAPI) -------------------
        import ddsapi
        client = ddsapi.Client()
        hourly_nc = data_dir / "era5_hourly.tmp.nc"
        const_nc  = data_dir / "era5_const.tmp.nc"
        out_tmp   = data_dir / (out_nc.name + ".tmp")
        area = {"north": max_lat, "south": min_lat, "east": max_lon, "west": min_lon}

        tlog("Downloading ERA5 (Italy downscaled) hourly")
        client.retrieve(
            "era5-downscaled-over-italy", "hourly",
            {
                "area": area,
                "vertical": [0.005],
                "time": {
                    "hour":  [f"{h:02}" for h in range(24)],
                    "year":  [str(y) for y in range(year_start, year_end + 1)],
                    "month": [str(m) for m in range(1, 13)],
                    "day":   [str(d) for d in range(1, 32)],
                },
                "variable": [
                    "grid_northward_wind",
                    "air_pressure_at_sea_level",
                    "surface_net_downward_shortwave_flux",
                    "surface_net_downward_longwave_flux",
                    "air_temperature",
                    "grid_eastward_wind",
                    "specific_humidity",
                ],
                "format": "netcdf",
            },
            str(hourly_nc),
        )

        tlog("Downloading ERA5 const (surface_altitude)")
        client.retrieve(
            "era5-downscaled-over-italy", "const",
            {"area": area, "variable": ["surface_altitude"], "format": "netcdf"},
            str(const_nc),
        )

        tlog("Embedding HSURF, computing RH, and standardizing names")
        with xr.open_dataset(hourly_nc, use_cftime=True) as ds_h, xr.open_dataset(const_nc, use_cftime=True) as ds_c:
            hsurf = ds_c["HSURF"].squeeze()
            if "time" in hsurf.dims:
                hsurf = hsurf.isel(time=0, drop=True)
            ds_h["surface_altitude"] = hsurf
            if "valid_time" in ds_h.variables: ds_h = ds_h.drop_vars("valid_time")
            # Relative humidity from T, q, and sea-level pressure (DDS variables)
            Tvar = ds_h.get("air_temperature") or ds_h.get("T_2M")
            Qvar = ds_h.get("specific_humidity") or ds_h.get("QV_2M")
            Pvar = ds_h.get("air_pressure_at_sea_level") or ds_h.get("PMSL")
            if (Tvar is not None) and (Qvar is not None) and (Pvar is not None):
                T = Tvar  # K
                q = Qvar  # kg/kg
                p = Pvar  # Pa
                e  = q * p / (0.622 + 0.378*q)     # vapor pressure (Pa)
                es = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))  # saturation vapor pressure (Pa)
                ds_h["relative_humidity"] = (100.0 * e/es).clip(min=0, max=100)
                # Rename immediately to requested name
                ds_h = ds_h.rename({"relative_humidity": "RH2M"})
            # Standardize dims/vars, keep only requested, set time encoding
            ren_dims = {k:v for k,v in DIMS_MAP.items() if (k in ds_h.dims or k in ds_h.coords)}
            ren_vars = {k:v for k,v in VAR_MAP.items()  if k in ds_h.variables}
            ds_h = ds_h.rename({**ren_dims, **ren_vars})
            # Exclude roughness length (Z0) for DDS over Italy per request
            wanted_no_z0 = [v for v in WANTED_VARS if v != "Z0"]
            ds_h = ds_h[[v for v in wanted_no_z0 if v in ds_h]]
            ds_h = add_uhi_indicator(ds_h)
            if "time" in ds_h.coords:
                ds_h["time"].encoding.update(units="hours since 1900-01-01 00:00:00", calendar="gregorian")

            ds_h.to_netcdf(str(out_tmp), format="NETCDF4", engine="netcdf4")
            os.replace(out_tmp, out_nc)

        # Keep DDS intermediate files for inspection (requested)
        tlog("Keeping DDS intermediate NetCDF files (hourly/const)")

        tlog(f"Saved ERA5 hourly + HSURF + RH: {out_nc}")
        return

    # ------------------- REST OF THE WORLD (Earth Engine single point) -------------------
    out_tmp = data_dir / (out_nc.name + ".tmp")

    # Use bbox centre as target coordinate for the point extraction
    lat = (min_lat + max_lat) / 2.0
    lon = (min_lon + max_lon) / 2.0
    point = ee.Geometry.Point([lon, lat])

    # Base bands
    bands = [
        "temperature_2m",
        "dewpoint_temperature_2m",
        "surface_pressure",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downwards",
        "geopotential",
    ]
    rename_map = {
        "temperature_2m": "t2m",
        "dewpoint_temperature_2m": "d2m",
        "surface_pressure": "sp",
        "u_component_of_wind_10m": "u10",
        "v_component_of_wind_10m": "v10",
        "surface_solar_radiation_downwards": "ssrd",
        "surface_thermal_radiation_downwards": "strd",
        "geopotential": "z",
    }
    # Try to include forecast_surface_roughness if available
    try:
        avail = ee.ImageCollection("ECMWF/ERA5/HOURLY").first().bandNames().getInfo()
    except Exception:
        avail = None
    extra_candidates = {
        "forecast_surface_roughness": "z0",
        # Common aliases in other ERA5 catalogs (fallbacks)
        "surface_roughness": "z0",
        "roughness_length": "z0",
        "z0": "z0",
    }
    if isinstance(avail, list):
        for bname, short in extra_candidates.items():
            if bname in avail and bname not in bands:
                bands.append(bname)
                rename_map[bname] = short

    frames = []
    now_utc = pd.Timestamp.now(tz="UTC")
    curr_year, curr_month = int(now_utc.year), int(now_utc.month)
    stop_future = False

    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            # Process only months strictly earlier than the *current* month.
            # Example: if today is 2025-10-01, we stop at 2025-09.
            if (year > curr_year) or (year == curr_year and month >= curr_month):
                tlog(f"Reached current month ({year}-{month:02d}); stopping ERA5 download at previous complete month.")
                stop_future = True
                break

            start_ee = ee.Date.fromYMD(year, month, 1)
            end_ee = start_ee.advance(1, "month")
            label = f"{year}-{month:02d}"

            tlog(f"Downloading ERA5 hourly {label} for point ({lat:.4f}, {lon:.4f}) via Earth Engine")
            col = (
                ee.ImageCollection("ECMWF/ERA5/HOURLY")
                .filterDate(start_ee, end_ee)
                .filterBounds(point)
                .select(bands)
            )

            tries, delay = 3, 10
            last_err = None
            for attempt in range(tries):
                try:
                    tbl = col.getRegion(point, 1000).getInfo()
                    break
                except Exception as e:
                    last_err = e
                    tlog(f"[{label}] getRegion attempt {attempt + 1}/{tries} failed: {e}")
                    time.sleep(delay)
            else:
                raise RuntimeError(f"Download failed for {label}") from last_err

            if not tbl or len(tbl) < 2:
                tlog(f"No ERA5 data for {label} at the requested point")
                continue

            df_month = pd.DataFrame(tbl[1:], columns=tbl[0])
            df_month = df_month.dropna(subset=["time"]).copy()
            keep_cols = ["time", "longitude", "latitude"] + bands
            df_month = df_month[[c for c in keep_cols if c in df_month.columns]]
            frames.append(df_month)

        if stop_future:
            break

    if not frames:
        fail("ERA5 point extraction produced no data")

    df_all = pd.concat(frames, ignore_index=True)
    df_all["time"] = pd.to_datetime(df_all["time"], unit="ms", utc=True)
    df_all = df_all.sort_values("time").reset_index(drop=True)
    df_all["time"] = df_all["time"].dt.tz_convert("UTC").dt.tz_localize(None)

    numeric_cols = ["longitude", "latitude"] + bands
    for col in numeric_cols:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    df_all = df_all.dropna(subset=[col for col in numeric_cols if col in df_all.columns])
    if df_all.empty:
        fail("ERA5 point dataframe empty after cleaning")

    df_all = df_all.rename(columns=rename_map)
    df_all = df_all.drop_duplicates(subset="time")

    times = pd.DatetimeIndex(df_all["time"])  # timezone-naive, monotonic
    lat_val = float(df_all["latitude"].mean())
    lon_val = float(df_all["longitude"].mean())

    data_vars = {}
    for var in rename_map.values():
        if var in df_all.columns:
            data = df_all[var].to_numpy(dtype=float).reshape(-1, 1, 1)
            data_vars[var] = xr.DataArray(
                data,
                coords={"time": times, "lat": [lat_val], "lon": [lon_val]},
                dims=("time", "lat", "lon"),
            )

    ds = xr.Dataset(data_vars)

    # Add metadata for roughness if present
    if "z0" in ds.variables:
        ds["z0"].attrs.update({
            "units": "m",
            "long_name": "forecast surface roughness (roughness length)",
            "standard_name": "surface_roughness_length",
        })

    # Compute relative humidity from temperature and dew point
    if {"t2m", "d2m"} <= set(ds.data_vars):
        T = ds["t2m"] - 273.15
        Td = ds["d2m"] - 273.15
        a, b = 17.625, 243.04
        es = 6.1094 * np.exp(a * T / (b + T))
        e = 6.1094 * np.exp(a * Td / (b + Td))
        rh = (e / es) * 100.0
        ds["RH2M"] = rh.clip(min=0, max=100)

    # Convert radiative fluxes from J m-2 to W m-2
    if "ssrd" in ds.data_vars:
        ds["ssrd"] = ds["ssrd"] / 3600.0
        ds["ssrd"].attrs.update({"units": "W m-2"})
    if "strd" in ds.data_vars:
        ds["strd"] = ds["strd"] / 3600.0
        ds["strd"].attrs.update({"units": "W m-2"})

    # Surface altitude from geopotential (z/g)
    if "z" in ds.data_vars:
        hsurf = ds["z"] / G0
        if "time" in hsurf.dims:
            hsurf = hsurf.isel(time=0, drop=True)
        hsurf.attrs.update({"long_name": "surface altitude", "units": "m"})
        ds["surface_altitude"] = hsurf

    # Drop intermediate dew point once RH is computed
    if "d2m" in ds.data_vars:
        ds = ds.drop_vars("d2m")
    if "z" in ds.data_vars:
        ds = ds.drop_vars("z")

    # Standardize names and keep the requested output variables
    ren_vars = {k: v for k, v in VAR_MAP.items() if k in ds.data_vars}
    ds = ds.rename(ren_vars)

    wanted = [v for v in WANTED_VARS if v in ds.data_vars]
    ds = ds[wanted]
    ds = add_uhi_indicator(ds)

    if "time" in ds.coords:
        ds["time"].encoding.update(units="hours since 1900-01-01 00:00:00", calendar="gregorian")

    ds.to_netcdf(str(out_tmp), format="NETCDF4", engine="netcdf4")
    os.replace(out_tmp, out_nc)
    tlog(f"Saved ERA5 single-point NetCDF → {out_nc}")



def main():
    ap = argparse.ArgumentParser(description="Build all static/forcing inputs for the SOLWEIG GPU pipeline.")
    ap.add_argument("--lat", type=float, default=DEFAULT_LAT)
    ap.add_argument("--lon", type=float, default=DEFAULT_LON)
    ap.add_argument("--city", type=str, default=None, help="Optional city name; if provided, skip reverse-geocoded name")
    ap.add_argument("--km-buffer", type=float, default=DEFAULT_KM_BUFFER, help="Half-size of initial bbox in km")
    ap.add_argument("--km-reduced-lat", type=float, default=DEFAULT_KM_REDUCED_LAT, help="Shrink (N/S) from bbox for SOLWEIG in km")
    ap.add_argument("--km-reduced-lon", type=float, default=DEFAULT_KM_REDUCED_LON, help="Shrink (E/W) from bbox for SOLWEIG in km")
    ap.add_argument("--year-start", type=int, default=DEFAULT_YEAR_START, help="Start year for meteorology (inclusive)")
    ap.add_argument("--year-end", type=int, default=DEFAULT_YEAR_END, help="End year for meteorology (inclusive)")
    ap.add_argument("--base-folder", type=str, default=DEFAULT_BASE, help="Workspace base folder")
    ap.add_argument("--resolution", type=float, default=DEFAULT_RES_M, help="Reference grid resolution (meters)")
    args = ap.parse_args()

    # Resolve workspace paths (per-city input and SOLWEIG output folders)
    base = Path(args.base_folder)
    ensure_dir(base)

    if args.city and args.city.strip():
        city = args.city.strip()
        tlog(f"Using provided city name: {city}")
        # Still derive region info for country-based logic
        _, state, cc = reverse_geocode(args.lat, args.lon)
    else:
        tlog("Reverse geocoding location name...")
        city, state, cc = reverse_geocode(args.lat, args.lon)
        tlog(f"Location: {city}")

    paths = build_paths(base, city)
    ensure_dir(paths.data_dir); ensure_dir(paths.out_dir)

    # Clean old TIFs in out_dir before rebuilding assets, but keep existing vector rasters
    keep_rasters = {paths.veg_ras.resolve(), paths.wat_ras.resolve(), paths.bld_ras.resolve()}
    for tif in paths.out_dir.glob("*.tif"):
        try:
            if tif.resolve() in keep_rasters and tif.stat().st_size > 0:
                continue
            tif.unlink()
        except Exception:
            pass

    # BBoxes
    tlog("Computing bounding boxes...")
    b = compute_bounding_boxes(
        args.lat, args.lon, args.km_buffer, args.km_reduced_lat, args.km_reduced_lon
    )

    # Earth Engine
    tlog("Initializing Earth Engine...")
    safe_initialize_ee()

    # Build vectors from OSM/GBA only if not already present
    vector_outputs = [paths.veg_fp, paths.wat_fp, paths.imprv_fp, paths.bld_fp]
    if all(fp.exists() and fp.stat().st_size > 0 for fp in vector_outputs):
        tlog("OSM/GBA vectors already exist; skipping fetch.")
    else:
        tlog("Fetching OSM/GBA vectors (missing outputs)...")
        build_vectors_from_osm_gba(paths, b.bbox4326)

    # Decide Italy via reverse geocode (prefer country code, fallback to region name)
    in_italy = (cc == "it") or (state in ITALY_REGIONS)
    tlog("Area is in Italy" if in_italy else "Area is not in Italy")

    # Downloads
    download_worldcover(paths.esa_tif, b.bbox4326)
    download_tree_dsm(paths.tree_tif, paths.tree_tiles, b.bbox4326)
    download_dem(paths.dem_tif, b.bbox_utm, b.crs_utm, b.bbox4326, in_italy)
    download_lcz(paths.lcz_tif, b.bbox4326)

    # Build the reference analysis grid (SOLWEIG domain) in UTM
    grid = compute_reference_grid(b.bbox_utm_solweig, args.resolution)

    # Resample to reference grid
    tlog("Resampling rasters to unified grid...")
    _resample_to_grid(paths.esa_tif,  paths.landuse_ras, b.crs_utm, grid.transform, grid.width, grid.height, Resampling.nearest)
    # Reclassify only the Landuse output
    reclassify_esa_worldcover_inplace(paths.landuse_ras)
    _resample_to_grid(paths.tree_tif, paths.tree_ras,    b.crs_utm, grid.transform, grid.width, grid.height, Resampling.bilinear)
    _resample_to_grid(paths.dem_tif,  paths.dem_ras,     b.crs_utm, grid.transform, grid.width, grid.height, Resampling.bilinear)
    _resample_to_grid(paths.lcz_tif,  paths.lcz_ras,     b.crs_utm, grid.transform, grid.width, grid.height, Resampling.nearest)

    # Vector → raster: vegetation, water, and buildings (with height attribute)
    vector_raster_outputs = [paths.veg_ras, paths.wat_ras, paths.bld_ras]
    if all(fp.exists() and fp.stat().st_size > 0 for fp in vector_raster_outputs):
        tlog("Rasterizing vectors skipped (raster outputs already present).")
    else:
        tlog("Rasterizing vectors...")
        rasterize_vector_checked(paths.veg_fp,  paths.veg_ras,  2,             grid.transform, grid.width, grid.height, b.crs_utm, "vegetation")
        rasterize_vector_checked(paths.wat_fp,  paths.wat_ras,  3,             grid.transform, grid.width, grid.height, b.crs_utm, "water")
        rasterize_vector_checked(paths.bld_fp,  paths.bld_ras,  "HEIGHT_ROOF", grid.transform, grid.width, grid.height, b.crs_utm, "buildings", dtype="float32")

    # Cleanup temporary tile directories (if they exist) to keep workspace tidy
    try:
        wc_tiles = paths.esa_tif.parent / "worldcover_tiles"
        if wc_tiles.exists() and wc_tiles.is_dir():
            shutil.rmtree(wc_tiles)
            tlog(f"Removed temporary WorldCover tiles: {wc_tiles}")
    except Exception as e:
        tlog(f"WARNING: could not remove WorldCover tiles: {e}")

    try:
        if paths.tree_tiles.exists() and paths.tree_tiles.is_dir():
            shutil.rmtree(paths.tree_tiles)
            tlog(f"Removed temporary tree DSM tiles: {paths.tree_tiles}")
    except Exception as e:
        tlog(f"WARNING: could not remove tree tiles: {e}")

    # Create in-memory urban mask (impervious ∪ buildings) without writing to disk
    imprv_gdf = _read_vector_or_warn(paths.imprv_fp, "impervious")
    bld_gdf   = _read_vector_or_warn(paths.bld_fp,   "building")
    urban_arr = None
    try:
        frames = []
        if imprv_gdf is not None and not imprv_gdf.empty:
            frames.append(imprv_gdf[["geometry"]])
        if bld_gdf is not None and not bld_gdf.empty:
            frames.append(bld_gdf[["geometry"]])
        if frames:
            urban_gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=imprv_gdf.crs if imprv_gdf is not None else bld_gdf.crs)
            if urban_gdf.crs.to_string() != b.crs_utm:
                urban_gdf = urban_gdf.to_crs(b.crs_utm)
            urban_arr = rasterize_polygons_to_array(urban_gdf, grid.transform, grid.width, grid.height, b.crs_utm, value=1, dtype="uint8")
    except Exception as e:
        tlog(f"WARNING: could not build in-memory urban mask: {e}")

    # Apply raster logic and create Building_DSM
    apply_raster_logic(paths.bld_ras, paths.tree_ras, paths.landuse_ras, paths.wat_ras, paths.veg_ras, paths.dem_ras, paths.dsm_plus_ras, urban_arr=urban_arr)

    # Meteorology: choose output prefix by source (cosmo for DDS over Italy, era5 otherwise)
    prefix = "cosmo" if in_italy else "era5"
    met_out = paths.out_dir / f"{prefix}_{city}_{args.year_start}_{args.year_end}.nc"
    download_and_embed_era5(met_out, b.bbox_osm, args.year_start, args.year_end, paths.data_dir, in_italy)

    # Compute wind coefficient AFTER meteorology, using ERA5 Z0 mean if available
    z0_ref_value = 0.03
    try:
        if met_out.exists():
            with xr.open_dataset(met_out) as dsm:
                if "Z0" in dsm.variables:
                    z0_candidate = float(dsm["Z0"].mean().values)
                    if np.isfinite(z0_candidate) and z0_candidate > 0:
                        z0_ref_value = z0_candidate
                        tlog(f"Using ERA5 Z0 mean as z0_ref: {z0_ref_value:.4f} m")
                    else:
                        tlog("Z0 present but invalid; using default z0_ref = 0.03 m")
                else:
                    tlog("Z0 not found in meteo file; using default z0_ref = 0.03 m")
    except Exception as e:
        tlog(f"WARNING: could not read Z0 from meteo file; using default z0_ref. Error: {e}")

    # Wind coefficient raster (now that z0_ref is known)
    try:
        compute_wind_coeff(paths, z0_ref=z0_ref_value)
    except Exception as e:
        tlog(f"WARNING: WindCoeff computation failed: {e}")

    # Alignment check (raises on mismatch)
    tlog("Checking alignment...")
    to_check = [paths.landuse_ras, paths.tree_ras, paths.dem_ras, paths.dsm_plus_ras, paths.lcz_ras]
    if paths.wind_coeff_ras.exists():
        to_check.append(paths.wind_coeff_ras)
    cb_fp = paths.out_dir / "WindCoeff_Urban.tif"
    ct_fp = paths.out_dir / "WindCoeff_Vegetation.tif"
    if cb_fp.exists():
        to_check.append(cb_fp)
    if ct_fp.exists():
        to_check.append(ct_fp)
    lf_b_fp = paths.out_dir / "LambdaF_Buildings.tif"
    if lf_b_fp.exists():
        to_check.append(lf_b_fp)
    check_raster_alignment(to_check)
    tlog("All rasters aligned (CRS, resolution, extents, dimensions).")

  
    # Final report
    final_outputs = [
        paths.landuse_ras, paths.tree_ras, paths.dem_ras, paths.dsm_plus_ras,
        paths.lcz_ras, paths.veg_ras, paths.wat_ras, paths.bld_ras,
        paths.wind_coeff_ras
    ]
    if lf_b_fp.exists():
        final_outputs.append(lf_b_fp)
    if cb_fp.exists():
        final_outputs.append(cb_fp)
    if ct_fp.exists():
        final_outputs.append(ct_fp)
    final_report_table(final_outputs)

    tlog("Completed successfully.")

if __name__ == "__main__":
    main()

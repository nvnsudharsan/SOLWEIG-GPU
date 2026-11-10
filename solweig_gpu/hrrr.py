"""
HRRR forcing for SOLWEIG-GPU.

Fetches the surface fields of NOAA's High-Resolution Rapid Refresh (HRRR, 3 km CONUS) from the public
AWS bucket ``noaa-hrrr-bdp-pds`` and writes the ERA5-style NetCDF pair that the ERA5 readers of this
package already understand::

    <out_dir>/data_stream-oper_stepType-instant.nc   t2m, d2m, sp, u10, v10 (+ fsr = HRRR SFCR roughness length)
    <out_dir>/data_stream-oper_stepType-accum.nc     ssrd = DSWRF x 3600 (J m-2 per hour)

on a regular latitude/longitude grid (default 0.03 deg) covering the model domain plus a buffer, with
``time`` = valid time (UTC). Only the required GRIB2 messages are downloaded, by HTTP byte ranges taken
from the ``.idx`` index files, so a 48 h window is a few tens of MB.

Use it through ``thermal_comfort(..., use_own_met=False, data_source_type='hrrr', data_folder=...)``,
which calls :func:`ensure_hrrr` (download when the folder does not already cover the window), or call
:func:`fetch_hrrr` directly. Optional dependencies: ``requests`` and ``eccodes`` (``pip install
solweig-gpu[hrrr]``).

HRRR is an operational forecast: the latest 6-hourly cycle (00/06/12/18 UTC) at or before the window
start that is already published is used, plus the cycle 24 h earlier so that the day before the window
is covered for the UHI cycle. Each 6-hourly cycle reaches 48 h ahead; windows beyond that horizon are
truncated with a warning. Where cycles overlap, the shortest lead time wins.
"""
import os
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BASE_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
INSTANT_FILE = "data_stream-oper_stepType-instant.nc"
ACCUM_FILE = "data_stream-oper_stepType-accum.nc"
# output name -> "VAR:level" key of the wrfsfc .idx records
FIELDS = {
    "t2m": "TMP:2 m above ground",
    "d2m": "DPT:2 m above ground",
    "sp": "PRES:surface",
    "u10": "UGRD:10 m above ground",
    "v10": "VGRD:10 m above ground",
    "dswrf": "DSWRF:surface",
    "fsr": "SFCR:surface",
}
CYCLE_HOURS = 6        # cycles with a 48 h horizon
CYCLE_HORIZON_H = 48


def _require_deps():
    try:
        import requests  # noqa: F401
        import eccodes   # noqa: F401
    except ImportError as e:   # pragma: no cover
        raise ImportError(
            "HRRR support needs the optional packages 'requests' and 'eccodes' "
            "(pip install solweig-gpu[hrrr], or conda install -c conda-forge eccodes python-eccodes)"
        ) from e


def parse_fhours(spec):
    """'0-47' or '0-5,12,18-20' -> sorted list of forecast hours."""
    out = []
    for part in str(spec).split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return sorted(set(out))


def parse_idx(text):
    """Parse a wrfsfc ``.idx`` file: list of (byte_start, byte_end_or_None, 'VAR:level')."""
    recs = [line.split(":") for line in text.splitlines() if line.strip()]
    out = []
    for i, r in enumerate(recs):
        start = int(r[1])
        end = int(recs[i + 1][1]) - 1 if i + 1 < len(recs) else None
        out.append((start, end, f"{r[3]}:{r[4]}"))
    return out


def file_url(cycle, fhour):
    """URL of the wrfsfc GRIB2 file of a cycle (datetime, UTC) and forecast hour."""
    return f"{BASE_URL}/hrrr.{cycle:%Y%m%d}/conus/hrrr.t{cycle:%H}z.wrfsfcf{fhour:02d}.grib2"


def cycle_available(cycle, fhour=CYCLE_HORIZON_H, timeout=30):
    """True when the file of this cycle and forecast hour is on AWS (the last file means the whole cycle is published)."""
    import urllib.request
    try:
        req = urllib.request.Request(file_url(cycle, fhour) + ".idx", method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def select_cycles(start_time, end_time, now=None, available=cycle_available, max_back=8):
    """
    Choose the HRRR cycles for a UTC window.

    Returns (cycles, fhours, horizon): the latest 6-hourly cycle at or before ``start_time`` that is
    published (walking back up to ``max_back`` cycles), preceded by the cycle 24 h earlier (padding for
    the UHI day before), the forecast hours '0-48', and the valid time the newest cycle reaches.
    ``available(cycle)`` is the publication test (replaceable for tests); ``now`` caps future windows.
    """
    start_time = _as_dt(start_time)
    end_time = _as_dt(end_time)
    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    c = start_time.replace(minute=0, second=0, microsecond=0)
    while c.hour % CYCLE_HOURS:
        c -= dt.timedelta(hours=1)
    while c > now:
        c -= dt.timedelta(hours=CYCLE_HOURS)
    for _ in range(max_back):
        if available(c):
            break
        print(f"[HRRR] cycle {c:%Y-%m-%dT%H}Z is not on AWS (yet); trying the previous one", flush=True)
        c -= dt.timedelta(hours=CYCLE_HOURS)
    else:
        raise RuntimeError(f"no complete HRRR cycle found on AWS back to {c:%Y-%m-%dT%H}Z (no internet?)")
    cycles = [c - dt.timedelta(hours=24), c]
    horizon = c + dt.timedelta(hours=CYCLE_HORIZON_H)
    if end_time > horizon:
        print(f"[HRRR] WARNING: cycle {c:%Y-%m-%dT%H}Z reaches {horizon:%Y-%m-%d %H}Z only; "
              f"the window to {end_time:%Y-%m-%d %H}Z is truncated", flush=True)
    return cycles, f"0-{CYCLE_HORIZON_H}", horizon


def _as_dt(t):
    if isinstance(t, dt.datetime):
        return t.replace(tzinfo=None) if t.tzinfo is None else t.astimezone(dt.timezone.utc).replace(tzinfo=None)
    t = str(t).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H", "%Y-%m-%dT%H", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(t, fmt)
        except ValueError:
            pass
    raise ValueError(f"cannot parse time {t!r}")


def bbox_from_raster(raster_path, buffer_deg=0.3):
    """(lonmin, lonmax, latmin, latmax) of a raster's extent in WGS84, padded by ``buffer_deg``."""
    import rasterio
    from rasterio.warp import transform as rio_transform
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"{raster_path} has no CRS")
        b = src.bounds
        xs = [b.left, b.right, b.right, b.left]
        ys = [b.bottom, b.bottom, b.top, b.top]
        if src.crs.to_epsg() == 4326:
            lons, lats = xs, ys
        else:
            lons, lats = rio_transform(src.crs, "EPSG:4326", xs, ys)
    return (min(lons) - buffer_deg, max(lons) + buffer_deg, min(lats) - buffer_deg, max(lats) + buffer_deg)


def target_grid(bbox, res_deg=0.03):
    """Regular lat/lon grid (lat descending like ERA5) covering the bbox."""
    lonmin, lonmax, latmin, latmax = bbox
    lons = np.arange(np.floor(lonmin / res_deg) * res_deg, lonmax + res_deg / 2, res_deg)
    lats = np.arange(np.ceil(latmax / res_deg) * res_deg, latmin - res_deg / 2, -res_deg)
    return lats.astype(np.float64), lons.astype(np.float64)


class _Fetcher:
    """Byte-range download and decoding of the wrfsfc messages, with the HRRR grid cached after the first message."""

    def __init__(self, lats, lons, workers=8, timeout=120, retries=3):
        import requests
        self.session = requests.Session()
        self.lats, self.lons = lats, lons
        self.LON, self.LAT = np.meshgrid(lons, lats)
        self.workers, self.timeout, self.retries = workers, timeout, retries
        self.grid = {}
        self.interp = {}

    def _get(self, url, **kw):
        last = None
        for _ in range(self.retries):
            try:
                r = self.session.get(url, timeout=self.timeout, **kw)
                r.raise_for_status()
                return r
            except Exception as e:   # transient S3/network errors
                last = e
        raise RuntimeError(f"HRRR download failed: {url} ({last})")

    def fetch(self, cycle, fhour, names=FIELDS):
        """{name: GRIB message bytes} for one file."""
        url = file_url(cycle, fhour)
        recs = parse_idx(self._get(url + ".idx").text)
        out = {}
        for name, key in names.items():
            for start, end, k in recs:
                if k == key:
                    rng = f"bytes={start}-{'' if end is None else end}"
                    out[name] = self._get(url, headers={"Range": rng}).content
                    break
            else:
                raise KeyError(f"{key} not in {url}.idx")
        return out

    def decode(self, msg):
        """GRIB message bytes -> 2-D values (Nj, Ni) on the HRRR grid (grid parameters cached)."""
        import eccodes as ec
        g = ec.codes_new_from_message(msg)
        try:
            ni, nj = ec.codes_get(g, "Ni"), ec.codes_get(g, "Nj")
            v = ec.codes_get_values(g).reshape(nj, ni)
            if not self.grid:
                self.grid.update(
                    lat=ec.codes_get_array(g, "latitudes").reshape(nj, ni),
                    lon=ec.codes_get_array(g, "longitudes").reshape(nj, ni),
                    lad=ec.codes_get(g, "LaDInDegrees"), lov=ec.codes_get(g, "LoVInDegrees"),
                    lat1=ec.codes_get(g, "Latin1InDegrees"), lat2=ec.codes_get(g, "Latin2InDegrees"),
                    R=ec.codes_get(g, "radius"))
            return v
        finally:
            ec.codes_release(g)

    def regrid(self, v):
        """The HRRR Lambert grid is regular in projected x/y: linear interpolation at the target lon/lat points."""
        from pyproj import CRS, Transformer
        from scipy.interpolate import RegularGridInterpolator
        if not self.interp:
            g = self.grid
            crs = CRS.from_proj4(f"+proj=lcc +lat_1={g['lat1']} +lat_2={g['lat2']} +lat_0={g['lad']} +lon_0={g['lov']} +R={g['R']} +units=m +no_defs")
            tr = Transformer.from_crs(4326, crs, always_xy=True)
            x, y = tr.transform(g["lon"], g["lat"])
            xs, ys = x[0, :], y[:, 0]
            if not (np.allclose(np.diff(xs), 3000, atol=5) and np.allclose(np.diff(ys), 3000, atol=5)):
                raise RuntimeError("HRRR grid is not regular in its Lambert projection")
            tx, ty = tr.transform(self.LON, self.LAT)
            if tx.min() < xs.min() or tx.max() > xs.max() or ty.min() < ys.min() or ty.max() > ys.max():
                raise ValueError("the requested bounding box is outside the HRRR CONUS grid")
            self.interp.update(xs=xs, ys=ys, pts=np.column_stack([ty.ravel(), tx.ravel()]))
        f = RegularGridInterpolator((self.interp["ys"], self.interp["xs"]), v, method="linear", bounds_error=True)
        return f(self.interp["pts"]).reshape(self.LAT.shape).astype(np.float32)


PAD_HOURS = 24   # hours kept before/after the window: the UHI reader (process_era5_data_uhi) pads the window itself


def fetch_hrrr(start_time, end_time, out_dir, bbox=None, raster=None, cycles=None, fhours=None,
               res_deg=0.03, buffer_deg=0.3, workers=8, sfcr_tif=None, pad_hours=PAD_HOURS):
    """
    Download HRRR surface fields and write the ERA5-style NetCDF pair into ``out_dir``.

    Parameters:
        start_time, end_time: UTC window (str "YYYY-MM-DD HH:MM:SS" or datetime). Pad the SOLWEIG
            day by at least 24 h on both sides when the UHI cycle is used (``thermal_comfort`` does).
        out_dir (str): Destination folder (created); becomes ``data_folder`` of the ERA5 readers.
        bbox (tuple): (lonmin, lonmax, latmin, latmax) in degrees, or
        raster (str): a raster of the model domain whose extent (+ ``buffer_deg``) defines the box.
        cycles (list of datetime): HRRR cycles to use; default :func:`select_cycles`.
        fhours (str): forecast hours to read from each cycle, e.g. '0-48' (default from select_cycles).
        res_deg (float): spacing of the output lat/lon grid (default 0.03 deg, about the 3 km of HRRR).
        workers (int): parallel downloads.
        pad_hours (int): hours kept before and after the window (default 24, what the UHI reader pads).
        sfcr_tif (str): optional GeoTIFF of the HRRR roughness length (EPSG:4326) for ``z0_ref_raster``.

    Returns:
        str: ``out_dir``.
    """
    _require_deps()
    import xarray as xr
    if bbox is None:
        if raster is None:
            raise ValueError("give bbox=(lonmin, lonmax, latmin, latmax) or raster=<path of a domain raster>")
        bbox = bbox_from_raster(raster, buffer_deg)
    start_time, end_time = _as_dt(start_time), _as_dt(end_time)
    if cycles is None:
        cycles, fh, _ = select_cycles(start_time, end_time)
        fhours = fhours or fh
    else:
        cycles = [_as_dt(c) for c in cycles]
        fhours = fhours or f"0-{CYCLE_HORIZON_H}"
    fhs = parse_fhours(fhours)
    lats, lons = target_grid(bbox, res_deg)
    print(f"[HRRR] grid {len(lats)}x{len(lons)} at {res_deg} deg, lon {lons[0]:.2f}..{lons[-1]:.2f}, lat {lats[-1]:.2f}..{lats[0]:.2f}; "
          f"cycles {', '.join(c.strftime('%Y-%m-%dT%H') + 'Z' for c in cycles)}, forecast hours {fhours}", flush=True)
    F = _Fetcher(lats, lons, workers=workers)
    lo, hi = start_time - dt.timedelta(hours=pad_hours), end_time + dt.timedelta(hours=pad_hours)
    jobs = [(c, fh) for c in cycles for fh in fhs if lo <= c + dt.timedelta(hours=fh) <= hi]
    if not jobs:
        raise ValueError("no HRRR forecast hours fall inside the requested window")
    data = {}   # valid time -> (lead, {name: 2-D})
    with ThreadPoolExecutor(workers) as ex:
        for (c, fh), msgs in zip(jobs, ex.map(lambda j: F.fetch(j[0], j[1]), jobs)):
            vt = c + dt.timedelta(hours=fh)
            if vt in data and data[vt][0] <= fh:
                continue
            data[vt] = (fh, {k: F.regrid(F.decode(m)) for k, m in msgs.items()})
    times = sorted(data)
    os.makedirs(out_dir, exist_ok=True)

    def stack(k):
        return np.stack([data[t][1][k] for t in times])

    coords = dict(time=np.array(times, dtype="datetime64[ns]"), latitude=lats, longitude=lons)
    inst = xr.Dataset({k: (("time", "latitude", "longitude"), stack(k)) for k in ("t2m", "d2m", "sp", "u10", "v10", "fsr")}, coords=coords)
    for k, u in (("t2m", "K"), ("d2m", "K"), ("sp", "Pa"), ("u10", "m s**-1"), ("v10", "m s**-1"), ("fsr", "m")):
        inst[k].attrs["units"] = u
    inst["fsr"].attrs["long_name"] = "Surface roughness length (HRRR SFCR), same name as ERA5 forecast surface roughness"
    acc = xr.Dataset({"ssrd": (("time", "latitude", "longitude"), stack("dswrf") * 3600.0)}, coords=coords)
    acc["ssrd"].attrs.update(units="J m**-2", long_name="Surface short-wave (solar) radiation downwards (HRRR DSWRF x 3600 s)")
    meta = dict(source="NOAA HRRR wrfsfc via noaa-hrrr-bdp-pds, linear regridding from the 3 km Lambert grid to a regular lat/lon grid",
                cycles=", ".join(c.strftime("%Y-%m-%dT%H") for c in cycles), forecast_hours=fhours, bbox=str(bbox))
    inst.attrs.update(meta); acc.attrs.update(meta)
    inst.to_netcdf(os.path.join(out_dir, INSTANT_FILE))
    acc.to_netcdf(os.path.join(out_dir, ACCUM_FILE))
    leads = [data[t][0] for t in times]
    print(f"[HRRR] wrote {out_dir}: {len(times)} hours {times[0]:%Y-%m-%d %H}Z .. {times[-1]:%Y-%m-%d %H}Z, lead times {min(leads)}..{max(leads)} h", flush=True)
    if sfcr_tif:
        import rasterio
        from rasterio.transform import from_origin
        z0 = data[times[0]][1]["fsr"]
        with rasterio.open(sfcr_tif, "w", driver="GTiff", width=len(lons), height=len(lats), count=1, dtype="float32", crs="EPSG:4326",
                           transform=from_origin(lons[0] - res_deg / 2, lats[0] + res_deg / 2, res_deg, res_deg), compress="deflate") as d:
            d.write(z0, 1)
    return out_dir


def covers(data_folder, start_time, end_time, pad_hours=PAD_HOURS):
    """True when ``data_folder`` holds the NetCDF pair and its times span the window padded by ``pad_hours`` before
    (the UHI reader needs the day before) and reach the window end."""
    inst = os.path.join(data_folder, INSTANT_FILE)
    if not (os.path.isfile(inst) and os.path.isfile(os.path.join(data_folder, ACCUM_FILE))):
        return False
    import xarray as xr
    with xr.open_dataset(inst) as ds:
        tname = "valid_time" if "valid_time" in ds.coords else "time"
        t = ds[tname].values.astype("datetime64[s]")
    s = np.datetime64(_as_dt(start_time) - dt.timedelta(hours=pad_hours))
    e = np.datetime64(_as_dt(end_time))
    return bool(t.min() <= s and t.max() >= e - np.timedelta64(1, "h"))


def ensure_hrrr(data_folder, start_time, end_time, raster=None, bbox=None, **kwargs):
    """
    Make sure ``data_folder`` holds HRRR forcing for the window; download it when missing or too short.

    This is what ``thermal_comfort``/``preprocess`` call for ``data_source_type='hrrr'``. Returns the folder.
    """
    if covers(data_folder, start_time, end_time):
        print(f"[HRRR] using the existing files in {data_folder}", flush=True)
        return data_folder
    return fetch_hrrr(start_time, end_time, data_folder, bbox=bbox, raster=raster, **kwargs)

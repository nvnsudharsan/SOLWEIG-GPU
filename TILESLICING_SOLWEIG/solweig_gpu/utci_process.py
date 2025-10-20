def _wait_day_ready(date_str: str, needed_vars, output_path: str, timeout_s: int = 1800, poll_ms: int = 200) -> bool:
    """
    Block until the writer has created the .ready markers for all required variables
    of the given day (intermediate files completed), or until timeout.

    GPUs block at the end of each day until the writer emits all .ready files (strict GPU→CPU synchronization).
    """
    t0 = time.time()
    missing = set(needed_vars)
    while True:
        done = set()
        for v in list(missing):
            # Prefer hidden marker; also accept legacy marker for backward compatibility
            p_hidden = os.path.join(output_path, f".{v}_{date_str}.tif.ready")
            p_legacy = os.path.join(output_path, f"{v}_{date_str}.tif.ready")
            if os.path.exists(p_hidden) or os.path.exists(p_legacy):
                done.add(v)
        missing -= done
        if not missing:
            return True
        if (time.time() - t0) > timeout_s:
            try:
                print(f"[SYNC][WARN] timeout waiting writer ready markers for {date_str}. Missing: {sorted(missing)}", flush=True)
            except Exception:
                pass
            return False
        time.sleep(max(1, poll_ms) / 1000.0)
from osgeo import gdal, osr
import numpy as np
import os
import time
import datetime
import pytz
import torch
import gc
import atexit
import ctypes

from .shadow import svf_calculator, create_patches
from .solweig import Solweig_2022a_calc, clearnessindex_2013b
from .sun_position import Solweig_2015a_metdata_noload
from .calculate_utci import utci_calculator
from .writers import TiffWriter

# Cache for expensive per-tile SVF tensors across per-day calls in multi-tile scheduling
_SVF_CACHE: dict = {}


def _safe_minmax(x):
    """Return (nanmin, nanmax) as floats without raising exceptions.
    Works for torch tensors and array-like inputs. Returns NaN if empty.
    """
    if torch.is_tensor(x):
        n = int(x.numel())
        if n == 0:
            return float("nan"), float("nan")
        # Older torch may not have nanmin/nanmax; filter finite values explicitly
        try:
            mask = torch.isfinite(x)
        except Exception:
            # Fallback: treat NaN as not finite
            mask = ~torch.isnan(x)
        if not bool(mask.any().item() if mask.numel() else False):
            return float("nan"), float("nan")
        xm = x[mask]
        return float(torch.min(xm).item()), float(torch.max(xm).item())
    arr = np.asarray(x)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmin(arr)), float(np.nanmax(arr))


def compute_utci(
        writer_q,
        **args):
    
    
    transVeg: float = 0.03
    albedo_b: float = 0.2
    albedo_g: float = 0.15
    ewall: float = 0.93
    eground: float = 0.95
    absK: float = 0.7
    absL: float = 0.95
    Fside: float = 0.22
    Fup: float = 0.06
    Fcyl: float = 0.28
    cyl: bool = True
    elvis: int = 0
    usevegdem: int = 1
    onlyglobal: int = 1
    landcover: int = 1
    firstdayleaf: int = 97
    lastdayleaf: int = 300
    conifer_bool: bool = False
    # Keep requested defaults for sky model
    anisotropic_sky: int = 1
    patch_option: int = 2


    # Extract gpu_id from kwargs for logging/compatibility
    gpu_id = int(args.get('gpu_id', 0))
    device_type = str(args.get('device_type', 'gpu')).lower()
    device_prefix = "GPU" if (device_type == 'gpu' and torch.cuda.is_available()) else "CPU"
    device_tag = f"{device_prefix}{gpu_id}"
    device_bracket = f"{device_prefix} {gpu_id}"
    use_cuda = (device_prefix == "GPU") and torch.cuda.is_available()


    """Compute daily UTCI (and optional fields) for one tile and write .dat buffers.
    At each completed day, create a ready marker and trigger a mosaic-from-dat step.
    All messages and comments are in English.
    Also writes Ta (2 m air temperature) and Va10m (10 m wind speed) daily GeoTIFF stacks alongside UTCI/TMRT.
    """
    # --- Map task args (kwargs) to local variables expected by the original implementation ---
    try:
        building_in         = args['DSM']
        tree_in             = args['Trees']
        landuse_in          = args['Landuse']
        dem_in              = args['DEM']
        walls_in            = args['Walls']
        aspect_in           = args['Aspect']
        windcoeff_in        = args.get('Windcoeff')
        hsurf_minus_dem_in  = args['HGTDEM']
        met_file            = args.get('met_data', None)
        met_path            = args.get('met_path', None)
        output_path         = args['output_path']
        number              = args['number']
        start_date          = args['start_time']
        ds_gt               = args['gt']
        ds_wkt              = args['wkt']
        full_gt             = args['full_gt']
        full_wkt            = args['full_wkt']
        full_rows           = args['full_rows']
        full_cols           = args['full_cols']
        num_tiles           = args['num_tiles']
        tile_coords         = args.get('tile_coords', None)
        base_path           = args.get('input_data_path', None)
        save_tmrt           = bool(args.get('save_tmrt', False))
        save_svf            = bool(args.get('save_svf', False))
        save_kup            = bool(args.get('save_kup', False))
        save_kdown          = bool(args.get('save_kdown', False))
        save_lup            = bool(args.get('save_lup', False))
        save_ldown          = bool(args.get('save_ldown', False))
        save_shadow         = bool(args.get('save_shadow', False))
        log_every           = int(args.get('log_every', 1))
        use_amp             = bool(args.get('use_amp', True))
        # Compression level is handled post-run; ignore any zstd_level in args
        expected_tiles_by_date = args.get('expected_tiles_by_date')
        # When running per-day round-robin (multi-tile/GPU), workers set this False and handle sync externally
        internal_sync       = bool(args.get('internal_sync', True))
        cache_evict         = bool(args.get('cache_evict', False))
        use_windcoeff       = bool(args.get('use_windcoeff', True))
        use_uhi_cycle       = bool(args.get('use_uhi_cycle', True))
    except KeyError as e:
        raise KeyError(f"Missing required task argument: {e}")

    torch.set_grad_enabled(False)
    device = torch.device('cuda' if use_cuda else 'cpu')
    use_amp = (device.type == 'cuda' and use_amp)
    log_every = max(1, int(log_every))
    # Prefer bfloat16 if supported (more stable); otherwise float16
    if use_amp and use_cuda:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float16
    # Short sleep helps OS scheduler and reduces frequent page faults under load
    try:
        time.sleep(0.01)
    except Exception:
        pass
    # --- Memory cleanup knobs ---
    # More frequent cleanup by default to keep memory low
    clear_every = max(1, int(os.environ.get('SOLWEIG_CUDA_CLEAR_EVERY', '2')))  # hours
    # Allow env to override sky anisotropy and patch discretization for memory control
    try:
        anisotropic_sky = int(os.environ.get('SOLWEIG_ANISOTROPIC_SKY', str(anisotropic_sky)))
    except Exception:
        pass
    try:
        patch_option = int(os.environ.get('SOLWEIG_PATCH_OPTION', str(patch_option)))
    except Exception:
        pass
    # Keep met arrays on CPU by default and pull scalars per hour
    met_on_cpu = (os.environ.get('SOLWEIG_MET_ON_CPU', '1') != '0')

    def _mem_cleanup():
        """Aggressive memory cleanup on CPU/GPU to avoid growth and freezes."""
        try:
            gc.collect()
        except Exception:
            pass
        if use_cuda:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        _trim_os_heap()

    def _trim_os_heap():
        """Return heap pages to the OS (glibc) when possible."""
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    # On process exit, always attempt to trim the heap
    atexit.register(_trim_os_heap)
    windcoeff = None

    # Mandatory rasters; optional handled separately
    if isinstance(building_in, str) and tile_coords is not None:
        import numpy as _np
        from osgeo import gdal as _gdal
        r0, r1, c0, c1 = map(int, tile_coords)
        w, h = int(c1 - c0), int(r1 - r0)
        def _read_tile(path):
            ds = _gdal.Open(path, _gdal.GA_ReadOnly)
            if ds is None:
                raise RuntimeError(f"Cannot open raster: {path}")
            arr = ds.GetRasterBand(1).ReadAsArray(int(c0), int(r0), int(w), int(h)).astype(_np.float32)
            return arr
        building_np  = _read_tile(building_in)
        landuse_np   = _read_tile(landuse_in)
        tree_np      = _read_tile(tree_in)
        dem_np       = _read_tile(dem_in)
        walls_np     = _read_tile(walls_in)
        aspect_np    = _read_tile(aspect_in)
        windcoeff_np = None
        if isinstance(windcoeff_in, str) and use_windcoeff:
            try:
                windcoeff_np = _read_tile(windcoeff_in)
            except RuntimeError as exc:
                print(f"[WARN] {exc}; ignoring wind coefficient raster")
        try:
            hsdiff_np = _read_tile(hsurf_minus_dem_in)
        except RuntimeError:
            hsdiff_np = _np.zeros_like(building_np, dtype=_np.float32)
        building     = torch.from_numpy(building_np).to(device)
        landuse      = torch.from_numpy(landuse_np).to(device)
        tree         = torch.from_numpy(tree_np).to(device)
        dem          = torch.from_numpy(dem_np).to(device)
        walls        = torch.from_numpy(walls_np).to(device)
        dirwalls     = torch.from_numpy(aspect_np).to(device)
        if windcoeff_np is not None:
            windcoeff = torch.from_numpy(windcoeff_np).to(device=device, dtype=building.dtype)
        hsdiff       = torch.from_numpy(hsdiff_np).to(device)
        try:
            del building_np, landuse_np, tree_np, dem_np, walls_np, aspect_np, windcoeff_np, hsdiff_np
        except Exception:
            pass
    else:
        building    = torch.from_numpy(building_in).to(device)
        landuse     = torch.from_numpy(landuse_in).to(device)
        tree        = torch.from_numpy(tree_in).to(device)
        dem         = torch.from_numpy(dem_in).to(device)
        walls       = torch.from_numpy(walls_in).to(device)
        dirwalls    = torch.from_numpy(aspect_in).to(device)
        if (windcoeff_in is not None) and use_windcoeff:
            windcoeff = torch.from_numpy(windcoeff_in).to(device=device, dtype=building.dtype)
        hsdiff = torch.from_numpy(hsurf_minus_dem_in).to(device) if hsurf_minus_dem_in is not None else None

    # Optional: downcast large static rasters to save memory on GPU
    # Control via env SOLWEIG_STATIC_DTYPE = fp32|fp16|bf16 (default fp32)
    static_dtype_env = os.environ.get('SOLWEIG_STATIC_DTYPE', 'fp32').lower().strip()
    if device.type == 'cuda' and static_dtype_env in ('fp16', 'half', 'bf16', 'bfloat16'):
        if static_dtype_env in ('bf16', 'bfloat16') and torch.cuda.is_bf16_supported():
            static_dtype = torch.bfloat16
        else:
            static_dtype = torch.float16
        try:
            building = building.to(static_dtype)
            landuse  = landuse.to(static_dtype)
            tree     = tree.to(static_dtype)
            dem      = dem.to(static_dtype)
            walls    = walls.to(static_dtype)
            dirwalls = dirwalls.to(static_dtype)
            if hsdiff is not None:
                hsdiff = hsdiff.to(static_dtype)
            if windcoeff is not None:
                windcoeff = windcoeff.to(static_dtype)
        except Exception:
            pass
    # Free original CPU numpy tiles to reduce RAM (we keep only torch tensors)
    try:
        del building_in, tree_in, landuse_in, dem_in, walls_in, aspect_in, windcoeff_in, hsurf_minus_dem_in
    except Exception:
        pass
    # Optional rasters fall back to neutral values
    if windcoeff is None:
        if use_windcoeff:
            windcoeff = torch.ones_like(building)
            if not getattr(compute_utci, "_warned_missing_windcoeff", False):
                print("[WARN] WindCoeff missing; using ones", flush=True)
                compute_utci._warned_missing_windcoeff = True
        else:
            if not getattr(compute_utci, "_info_wcoeff_disabled", False):
                print("[INFO] WindCoeff disabled by switch; not applying wind scaling", flush=True)
                compute_utci._info_wcoeff_disabled = True
    if hsdiff is None:
        hsdiff = torch.zeros_like(building)
        if not getattr(compute_utci, "_warned_missing_hsdiff", False):
            print("[WARN] HGTDEM diff missing; using zeros")
            compute_utci._warned_missing_hsdiff = True
    base_date = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    # Optional day-wise scheduling: process only a single day index (0-based)
    day_index = int(args.get('day_index', -1)) if args.get('day_index', None) is not None else None
    rows, cols = building.shape
    if windcoeff is not None and windcoeff.shape != (rows, cols):
        print(
            f"[WARN] WindCoeff raster shape {tuple(windcoeff.shape)} does not match tile {(rows, cols)}; ignoring wind coefficients"
        )
        windcoeff = None
    try:
        parts = number.split('_')
        if len(parts) == 4:
            r0, r1, c0, c1 = map(int, parts)
        else:
            # backward compatibility: only rows provided
            r0, r1 = map(int, parts[:2])
            c0, c1 = 0, cols
        tile_rows = int(r1) - int(r0)
        tile_cols = int(c1) - int(c0)
    except Exception:
        r0, r1, c0, c1 = 0, rows, 0, cols
        tile_rows = rows
        tile_cols = cols
    # Compression level is set globally via environment (see top of file)
    geotransform = ds_gt
    scale = 1 / geotransform[1]
    projection_wkt = ds_wkt
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(projection_wkt)
    new_cs = osr.SpatialReference()
    new_cs.ImportFromEPSG(4326)
    transform = osr.CoordinateTransformation(old_cs, new_cs)
    minx = geotransform[0]
    miny = geotransform[3] + cols * geotransform[4] + rows * geotransform[5]
    lonlat = transform.TransformPoint(minx, miny)
    gdalver = float(gdal.__version__[0])
    if gdalver == 3.:
        lon = lonlat[1]
        lat = lonlat[0]
    else:
        lon = lonlat[0]
        lat = lonlat[1]
    alt = torch.median(dem)
    alt = alt.cpu().item()
    if alt > 0:
        alt = 3.
    location = {'longitude': lon, 'latitude': lat, 'altitude': alt}
    timezone_name = "UTC"
    utc = 0.0
    # print(f"TZ {timezone_name} UTC{utc:+.1f} (forced)")

    # DEBUG flag removed per instructions

    def _qput(msg, timeout=5.0):
        try:
            writer_q.put(msg, timeout=timeout)
        except Exception:
            writer_q.put(msg)
    # --- BEGIN AUTOCast block ---
    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            # Load met either from provided array or from file path
            met_columns = args.get('met_columns')
            if met_columns is None and met_path is not None:
                try:
                    with open(met_path, 'r', encoding='utf-8') as f:
                        header_line = f.readline()
                    tokens = [tok for tok in header_line.strip().split() if tok]
                    met_columns = tokens if tokens else None
                except Exception:
                    met_columns = None

            if met_file is None and met_path is not None:
                met_file = np.loadtxt(met_path, skiprows=1, delimiter=' ')
            if met_file.ndim == 1:
                met_file = met_file[np.newaxis, :]
            # Slice to a single day if requested
            if day_index is not None and day_index >= 0:
                start_row = int(day_index * 24)
                end_row = start_row + 24
                if start_row < met_file.shape[0]:
                    met_file = met_file[start_row:min(end_row, met_file.shape[0]), :]
                else:
                    # No data for this day index; nothing to do
                    return
                # Shift base_date to the requested day
                base_date = base_date + datetime.timedelta(days=day_index)
            if met_file.ndim == 1:
                met_file = met_file[np.newaxis, :]

            col_map = {}
            if met_columns is not None and len(met_columns) == met_file.shape[1]:
                col_map = {name.strip().lower(): idx for idx, name in enumerate(met_columns)}

            def _col(name: str, fallback: int, *aliases: str) -> int:
                names = (name.lower(),) + tuple(alias.lower() for alias in aliases)
                for key in names:
                    if key in col_map:
                        return int(col_map[key])
                return fallback

            COL_WIND = _col('wind', 4)
            COL_RH = _col('rh', 5, 'relhum', 'relative_humidity')
            COL_TA = _col('td', 6, 'ta', 'tair', 'temp', 'temperature')
            COL_PRESS = _col('press', 7, 'pressure', 'psfc')
            COL_KDN = _col('kdn', 8, 'swdown')
            COL_KDIFF = _col('kdiff', 9)
            COL_KDIR = _col('kdir', 10)

            uhi_col_idx = None
            if col_map:
                uhi_col_idx = col_map.get('uhi_cycle') or col_map.get('uhi')
            if uhi_col_idx is None and met_file.shape[1] > 11:
                uhi_col_idx = 11

            if use_uhi_cycle and (uhi_col_idx is not None) and (0 <= int(uhi_col_idx) < met_file.shape[1]):
                uhi_series = met_file[:, int(uhi_col_idx)]
            else:
                uhi_series = np.zeros((met_file.shape[0],), dtype=met_file.dtype)
                if use_uhi_cycle:
                    if not getattr(compute_utci, "_warned_missing_uhi", False):
                        print("[WARN] UHI_CYCLE column missing in metfile; assuming 0.0", flush=True)
                        compute_utci._warned_missing_uhi = True
                else:
                    if not getattr(compute_utci, "_info_uhi_disabled", False):
                        print("[INFO] UHI_CYCLE disabled by switch; using zeros", flush=True)
                        compute_utci._info_uhi_disabled = True
            YYYY, altitude, azimuth, zen, jday, leafon, dectime, altmax = Solweig_2015a_metdata_noload(met_file, location, utc)

            tree[tree < 0.] = 0.
            vegdem = tree + dem
            vegdem2 = torch.add(tree * 0.25, dem)
            bush = torch.logical_not(vegdem2 * vegdem) * vegdem
            vegdsm = tree + building
            vegdsm[vegdsm == building] = 0
            vegdsm2 = tree * 0.25 + building
            vegdsm2[vegdsm2 == building] = 0
            amaxvalue = torch.maximum(building.max(), vegdem.max())
            buildings = building - dem
            buildings[buildings < 2.] = 1.
            buildings[buildings >= 2.] = 0
            valid_mask = (buildings == 1)
            _dtype = building.dtype
            Knight = torch.zeros((rows, cols), device=device, dtype=_dtype)
            Tgmap1 = torch.zeros((rows, cols), device=device, dtype=_dtype)
            Tgmap1E = torch.zeros((rows, cols), device=device, dtype=_dtype)
            Tgmap1S = torch.zeros((rows, cols), device=device, dtype=_dtype)
            Tgmap1W = torch.zeros((rows, cols), device=device, dtype=_dtype)
            Tgmap1N = torch.zeros((rows, cols), device=device, dtype=_dtype)
            TgOut1 = torch.zeros((rows, cols), device=device, dtype=_dtype)
            TgK_wall = 0.37
            Tstart_wall = -3.41
            TmaxLST_wall = 15.
            valid_mask_np = valid_mask.detach().cpu().numpy()
            TgK        = torch.zeros_like(Knight)
            Tstart     = torch.zeros_like(Knight)
            alb        = torch.zeros_like(Knight)
            emis       = torch.zeros_like(Knight)
            TmaxLST    = torch.zeros_like(Knight)
            
            mask1 = landuse == 1
            TgK[mask1]          = Knight[mask1] + 0.58
            Tstart[mask1]       = Knight[mask1] - 9.78
            alb[mask1]     = Knight[mask1] + 0.18
            emis[mask1]    = Knight[mask1] + 0.95
            TmaxLST[mask1] = Knight[mask1] + 15.0
            mask2 = landuse == 2
            TgK[mask2]       = Knight[mask2] + 0.21
            Tstart[mask2]       = Knight[mask2] - 3.38
            alb[mask2]     = Knight[mask2] + 0.16
            emis[mask2]    = Knight[mask2] + 0.94
            TmaxLST[mask2] = Knight[mask2] + 14.0
            mask3 = landuse == 3
            TgK[mask3]       = Knight[mask3] + 0.00
            Tstart[mask3]       = Knight[mask3] + 0.00
            alb[mask3]     = Knight[mask3] + 0.05
            emis[mask3]    = Knight[mask3] + 0.98
            TmaxLST[mask3] = Knight[mask3] + 12.0
            mask4 = landuse == 4
            TgK[mask4]       = Knight[mask4] + 0.33
            Tstart[mask4]       = Knight[mask4] -3.01
            alb[mask4]     = Knight[mask4] + 0.25
            emis[mask4]    = Knight[mask4] + 0.94
            TmaxLST[mask4] = Knight[mask4] + 14.0
            if met_on_cpu:
                DOY = torch.tensor(met_file[:, 1], device=device)
                Ta_cpu = met_file[:, COL_TA]
                RH_cpu = met_file[:, COL_RH]
                radG_cpu = met_file[:, COL_KDN]
                radD_cpu = met_file[:, COL_KDIFF]
                radI_cpu = met_file[:, COL_KDIR]
                P_cpu = met_file[:, COL_PRESS]
                Ws_cpu = met_file[:, COL_WIND]
            else:
                DOY = torch.tensor(met_file[:, 1], device=device, dtype=torch.float32)
                Ta = torch.tensor(met_file[:, COL_TA], device=device, dtype=torch.float32)
                RH = torch.tensor(met_file[:, COL_RH], device=device, dtype=torch.float32)
                radG = torch.tensor(met_file[:, COL_KDN], device=device, dtype=torch.float32)
                radD = torch.tensor(met_file[:, COL_KDIFF], device=device, dtype=torch.float32)
                radI = torch.tensor(met_file[:, COL_KDIR], device=device, dtype=torch.float32)
                P = torch.tensor(met_file[:, COL_PRESS], device=device, dtype=torch.float32)
                Ws = torch.tensor(met_file[:, COL_WIND], device=device, dtype=torch.float32)
            if conifer_bool:
                leafon = torch.ones((1, DOY.shape[0]), device=device)
            else:
                leafon = torch.zeros((1, DOY.shape[0]), device=device)
                if firstdayleaf > lastdayleaf:
                    leaf_bool = ((DOY > firstdayleaf) | (DOY < lastdayleaf))
                else:
                    leaf_bool = ((DOY > firstdayleaf) & (DOY < lastdayleaf))
                leafon[0, leaf_bool] = 1
            psi = leafon * transVeg
            psi[leafon == 0] = 0.5
            Twater = []
            height = 1.1
            height = torch.tensor(height, device=device)
            first = torch.round(height.clone().detach().to(device))
            if first == 0.:
                first = torch.tensor(1., device=device)
            second = torch.round(height * 20.)
            if (len(Ta_cpu) == 1) if met_on_cpu else (len(Ta) == 1):
                timestepdec = 0
            else:
                timestepdec = dectime[1] - dectime[0]
            timeadd = 0.
            firstdaytime = 1.
            # SVF cache key per tile
            tile_key = str(number)
            cache = _SVF_CACHE.get(tile_key) if (day_index is not None and day_index >= 0) else None
            if cache is not None and cache.get('shape') == (int(rows), int(cols)):
                svf = cache['svf']; svfaveg = cache['svfaveg']
                svfE = cache['svfE']; svfEaveg = cache['svfEaveg']; svfEveg = cache['svfEveg']
                svfN = cache['svfN']; svfNaveg = cache['svfNaveg']; svfNveg = cache['svfNveg']
                svfS = cache['svfS']; svfSaveg = cache['svfSaveg']; svfSveg = cache['svfSveg']
                svfveg = cache['svfveg']
                svfW = cache['svfW']; svfWaveg = cache['svfWaveg']; svfWveg = cache['svfWveg']
                vegshmat = cache['vegshmat']; vbshvegshmat = cache['vbshvegshmat']; shmat = cache['shmat']
                svftotal = cache['svftotal']
            else:
                svf, svfaveg, svfE, svfEaveg, svfEveg, svfN, svfNaveg, svfNveg, svfS, svfSaveg, svfSveg, svfveg, svfW, svfWaveg, svfWveg, vegshmat, vbshvegshmat, shmat, svftotal = svf_calculator(patch_option, amaxvalue, building, vegdsm, vegdsm2, bush, scale)
                if save_svf:
                    _qput(('open', 'SVF', 'static', 1, int(full_rows), int(full_cols), full_gt, full_wkt, output_path, num_tiles))
                    y0 = int(r0)
                    x0 = int(c0)
                    _qput(('write', 'SVF', 'static', 1, x0, y0, svftotal.detach().cpu().numpy().astype('float32')))
                    _qput(('tile_done', 'SVF', 'static'))
                if day_index is not None and day_index >= 0:
                    _SVF_CACHE[tile_key] = {
                        'shape': (int(rows), int(cols)),
                        'svf': svf, 'svfaveg': svfaveg,
                        'svfE': svfE, 'svfEaveg': svfEaveg, 'svfEveg': svfEveg,
                        'svfN': svfN, 'svfNaveg': svfNaveg, 'svfNveg': svfNveg,
                        'svfS': svfS, 'svfSaveg': svfSaveg, 'svfSveg': svfSveg,
                        'svfveg': svfveg,
                        'svfW': svfW, 'svfWaveg': svfWaveg, 'svfWveg': svfWveg,
                        'vegshmat': vegshmat, 'vbshvegshmat': vbshvegshmat, 'shmat': shmat,
                        'svftotal': svftotal,
                    }
            # Hint to GC: the SVF static frame was enqueued; keep tensors needed downstream, drop temporary CPU copy
            svfbuveg = svf - (1.0 - svfveg) * (1.0 - transVeg)
            asvf = torch.acos(torch.sqrt(svf))
            # Do not allocate full diffsh (3D); compute on the fly downstream
            tmp = svf + svfveg - 1.0
            tmp[tmp < 0.0] = 0.0
            svfalfa = torch.asin(torch.exp(torch.log(1.0 - tmp) / 2.0))
            fveg = (tree > 0).to(building.dtype)
            value_shape = (rows, cols)
            scalar_dtype = building.dtype
            ws_scalar = torch.empty((), device=device, dtype=scalar_dtype)
            ta_scalar = torch.empty((), device=device, dtype=scalar_dtype)
            rh_scalar = torch.empty((), device=device, dtype=scalar_dtype)
            p_scalar = torch.empty((), device=device, dtype=scalar_dtype)
            radg_scalar = torch.empty((), device=device, dtype=scalar_dtype)
            radd_scalar = torch.empty((), device=device, dtype=scalar_dtype)
            radi_scalar = torch.empty((), device=device, dtype=scalar_dtype)
            uhi_scalar = torch.empty((), device=device, dtype=scalar_dtype)
            va10m_mat = torch.empty(value_shape, dtype=building.dtype, device=device)
            va10m_safe_buf = torch.empty_like(va10m_mat)
            current_date_str = None
            hour_counter = 0
            # Pre-allocate output buffers once and reuse to avoid per-hour allocations
            out_utci = np.full((rows, cols), -9999.0, dtype=np.float32)
            out_tmrt = np.full((rows, cols), -9999.0, dtype=np.float32) if save_tmrt else None
            out_ta = np.full((rows, cols), -9999.0, dtype=np.float32)
            out_va = np.full((rows, cols), -9999.0, dtype=np.float32)
            
            def _open_day(current_date_str):
                def _open_var(var):
                    exp_tiles = int(num_tiles)
                    try:
                        if isinstance(expected_tiles_by_date, dict):
                            exp_tiles = int(expected_tiles_by_date.get(current_date_str, exp_tiles))
                    except Exception:
                        exp_tiles = int(num_tiles)
                    _qput(('open', var, current_date_str, 24, int(full_rows), int(full_cols),
                                  full_gt, full_wkt, output_path, exp_tiles))
                # Always open UTCI and meteorological state fields
                _open_var('UTCI')
                _open_var('Ta')
                _open_var('Va10m')
                # Optional radiation/diagnostics
                if save_tmrt:  _open_var('TMRT')
                if save_kup:   _open_var('Kup')
                if save_kdown: _open_var('Kdown')
                if save_lup:   _open_var('Lup')
                if save_ldown: _open_var('Ldown')
                if save_shadow:_open_var('Shadow')


                
            CI = None
            # Length of met sequence
            n_hours = (Ta.shape[0] if not met_on_cpu else len(Ta_cpu))
            for i in np.arange(0, n_hours):
                try:
                    hour_t0 = time.time()
                    if hour_counter % 24 == 0:
                        current_date_str = pytz.utc.localize(base_date + datetime.timedelta(hours=hour_counter)) \
                                               .astimezone(pytz.timezone("UTC")).strftime('%Y%m%d')
                        _open_day(current_date_str)
                        # start per-day timer for this sub-tile
                        day_t0 = time.time()
                   
                   
                    if landcover == 1:
                        if ((dectime[i] - np.floor(dectime[i]))) == 0 or (i == 0):
                            Ta_ = (Ta_cpu if met_on_cpu else Ta.cpu().numpy())
                            val = float(dectime[i])
                            day_idx = float(np.floor(val))
                            jday_np = np.asarray(jday)[0]
                            mask = (jday_np == day_idx)
                            Twater = float(np.mean(Ta_[mask])) if np.any(mask) else float(np.mean(Ta_))

                    if CI is None and i == 0:
                        
                        CI_tmp = 1.0

                        if (dectime[i] - np.floor(dectime[i])) == 0:
                            daylines = np.where(np.floor(dectime) == dectime[i])
                            if len(daylines) > 1:
                                alt = altitude[0][daylines]
                                alt2 = np.where(alt > 1)
                                if alt2[0].size > 0:
                                    rise = alt2[0][0]
                                    # Calcolo CI come in precedenza
                                    [_, CI_tmp, _, _, _] = clearnessindex_2013b(
                                        torch.as_tensor(zen[0, i + rise + 1],device=device),
                                        torch.as_tensor(jday[0, i + rise + 1],device=device),
                                        (torch.tensor(float(Ta_cpu[i + rise + 1]), device=device) if met_on_cpu else Ta[i + rise + 1]),
                                        (torch.tensor(float(RH_cpu[i + rise + 1] / 100.0), device=device) if met_on_cpu else (RH[i + rise + 1] / 100.0)),
                                        (torch.tensor(float(radG_cpu[i + rise + 1]), device=device) if met_on_cpu else radG[i + rise + 1]),
                                        location,
                                        (torch.tensor(float(P_cpu[i + rise + 1]), device=device) if met_on_cpu else P[i + rise + 1])
                                    )

                                    if (CI_tmp > 1.0) or (CI_tmp == np.inf) or (np.isnan(CI_tmp)):
                                        CI_tmp = 1.0
                            else:
                                CI_tmp = 1.0

                        CI = float(CI_tmp)

                    
                    g, cp, Rd, Rv = 9.80665, 1004., 287., 461.
                    eps = Rd/Rv
                    esat=lambda Tc:611.2*torch.exp(17.67*Tc/(Tc+243.5))
                    qsat=lambda Tc,p:(eps*esat(Tc))/(p-(1-eps)*esat(Tc))
                    gamma_m=lambda Tc,p:(g*(1+2.5e6*qsat(Tc,p*100)/(Rd*(Tc+273.15))))/(cp+((2.5e6)**2*qsat(Tc,p)*eps)/(Rd*(Tc+273.15)**2))
                    gamma_eff=lambda Tc,p,RH:(1-((RH/100-0.6)/0.35).clamp(0,1))*(g/cp)+((RH/100-0.6)/0.35).clamp(0,1)*gamma_m(Tc,p)
                    temperature_at_height=lambda Tc,p,RH,hs:Tc+gamma_eff(Tc,p,RH)*hs

                    # Pull per-hour met scalars (optionally from CPU)
                    if met_on_cpu:
                        Ta_i = float(Ta_cpu[i]); RH_i = float(RH_cpu[i]); radG_i = float(radG_cpu[i]); radD_i = float(radD_cpu[i]); radI_i = float(radI_cpu[i]); P_i = float(P_cpu[i]); Ws_i = float(Ws_cpu[i])
                    else:
                        Ta_i = float(Ta[i].item()); RH_i = float(RH[i].item()); radG_i = float(radG[i].item()); radD_i = float(radD[i].item()); radI_i = float(radI[i].item()); P_i = float(P[i].item()); Ws_i = float(Ws[i].item())

                    ta_scalar.fill_(Ta_i)
                    rh_scalar.fill_(RH_i)
                    p_scalar.fill_(P_i)
                    radg_scalar.fill_(radG_i)
                    radd_scalar.fill_(radD_i)
                    radi_scalar.fill_(radI_i)
                    ws_scalar.fill_(Ws_i)
                    uhi_scalar.fill_(float(uhi_series[i]))

                    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                        if windcoeff is not None:
                            torch.mul(windcoeff, ws_scalar, out=va10m_mat)
                        else:
                            va10m_mat.fill_(Ws_i)
                        va10m_safe = torch.clamp(va10m_mat, min=1e-3, out=va10m_safe_buf)

                        Ta_mat = temperature_at_height(ta_scalar, p_scalar, rh_scalar, hsdiff)

                        if use_uhi_cycle:
                            wind_factor = torch.pow(va10m_safe.to(torch.float32), -0.25).to(building.dtype)
                            uhi_term = (2.0 - svf - fveg) * uhi_scalar * wind_factor
                            Ta_mat = Ta_mat + uhi_term

                        Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, shadow, firstdaytime, timestepdec, timeadd, \
                Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, Keast, Ksouth, Kwest, Knorth, Least, Lsouth, Lwest, Lnorth, \
                 KsideI, TgOut1, TgOut, radIout, radDout, Lside, Lsky_patch_characteristics, CI_Tg, CI_TgG, KsideD, dRad, Kside = Solweig_2022a_calc(
                i, building, scale, rows, cols, svf, svfN, svfW, svfE, svfS, svfveg, svfNveg, svfEveg, svfSveg, svfWveg, svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg, vegdsm, vegdsm2, albedo_b, absK, absL, ewall, Fside, Fup, Fcyl,
                altitude[0][i], azimuth[0][i], zen[0][i], jday[0][i], usevegdem, onlyglobal, buildings, location, psi[0][i], landcover, landuse, hsdiff, dectime[i], altmax[0][i], dirwalls, walls, cyl, elvis, Ta_mat,
                rh_scalar, radg_scalar, radd_scalar, radi_scalar, p_scalar,
                amaxvalue, bush, Twater, TgK, Tstart, alb, emis, TgK_wall, Tstart_wall, TmaxLST, TmaxLST_wall, first, second, svfalfa, svfbuveg, transVeg, firstdaytime, timeadd, timestepdec, Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N,
                CI,TgOut1, shmat, vegshmat, vbshvegshmat, anisotropic_sky, asvf, patch_option)

                        RH_mat = torch.full(value_shape, RH_i, dtype=building.dtype, device=device)
                        UTCI_mat = utci_calculator(Ta_mat, RH_mat, Tmrt.expand(value_shape), va10m_mat)
                        # UTCI min/max for logging (valid pixels only)
                        utci_min_log, utci_max_log = _safe_minmax(UTCI_mat[valid_mask])
                    # Compute per-hour summary stats for logging before freeing tensors (no try/except)
                    ta_min_log, ta_max_log = _safe_minmax(Ta_mat)
                    tmrt_min_log, tmrt_max_log = _safe_minmax(Tmrt)
                    wind_min_log, wind_max_log = _safe_minmax(va10m_mat)

                    band = (hour_counter % 24) + 1
                    y0 = int(r0); x0 = int(c0)
                    out_utci.fill(-9999.0)
                    out_utci[valid_mask_np] = UTCI_mat[valid_mask].detach().cpu().numpy()
                    _qput(('write', 'UTCI', current_date_str, int(band), x0, y0, out_utci))
                    # Always write Ta and Va10m (masked by valid_mask)
                    out_ta.fill(-9999.0)
                    out_ta[valid_mask_np] = Ta_mat[valid_mask].detach().cpu().numpy()
                    _qput(('write', 'Ta', current_date_str, int(band), x0, y0, out_ta))

                    out_va.fill(-9999.0)
                    out_va[valid_mask_np] = va10m_mat[valid_mask].detach().cpu().numpy()
                    _qput(('write', 'Va10m', current_date_str, int(band), x0, y0, out_va))

                    if save_tmrt:
                        out_tmrt.fill(-9999.0)
                        out_tmrt[valid_mask_np] = Tmrt[valid_mask].detach().cpu().numpy()
                        _qput(('write', 'TMRT', current_date_str, int(band), x0, y0, out_tmrt))
                    if save_kup:   _qput(('write', 'Kup',   current_date_str, int(band), x0, y0, Kup.detach().cpu().numpy().astype('float32')))
                    if save_kdown:_qput(('write', 'Kdown', current_date_str, int(band), x0, y0, Kdown.detach().cpu().numpy().astype('float32')))
                    if save_lup:  _qput(('write', 'Lup',   current_date_str, int(band), x0, y0, Lup.detach().cpu().numpy().astype('float32')))
                    if save_ldown:_qput(('write', 'Ldown', current_date_str, int(band), x0, y0, Ldown.detach().cpu().numpy().astype('float32')))
                    if save_shadow:_qput(('write', 'Shadow',current_date_str, int(band), x0, y0, shadow.detach().cpu().numpy().astype('float32')))

                    # --- Backpressure: slow down GPU if writer queue grows too much ---
                    try:
                        maxq = int(os.environ.get('SOLWEIG_BACKPRESSURE_MAXQ', '0'))
                        if maxq > 0:
                            try:
                                cur_q = writer_q.qsize()
                            except Exception:
                                cur_q = 0
                            if cur_q >= maxq:
                                sl_ms = int(os.environ.get('SOLWEIG_BACKPRESSURE_SLEEP_MS', '50'))
                                time.sleep(max(1, sl_ms) / 1000.0)
                    except Exception:
                        pass

                    try:
                        del UTCI_mat, RH_mat, Kdown, Kup, Ldown, Lup, shadow
                    except Exception:
                        pass
                    try:
                        # Free last to allow stats computed above
                        del Ta_mat, Tmrt
                    except Exception:
                        pass

                    # Aggressive per-hour memory hygiene (reduces long-run growth)
                    _mem_cleanup()

                    hour_counter += 1
                    if (hour_counter % log_every) == 0:
                        elapsed = time.time() - hour_t0
                        current_datetime = (pytz.utc.localize(base_date + datetime.timedelta(hours=hour_counter))).astimezone(pytz.timezone("UTC"))
                        utc_stamp = current_datetime.strftime('%Y-%m-%d:%H')
                        r0p = int(r0) if 'r0' in locals() else 0
                        r1p = int(r1) if 'r1' in locals() else rows
                        c0p = int(c0) if 'c0' in locals() else 0
                        c1p = int(c1) if 'c1' in locals() else cols
                        rh_val = float(RH_i)
                        tile_desc = str(number) if number is not None else f"rows {r0p}-{r1p} cols {c0p}-{c1p}"
                        print(
                            f"[HOUR] {device_tag} tile {tile_desc} UTC {utc_stamp} (elapsed {elapsed:.2f}s)\n"
                            f"        RH={rh_val:.2f}% Ta=[{ta_min_log:.2f},{ta_max_log:.2f}] "
                            f"Tmrt=[{tmrt_min_log:.2f},{tmrt_max_log:.2f}] UTCI=[{utci_min_log:.2f},{utci_max_log:.2f}] "
                            f"Wind=[{wind_min_log:.2f},{wind_max_log:.2f}]",
                            flush=True
                        )

                    if hour_counter == 24:
                        _qput(('tile_done', 'UTCI', current_date_str))
                        _qput(('tile_done', 'Ta',   current_date_str))
                        _qput(('tile_done', 'Va10m',current_date_str))
                        if save_tmrt:  _qput(('tile_done', 'TMRT',  current_date_str))
                        if save_kup:   _qput(('tile_done', 'Kup',   current_date_str))
                        if save_kdown: _qput(('tile_done', 'Kdown', current_date_str))
                        if save_lup:   _qput(('tile_done', 'Lup',   current_date_str))
                        if save_ldown: _qput(('tile_done', 'Ldown', current_date_str))
                        if save_shadow:_qput(('tile_done', 'Shadow',current_date_str))
                        # --- Per-day synchronization (only if internal_sync=True)
                        if internal_sync:
                            _needed = ['UTCI', 'Ta', 'Va10m']
                            if save_tmrt:  _needed.append('TMRT')
                            if save_kup:   _needed.append('Kup')
                            if save_kdown: _needed.append('Kdown')
                            if save_lup:   _needed.append('Lup')
                            if save_ldown: _needed.append('Ldown')
                            if save_shadow:_needed.append('Shadow')
                            _wait_day_ready(current_date_str, _needed, output_path)
                        # end-of-day log for this sub-tile
                        dt_day = time.time() - day_t0
                        r0p = int(r0); r1p = int(r1); c0p = int(c0); c1p = int(c1)
                        print(f"[DAY] {device_tag} sub_tile r[{r0p}:{r1p}] c[{c0p}:{c1p}] UTC {current_date_str} elapsed {dt_day:.1f}s", flush=True)
                        base_date += datetime.timedelta(days=1)
                        try:
                            gc.collect()
                            if use_cuda:
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                        except Exception:
                            pass
                        _trim_os_heap()
                        hour_counter = 0
                except Exception as e:
                    import traceback
                    print(f"[{device_bracket}] ERROR hour {i} in tile {number}: {e}", flush=True)
                    traceback.print_exc()
                    # Fallback: if the day has started, still emit tile_done markers
                    if (hour_counter > 0) and (current_date_str is not None):
                        _qput(('tile_done', 'UTCI', current_date_str))
                        _qput(('tile_done', 'Ta',   current_date_str))
                        _qput(('tile_done', 'Va10m',current_date_str))
                        if save_tmrt:  _qput(('tile_done', 'TMRT',  current_date_str))
                        if save_kup:   _qput(('tile_done', 'Kup',   current_date_str))
                        if save_kdown: _qput(('tile_done', 'Kdown', current_date_str))
                        if save_lup:   _qput(('tile_done', 'Lup',   current_date_str))
                        if save_ldown: _qput(('tile_done', 'Ldown', current_date_str))
                        if save_shadow:_qput(('tile_done', 'Shadow',current_date_str))
                        try:
                            gc.collect()
                            if use_cuda:
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                        except Exception:
                            pass
                        _trim_os_heap()
                    break

            # --- PARTIAL-DAY FLUSH (if loop ended mid-day) ---
            if (hour_counter > 0) and (hour_counter < 24) and (current_date_str is not None):
                _qput(('tile_done', 'UTCI', current_date_str))
                _qput(('tile_done', 'Ta',   current_date_str))
                _qput(('tile_done', 'Va10m',current_date_str))
                if save_tmrt:  _qput(('tile_done', 'TMRT',  current_date_str))
                if save_kup:   _qput(('tile_done', 'Kup',   current_date_str))
                if save_kdown: _qput(('tile_done', 'Kdown', current_date_str))
                if save_lup:   _qput(('tile_done', 'Lup',   current_date_str))
                if save_ldown: _qput(('tile_done', 'Ldown', current_date_str))
                if save_shadow:_qput(('tile_done', 'Shadow',current_date_str))
                dt_day = time.time() - day_t0
                r0p = int(r0); r1p = int(r1); c0p = int(c0); c1p = int(c1)
                print(f"[DAY][partial] {device_tag} sub_tile r[{r0p}:{r1p}] c[{c0p}:{c1p}] UTC {current_date_str} elapsed {dt_day:.1f}s", flush=True)
                try:
                    gc.collect()
                    if use_cuda:
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception:
                    pass
                _trim_os_heap()
                hour_counter = 0

        # --- END AUTOCast block ---
        if use_cuda:
            torch.cuda.empty_cache()
        _mem_cleanup()
        try:
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        _trim_os_heap()
        # Optional: evict cached SVF tensors for this tile after its final day (multi-tile per GPU mode)
        try:
            if cache_evict:
                _SVF_CACHE.pop(str(number), None)
        except Exception:
            pass
        
"""
UTCI processing pipeline for SOLWEIG tiles.

This module orchestrates per-tile UTCI computations on CPU/GPU, handles
GeoTIFF writing via a dedicated process, and manages memory hygiene.
All comments are in English for clarity.
"""

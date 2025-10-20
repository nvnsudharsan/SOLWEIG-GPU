import math
import os

import numpy as np
from osgeo import gdal
from scipy.ndimage import maximum_filter, rotate
gdal.UseExceptions()

# Default: enable fast walls/aspect path unless overridden by env
os.environ.setdefault('SOLWEIG_FAST_WALLS_ASPECT', '1')

# Wall height threshold
walllimit = 3.0

def findwalls(dem_array, walllimit):
    """Fast wall detector using a cross-shaped maximum filter on DEM.
    Equivalent to taking the max of N,S,E,W neighbors and subtracting the center.
    """
    # Cross-shaped footprint: [[0,1,0],[1,0,1],[0,1,0]]
    footprint = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.uint8)
    neigh_max = maximum_filter(dem_array, footprint=footprint, mode='nearest')
    walls = neigh_max.astype(np.float32) - dem_array.astype(np.float32)
    # Threshold and clear borders to match original behavior
    walls[walls < walllimit] = 0.0
    walls[:, 0] = 0.0
    walls[:, -1] = 0.0
    walls[0, :] = 0.0
    walls[-1, :] = 0.0
    return walls

def cart2pol(x, y, units='deg'):
    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    if units in ['deg', 'degs']:
        theta = theta * 180 / np.pi
    return theta, radius

def get_ders(dsm, scale):
    dx = 1 / scale
    fy, fx = np.gradient(dsm, dx, dx)
    asp, grad = cart2pol(fy, fx, 'rad')
    grad = np.arctan(grad)
    asp = -asp
    asp[asp < 0] += 2 * np.pi
    return grad, asp

# --- FAST aspect helper ---
def _fast_aspect_from_gradient(dsm, scale):
    """Compute aspect in degrees [0,360) from DEM gradients (fast path)."""
    dx = 1.0 / scale
    fy, fx = np.gradient(dsm, dx, dx)
    theta = np.arctan2(fy, fx)        # radians
    deg = -np.degrees(theta)          # invert to align with original sign convention
    deg[deg < 0] += 360.0
    return deg.astype(np.float32)

def filter1Goodwin_as_aspect_v3(walls, scale, a):
    """Return wall aspect angles (degrees). Fast path uses DEM gradient.
    Set env SOLWEIG_FAST_WALLS_ASPECT=0 to revert to the legacy exhaustive filter (slow).
    """
    # --- FAST PATH (default) ---
    if os.environ.get('SOLWEIG_FAST_WALLS_ASPECT', '1') == '1':
        aspects_deg = _fast_aspect_from_gradient(a, scale)
        # Only keep directions where walls exist; elsewhere 0 (as previous code would keep 0 for non-walls)
        out = np.zeros_like(aspects_deg, dtype=np.float32)
        wall_mask = (walls > 0)
        out[wall_mask] = aspects_deg[wall_mask]
        return out

    # --- LEGACY SLOW PATH BELOW (kept for compatibility if explicitly enabled) ---
    row, col = a.shape
    filtersize = int(np.floor((scale + 1e-10) * 9))
    if filtersize <= 2:
        filtersize = 3
    elif filtersize != 9 and filtersize % 2 == 0:
        filtersize += 1

    n = filtersize - 1
    filthalveceil = int(np.ceil(filtersize / 2.))
    filthalvefloor = int(np.floor(filtersize / 2.))

    filtmatrix = np.zeros((filtersize, filtersize))
    buildfilt = np.zeros((filtersize, filtersize))
    filtmatrix[:, filthalveceil - 1] = 1
    buildfilt[filthalveceil - 1, :filthalvefloor] = 1
    buildfilt[filthalveceil - 1, filthalveceil:] = 2

    y = np.zeros((row, col))
    z = np.zeros((row, col))
    x = np.zeros((row, col))
    walls = (walls > 0).astype(np.uint8)

    for h in range(0, 180):
        filtmatrix1 = np.round(rotate(filtmatrix, h, order=1, reshape=False, mode='nearest'))
        filtmatrixbuild = np.round(rotate(buildfilt, h, order=0, reshape=False, mode='nearest'))
        index = 270 - h

        if h in [150, 30]:
            filtmatrixbuild[:, n] = 0
        if index == 225:
            filtmatrix1[0, 0] = filtmatrix1[n, n] = 1
        if index == 135:
            filtmatrix1[0, n] = filtmatrix1[n, 0] = 1

        for i in range(filthalveceil - 1, row - filthalveceil - 1):
            for j in range(filthalveceil - 1, col - filthalveceil - 1):
                if walls[i, j] == 1:
                    wallscut = walls[i - filthalvefloor:i + filthalvefloor + 1,
                                     j - filthalvefloor:j + filthalvefloor + 1] * filtmatrix1
                    dsmcut = a[i - filthalvefloor:i + filthalvefloor + 1,
                               j - filthalvefloor:j + filthalvefloor + 1]
                    if z[i, j] < wallscut.sum():
                        z[i, j] = wallscut.sum()
                        x[i, j] = 1 if np.sum(dsmcut[filtmatrixbuild == 1]) > np.sum(dsmcut[filtmatrixbuild == 2]) else 2
                        y[i, j] = index

    y[x == 1] -= 180
    y[y < 0] += 360

    grad, asp = get_ders(a, scale)
    y += ((walls == 1) & (y == 0)) * (asp / (math.pi / 180.))

    return y



def run_parallel_processing(dem_tile, wall_path, aspect_path, geotransform, projection, r_start, c_start):
    """
    Generate walls/aspect rasters for a tile in memory,
    correcting the geotransform based on the row and column offsets (r_start, c_start).
    """
    from osgeo import gdal
    import numpy as np

    try:
        if dem_tile is None or dem_tile.size == 0 or np.all(np.isnan(dem_tile)):
            print("⚠️ Skipping tile: invalid or empty DEM.")
            return False

        # Adjust the GT for the tile (shift the origin to the top-left of the sub-domain)
        gt = list(geotransform)
        # Shift both by row and col offsets using the full affine terms
        x0, px_w, rot_x, y0, rot_y, px_h = gt
        row_off = float(r_start)
        col_off = float(c_start)
        x0_new = x0 + col_off * px_w + row_off * rot_x
        y0_new = y0 + col_off * rot_y + row_off * px_h
        gt[0] = x0_new
        gt[3] = y0_new

        scale = 1.0 / gt[1]

        # Compute walls/aspect from the tile
        walls = findwalls(dem_tile, walllimit)
        aspects = filter1Goodwin_as_aspect_v3(walls, scale, dem_tile)

        driver = gdal.GetDriverByName('GTiff')
        rows, cols = dem_tile.shape
        opts = [
            'COMPRESS=NONE',
            'TILED=YES',
            'BLOCKXSIZE=256',
            'BLOCKYSIZE=256',
            'INTERLEAVE=BAND',
            'BIGTIFF=YES',
            'NUM_THREADS=ALL_CPUS'
        ]

        walls = walls.astype(np.float32, copy=False)
        aspects = aspects.astype(np.float32, copy=False)

        # Save wall raster
        out_ds_walls = driver.Create(wall_path, cols, rows, 1, gdal.GDT_Float32, options=opts)
        out_ds_walls.SetGeoTransform(tuple(gt))
        out_ds_walls.SetProjection(projection)
        out_ds_walls.GetRasterBand(1).WriteArray(walls)
        out_ds_walls.GetRasterBand(1).SetNoDataValue(0.0)
        out_ds_walls.FlushCache()
        out_ds_walls = None

        # Save aspect raster
        out_ds_aspect = driver.Create(aspect_path, cols, rows, 1, gdal.GDT_Float32, options=opts)
        out_ds_aspect.SetGeoTransform(tuple(gt))
        out_ds_aspect.SetProjection(projection)
        out_ds_aspect.GetRasterBand(1).WriteArray(aspects)
        out_ds_aspect.GetRasterBand(1).SetNoDataValue(0.0)
        out_ds_aspect.FlushCache()
        out_ds_aspect = None

        print(f"Tile OK: r{r_start}-{r_start + dem_tile.shape[0]} c{c_start}-{c_start + dem_tile.shape[1]}")
        return True

    except Exception as e:
        print(f"Tile ERROR r{r_start} c{c_start}: {e}")
        return False


# --- Mosaic and cleanup helper ---
def mosaic_and_cleanup_tiles(input_data_path, walls_dir, aspect_dir):
    """
    Build final walls/aspect mosaics from per-tile TIFs and remove tile files.
    """
    from osgeo import gdal
    import glob
    import os

    walls_tiles = sorted(glob.glob(os.path.join(input_data_path, "walls_*.tif")))
    aspect_tiles = sorted(glob.glob(os.path.join(input_data_path, "aspect_*.tif")))

    if walls_tiles:
        walls_vrt = os.path.join(input_data_path, "walls_tiles.vrt")
        print(f"Mosaic walls: {len(walls_tiles)} tiles → {walls_dir}")
        vrt = gdal.BuildVRT(walls_vrt, walls_tiles)
        if vrt is not None:
            vrt = None
            translate_opts = gdal.TranslateOptions(creationOptions=['COMPRESS=NONE','TILED=YES','INTERLEAVE=BAND','BIGTIFF=YES','NUM_THREADS=ALL_CPUS'])
            if gdal.Translate(walls_dir, walls_vrt, options=translate_opts) is not None:
                os.remove(walls_vrt)
                for fp in walls_tiles:
                    try:
                        os.remove(fp)
                    except Exception as e:
                        print(f"Cleanup warn: {fp} not removed: {e}")
            else:
                print("Mosaic walls: translate failed")
        else:
            print("Mosaic walls: buildvrt failed")

    if aspect_tiles:
        aspect_vrt = os.path.join(input_data_path, "aspect_tiles.vrt")
        print(f"Mosaic aspect: {len(aspect_tiles)} tiles → {aspect_dir}")
        vrt = gdal.BuildVRT(aspect_vrt, aspect_tiles)
        if vrt is not None:
            vrt = None
            translate_opts = gdal.TranslateOptions(creationOptions=['COMPRESS=NONE','TILED=YES','INTERLEAVE=BAND','BIGTIFF=YES','NUM_THREADS=ALL_CPUS'])
            if gdal.Translate(aspect_dir, aspect_vrt, options=translate_opts) is not None:
                os.remove(aspect_vrt)
                for fp in aspect_tiles:
                    try:
                        os.remove(fp)
                    except Exception as e:
                        print(f"Cleanup warn: {fp} not removed: {e}")
            else:
                print("Mosaic aspect: translate failed")
        else:
            print("Mosaic aspect: buildvrt failed")

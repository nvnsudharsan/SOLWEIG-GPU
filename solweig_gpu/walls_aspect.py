import os
import numpy as np
from osgeo import gdal
from scipy.ndimage import rotate
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Wall height threshold
walllimit = 3.0

def findwalls(dem_array, walllimit):
    col, row = dem_array.shape
    walls = np.zeros((col, row))
    domain = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            dom = dem_array[j - 1:j + 2, i - 1:i + 2]
            walls[j, i] = np.max(dom[domain == 1])

    walls = walls - dem_array
    walls[walls < walllimit] = 0

    walls[:, 0] = 0
    walls[:, -1] = 0
    walls[0, :] = 0
    walls[-1, :] = 0

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

def filter1Goodwin_as_aspect_v3(walls, scale, a):
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

def process_file_parallel(args):
    filename, dem_folder_path, wall_output_path, aspect_output_path = args
    dem_path = os.path.join(dem_folder_path, filename)

    try:
        dataset = gdal.Open(dem_path)
        if dataset is None:
            print(f"Could not open {filename}")
            return filename

        band = dataset.GetRasterBand(1)
        a = band.ReadAsArray().astype(np.float32)

        if a is None or np.all(np.isnan(a)):
            print(f"Skipping {filename}, invalid DEM.")
            return filename

        scale = 1 / dataset.GetGeoTransform()[1]
        walls = findwalls(a, walllimit)
        aspects = filter1Goodwin_as_aspect_v3(walls, scale, a)

        driver = gdal.GetDriverByName('GTiff')
        out_names = [f"walls_{filename[13:-4]}.tif", f"aspect_{filename[13:-4]}.tif"]
        out_paths = [os.path.join(wall_output_path, out_names[0]),
                     os.path.join(aspect_output_path, out_names[1])]

        for out_path, data in zip(out_paths, [walls, aspects]):
            out_ds = driver.Create(out_path, a.shape[1], a.shape[0], 1, gdal.GDT_Float32)
            out_ds.SetGeoTransform(dataset.GetGeoTransform())
            out_ds.SetProjection(dataset.GetProjection())
            out_ds.GetRasterBand(1).WriteArray(data)
            out_ds.FlushCache()
            out_ds = None

        return filename

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return filename

def run_parallel_processing(dem_folder_path, wall_output_path, aspect_output_path):
    os.makedirs(wall_output_path, exist_ok=True)
    os.makedirs(aspect_output_path, exist_ok=True)

    dem_files = [f for f in os.listdir(dem_folder_path) if f.endswith('.tif') and not f.startswith('.')]
    args_list = [(f, dem_folder_path, wall_output_path, aspect_output_path) for f in dem_files]

    max_workers = min(32, os.cpu_count() or 1)
    print(f"Using {max_workers} parallel workers")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file_parallel, args): args[0] for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing Wall Height and Aspect", mininterval = 1):
            _ = future.result()


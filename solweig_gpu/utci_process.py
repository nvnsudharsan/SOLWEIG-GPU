from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import numpy as np
from math import radians
from copy import deepcopy
from osgeo import gdal, osr
import datetime
import calendar
import scipy.ndimage.interpolation as sc
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate
import time
from timezonefinder import TimezoneFinder
import pytz
import datetime
from .Tgmaps_v1 import Tgmaps_v1
from .sun_position import Solweig_2015a_metdata_noload
from .shadow import svf_calculator, create_patches
from .solweig import Solweig_2022a_calc, clearnessindex_2013b
from .calculate_utci import utci_calculator
import os
import re
# from .preprocessor import ppr
from .walls_aspect import run_parallel_processing
gdal.UseExceptions()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = os.path.dirname(__file__)
landcover_classes_path = os.path.join(script_dir, 'landcoverclasses_2016a.txt')

# Wall and ground emissivity and albedo
albedo_b = 0.2
albedo_g = 0.15
ewall = 0.9
eground = 0.95
absK = 0.7
absL = 0.95

# Standing position
Fside = 0.22
Fup = 0.06
Fcyl = 0.28

cyl = True
elvis = 0
usevegdem = 1
onlyglobal = 1

firstdayleaf = 97
lastdayleaf = 300
conifer_bool = False

def load_raster_to_tensor(dem_path):
    dataset = gdal.Open(dem_path)
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray().astype(np.float32)
    return torch.tensor(array, device=device), dataset

def extract_key(filename, is_metfile=False):

    if is_metfile:
        # look for metfile_X_Y_DATE
        match = re.search(r'metfile_(\d+)_(\d+)_\d{4}-\d{2}-\d{2}', filename)
    else:
        # look for ..._X_Y.tif
        match = re.search(r'_(\d+)_(\d+)', filename)

    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return None

# Function to list matching files in a directory
def get_matching_files(directory, extension):
    return sorted([f for f in os.listdir(directory) if f.endswith(extension)])

def map_files_by_key(directory, extension, is_metfile=False):
    files = get_matching_files(directory, extension)
    mapping = {}
    for f in files:
        key = extract_key(f, is_metfile=is_metfile)
        if key:
            mapping[key] = os.path.join(directory, f)
    return mapping

def extract_number_from_filename(filename):
    number = filename[13:-4] # change according to the naming of building DSM files
    return number


def compute_utci(building_dsm_path, tree_path, dem_path, walls_path, aspect_path, landcover_path, met_file, 
                output_path,number,selected_date_str,save_tmrt=False,save_svf=False, save_kup=False,save_kdown=False,save_lup=False,save_ldown=False,save_shadow=False):
    a, dataset = load_raster_to_tensor(building_dsm_path)
    temp1, dataset2 = load_raster_to_tensor(tree_path)
    temp2, dataset3 = load_raster_to_tensor(dem_path)
    walls, dataset4 = load_raster_to_tensor(walls_path)
    dirwalls, dataset5 = load_raster_to_tensor(aspect_path)
 
    # Added
    landcover = 0

    if landcover_path is not None:
        landcover = 1
        lcgrid_torch, dataset6 = load_raster_to_tensor(landcover_path)
        lcgrid_np = lcgrid_torch.cpu().numpy()
        #lcgrid_np = lcgrid_np.astype(int)
        
        mask_invalid = (lcgrid_np < 1) | (lcgrid_np > 7)
        if mask_invalid.any():
            print("Warning: land-cover grid contains values outside 1-7. "
                "Invalid cells are set to 6 (bare soil). ")
            lcgrid_np[mask_invalid] = 6
        
        mask_vegetation = (lcgrid_np == 3) | (lcgrid_np == 4)
        if mask_vegetation.any():
            print("Attention!",
                  "The land cover grid includes values (deciduous and/or conifer) not appropriate for the SOLWEIG-formatted land cover grid (should not include 3 or 4). "
                  "Land cover under the vegetation is required. "
                  "Setting the invalid landcover types to grass.")
            lcgrid_np[mask_vegetation] = 5

        with open(landcover_classes_path) as f:
            lines = f.readlines()[1:]                            # skip header line
        lc_class = np.empty((len(lines), 6), dtype=float)
        for i, ln in enumerate(lines):
            lc_class[i, :] = [float(x) for x in ln.split()[1:]]  # cols 1-6
    # Added
    
    base_date = datetime.datetime.strptime(selected_date_str, "%Y-%m-%d")
    rows, cols = a.shape
    geotransform = dataset.GetGeoTransform()
    scale = 1 / geotransform[1]
    projection_wkt = dataset.GetProjection()
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(projection_wkt) 
    old_cs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    wgs84_wkt = """GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]"""
    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)
    new_cs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(old_cs, new_cs)
    widthx = dataset.RasterXSize
    heightx = dataset.RasterYSize
    geotransform = dataset.GetGeoTransform()
    #minx = geotransform[0]
    #miny = geotransform[3] + widthx * geotransform[4] + heightx * geotransform[5]
    #lonlat = transform.TransformPoint(minx, miny)
    #gdalver = float(gdal.__version__[0])
    #if gdalver == 3.:
    #    lon = lonlat[1]  # changed to gdal 3
    #    lat = lonlat[0]  # changed to gdal 3
    #else:
    #    lon = lonlat[0]  # changed to gdal 2
    #    lat = lonlat[1]  # changed to gdal 2
    centre_x = geotransform[0] + geotransform[1] * widthx  / 2.0
    centre_y = geotransform[3] + geotransform[5] * heightx / 2.0
    lon, lat = transform.TransformPoint(centre_x, centre_y)[:2]
    alt = torch.median(temp2)
    alt = alt.cpu().item()
    if alt > 0:
        alt = 3.
    location = {'longitude': lon, 'latitude': lat, 'altitude': alt}
    # After computing lat and lon
    tf = TimezoneFinder()
    timezone_name = tf.timezone_at(lat=lat, lng=lon) or "UTC"
    local_tz = pytz.timezone(timezone_name)
    # Use a sample date (today or specific) to get current UTC offset
    local_dt = local_tz.localize(base_date)
    utc = local_dt.utcoffset().total_seconds() / 3600
    print(f"[INFO] Timezone: {timezone_name}, UTC offset: {utc} hours")
    YYYY, altitude, azimuth, zen, jday, leafon, dectime, altmax = Solweig_2015a_metdata_noload(met_file, location, utc)
    temp1[temp1 < 0.] = 0.
    vegdem = temp1 + temp2
    vegdem2 = torch.add(temp1 * 0.25, temp2)
    bush = torch.logical_not(vegdem2 * vegdem) * vegdem
    vegdsm = temp1 + a
    vegdsm[vegdsm == a] = 0
    vegdsm2 = temp1 * 0.25 + a
    vegdsm2[vegdsm2 == a] = 0
    amaxvalue = torch.maximum(a.max(), vegdem.max())
    buildings = a - temp2
    buildings[buildings < 2.] = 1.
    buildings[buildings >= 2.] = 0.
    valid_mask = (buildings == 1)
    Knight = torch.zeros((rows, cols), device=device)
    Tgmap1 = torch.zeros((rows, cols), device=device)
    Tgmap1E = torch.zeros((rows, cols), device=device)
    Tgmap1S = torch.zeros((rows, cols), device=device)
    Tgmap1W = torch.zeros((rows, cols), device=device)
    Tgmap1N = torch.zeros((rows, cols), device=device)
    TgOut1 = torch.zeros((rows, cols), device=device)
    
    # Added
    if landcover == 1:                                     
        (TgK_np, Tstart_np, alb_np, emis_np, TgK_wall_np, Tstart_wall_np, TmaxLST_np,
         TmaxLST_wall_np) = Tgmaps_v1(lcgrid_np, lc_class)
           
        TgK           = torch.from_numpy(TgK_np).to(device).float()
        Tstart        = torch.from_numpy(Tstart_np).to(device).float()
        alb_grid      = torch.from_numpy(alb_np).to(device).float()
        emis_grid     = torch.from_numpy(emis_np).to(device).float()
        TgK_wall      = torch.tensor(float(TgK_wall_np)     , device=device)
        Tstart_wall   = torch.tensor(float(Tstart_wall_np)  , device=device)
        TmaxLST       = torch.from_numpy(TmaxLST_np ).to(device).float()
        TmaxLST_wall  = torch.tensor(float(TmaxLST_wall_np) , device=device)
    else:
        TgK = Knight + 0.37
        Tstart = Knight - 3.41
        alb_grid = Knight + albedo_g
        emis_grid = Knight + eground
        TgK_wall = 0.37
        Tstart_wall = -3.41
        TmaxLST = 15.
        TmaxLST_wall = 15.
    # Added
    
    transVeg = 3. / 100.
    # landcover = 1 # Modified
    if landcover == 1:
        lcgrid = lcgrid_torch
    else:
        lcgrid = False
    anisotropic_sky = 1
    patch_option = 2
    DOY = torch.tensor(met_file[:, 1], device=device)
    hours = torch.tensor(met_file[:, 2], device=device)
    minu = torch.tensor(met_file[:, 3], device=device)
    Ta = torch.tensor(met_file[:, 11], device=device)
    RH = torch.tensor(met_file[:, 10], device=device)
    radG = torch.tensor(met_file[:, 14], device=device)
    radD = torch.tensor(met_file[:, 21], device=device)
    radI = torch.tensor(met_file[:, 22], device=device)
    P = torch.tensor(met_file[:, 12], device=device)
    Ws = torch.tensor(met_file[:, 9], device=device)
    # Prepare leafon based on vegetation type
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
    #first = torch.round(torch.tensor(height, device=device))
    first = torch.round(height.clone().detach().to(device))
    if first == 0.:
        first = torch.tensor(1., device=device)
    second = torch.round(height * 20.)
    if len(Ta) == 1:
        timestepdec = 0
    else:
        timestepdec = dectime[1] - dectime[0]
    timeadd = 0.
    firstdaytime = 1.
    start_time = time.time()
    # Calculate SVF and related parameters (remains unchanged)
    svf, svfaveg, svfE, svfEaveg, svfEveg, svfN, svfNaveg, svfNveg, svfS, svfSaveg, svfSveg, svfveg, svfW, svfWaveg, svfWveg, vegshmat, vbshvegshmat, shmat, svftotal = svf_calculator(patch_option, amaxvalue, a, vegdsm, vegdsm2, bush, scale)
    svfbuveg = svf - (1.0 - svfveg) * (1.0 - transVeg)
    asvf = torch.acos(torch.sqrt(svf))
    diffsh = torch.zeros((rows, cols, shmat.shape[2]), device=device)
    for i in range(shmat.shape[2]):
        diffsh[:, :, i] = shmat[:, :, i] - (1 - vegshmat[:, :, i]) * (1 - transVeg)
    tmp = svf + svfveg - 1.0
    tmp[tmp < 0.0] = 0.0
    svfalfa = torch.asin(torch.exp(torch.log(1.0 - tmp) / 2.0))
    # Prepare lists to store results for all time steps
    UTCI_all  = []
    TMRT_all  = []
    Kup_all   = []
    Kdown_all = []
    Lup_all   = []
    Ldown_all = []
    Shadow_all= []
    CI = 1.0
    for i in np.arange(0, Ta.__len__()):
        if landcover == 1:
            if ((dectime[i] - np.floor(dectime[i]))) == 0 or (i == 0):
                Ta_      = Ta.cpu().numpy()  # Added
                Twater = np.mean(Ta_[jday[0] == np.floor(dectime[i])])  # Added
        if (dectime[i] - np.floor(dectime[i])) == 0:
            daylines = np.where(np.floor(dectime) == dectime[i])
            if daylines.__len__() > 1:
                alt = altitude[0][daylines]
                alt2 = np.where(alt > 1)
                rise = alt2[0][0]
                [_, CI, _, _, _] = clearnessindex_2013b(zen[0, i + rise + 1], jday[0, i + rise + 1], Ta[i + rise + 1],
                                                        RH[i + rise + 1] / 100., radG[i + rise + 1], location, P[i + rise + 1])
                if (CI > 1.) or (CI == np.inf):
                    CI = 1.
            else:
                CI = 1.
        Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, shadow, firstdaytime, timestepdec, timeadd, \
        Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, Keast, Ksouth, Kwest, Knorth, Least, Lsouth, Lwest, Lnorth, \
        KsideI, TgOut1, TgOut, radIout, radDout, Lside, Lsky_patch_characteristics, CI_Tg, CI_TgG, KsideD, dRad, Kside = Solweig_2022a_calc(
            i, a, scale, rows, cols, svf, svfN, svfW, svfE, svfS, svfveg, svfNveg, svfEveg, svfSveg, svfWveg, svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg, vegdsm, vegdsm2, albedo_b, absK, absL, ewall, Fside, Fup, Fcyl,
            altitude[0][i], azimuth[0][i], zen[0][i], jday[0][i], usevegdem, onlyglobal, buildings, location, psi[0][i], landcover, lcgrid, dectime[i], altmax[0][i], dirwalls, walls, cyl, elvis, Ta[i], RH[i], radG[i], radD[i], radI[i], P[i],
            amaxvalue, bush, Twater, TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST, TmaxLST_wall, first, second, svfalfa, svfbuveg, firstdaytime, timeadd, timestepdec, Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N,
            CI, TgOut1, diffsh, shmat, vegshmat, vbshvegshmat, anisotropic_sky, asvf, patch_option)
        # Create matrices for meteorological parameters for the current time step
        Ta_mat = torch.zeros((rows, cols), device=device) + Ta[i]
        RH_mat = torch.zeros((rows, cols), device=device) + RH[i]
        Tmrt_mat = torch.zeros((rows, cols), device=device) + Tmrt
        va10m_mat = torch.zeros((rows, cols), device=device) + Ws[i]
        UTCI_mat = utci_calculator(Ta_mat, RH_mat, Tmrt_mat, va10m_mat)
        UTCI = torch.full(UTCI_mat.shape, float('nan'), device=device)
        UTCI[valid_mask] = UTCI_mat[valid_mask]
        # Append results (converted to CPU numpy arrays) to the lists
        UTCI_all.append(UTCI.cpu().numpy())
        TMRT_all.append(Tmrt.cpu().numpy())
        Kup_all.append(Kup.cpu().numpy())
        Kdown_all.append(Kdown.cpu().numpy())
        Lup_all.append(Lup.cpu().numpy())
        Ldown_all.append(Ldown.cpu().numpy())
        Shadow_all.append(shadow.cpu().numpy())
    # Convert the lists to numpy arrays with shape (time_steps, rows, cols)
    UTCI_all  = np.array(UTCI_all)
    TMRT_all  = np.array(TMRT_all)
    Kup_all   = np.array(Kup_all)
    Kdown_all = np.array(Kdown_all)
    Lup_all   = np.array(Lup_all)
    Ldown_all = np.array(Ldown_all)
    Shadow_all= np.array(Shadow_all)
    # Write a multi-band GeoTIFF for UTCI (each band corresponds to one time step)
    driver = gdal.GetDriverByName('GTiff')
    out_file_path = os.path.join(output_path, f'UTCI_{number}.tif')
    num_bands = UTCI_all.shape[0]
    out_dataset = driver.Create(out_file_path, cols, rows, num_bands, gdal.GDT_Float32)
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset.SetProjection(dataset.GetProjection())
    for band in range(num_bands):
        out_band = out_dataset.GetRasterBand(band + 1)
        out_band.WriteArray(UTCI_all[band])
        out_band.FlushCache()
        hour = int(hours[band].cpu().item())
        minute = int(minu[band].cpu().item())
        timestamp = base_date.replace(hour=hour, minute=minute).isoformat()
        out_band.SetMetadata({'Time': timestamp})
    out_dataset = None
    # Optionally, you can similarly write TMRT to a single multi-band file:
    if save_tmrt:
        out_file_path_op = os.path.join(output_path, f'TMRT_{number}.tif')
        num_bands_op = TMRT_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(TMRT_all[band])
            out_band.FlushCache()
            hour = int(hours[band].cpu().item())
            minute = int(minu[band].cpu().item())
            timestamp = base_date.replace(hour=hour, minute=minute).isoformat()
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None
    if save_svf:
        out_file_path_op = os.path.join(output_path, f'SVF_{number}.tif')
        SVF = svftotal.cpu().numpy()
        SVF = np.array(SVF)
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, 1, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        out_band = out_dataset_op.GetRasterBand(1)
        out_band.WriteArray(SVF)
        out_band.FlushCache()
        out_dataset_op = None
    if save_kup:
        out_file_path_op = os.path.join(output_path, f'Kup_{number}.tif')
        num_bands_op = Kup_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(Kup_all[band])
            out_band.FlushCache()
            hour = int(hours[band].cpu().item())
            minute = int(minu[band].cpu().item())
            timestamp = base_date.replace(hour=hour, minute=minute).isoformat()
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None
    if save_kdown:
        out_file_path_op = os.path.join(output_path, f'Kdown_{number}.tif')
        num_bands_op = Kdown_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(Kdown_all[band])
            out_band.FlushCache()
            hour = int(hours[band].cpu().item())
            minute = int(minu[band].cpu().item())
            timestamp = base_date.replace(hour=hour, minute=minute).isoformat()
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None
    if save_lup:
        out_file_path_op = os.path.join(output_path, f'Lup_{number}.tif')
        num_bands_op = Lup_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(Lup_all[band])
            out_band.FlushCache()
            hour = int(hours[band].cpu().item())
            minute = int(minu[band].cpu().item())
            timestamp = base_date.replace(hour=hour, minute=minute).isoformat()
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None
    if save_ldown:
        out_file_path_op = os.path.join(output_path, f'Ldown_{number}.tif')
        num_bands_op = Ldown_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(Ldown_all[band])
            out_band.FlushCache()
            hour = int(hours[band].cpu().item())
            minute = int(minu[band].cpu().item())
            timestamp = base_date.replace(hour=hour, minute=minute).isoformat()
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None
    if save_shadow:
        out_file_path_op = os.path.join(output_path, f'Shadow_{number}.tif')
        num_bands_op = Shadow_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(Shadow_all[band])
            out_band.FlushCache()
            hour = int(hours[band].cpu().item())
            minute = int(minu[band].cpu().item())
            timestamp = base_date.replace(hour=hour, minute=minute).isoformat()
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None

    # Clean up datasets
    dataset = None
    dataset2 = None
    dataset3 = None
    dataset4 = None
    dataset5 = None
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken to execute tile {number}: {time_taken:.2f} seconds")



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
from .shadow import create_patches
gdal.UseExceptions()

def ensure_tensor(x, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, device=device)
    return x

def daylen(DOY, XLAT):
    RAD = torch.pi / 180.0

    DEC = -23.45 * torch.cos(2.0 * torch.pi * (DOY + 10.0) / 365.0)

    SOC = torch.tan(RAD * DEC) * torch.tan(RAD * XLAT)
    SOC = torch.clamp(SOC, -1.0, 1.0)

    DAYL = 12.0 + 24.0 * torch.arcsin(SOC) / torch.pi
    SNUP = 12.0 - DAYL / 2.0
    SNDN = 12.0 + DAYL / 2.0

    return DAYL, DEC, SNDN, SNUP

def sunonsurface_2018a(azimuthA, scale, buildings, shadow, sunwall, first, second, aspect, walls, Tg, Tgwall, Ta,
                       emis_grid, ewall, alb_grid, SBC, albedo_b, Twater, lc_grid, landcover):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert inputs to tensors on the GPU

    scale = torch.tensor(scale, device=device).clone().detach()
    ewall = torch.tensor(ewall, device=device).clone().detach()
    albedo_b = torch.tensor(albedo_b, device=device).clone().detach()
    landcover = torch.tensor(landcover, device=device).clone().detach()

    sizex = walls.shape[0]
    sizey = walls.shape[1]

    wallbol = (walls > 0).float()
    sunwall[sunwall > 0] = 1

    azimuth = azimuthA * (torch.pi / 180)

    index = 0
    f = buildings
    Lup = SBC * emis_grid * (Tg * shadow + Ta + 273.15) ** 4 - SBC * emis_grid * (Ta + 273.15) ** 4
    if landcover == 1:
        Tg[lc_grid == 3] = (Twater - Ta).float()

    Lwall = SBC * ewall * (Tgwall + Ta + 273.15) ** 4 - SBC * ewall * (Ta + 273.15) ** 4
    albshadow = alb_grid * shadow
    alb = alb_grid

    tempsh = torch.zeros((sizex, sizey), device=device)
    tempbu = torch.zeros((sizex, sizey), device=device)
    tempbub = torch.zeros((sizex, sizey), device=device)
    tempbubwall = torch.zeros((sizex, sizey), device=device)
    tempwallsun = torch.zeros((sizex, sizey), device=device)
    weightsumsh = torch.zeros((sizex, sizey), device=device)
    weightsumwall = torch.zeros((sizex, sizey), device=device)
    first = torch.round(first * scale)
    if first < 1:
        first = 1
    second = torch.round(second * scale)
    weightsumLupsh = torch.zeros((sizex, sizey), device=device)
    weightsumLwall = torch.zeros((sizex, sizey), device=device)
    weightsumalbsh = torch.zeros((sizex, sizey), device=device)
    weightsumalbwall = torch.zeros((sizex, sizey), device=device)
    weightsumalbnosh = torch.zeros((sizex, sizey), device=device)
    weightsumalbwallnosh = torch.zeros((sizex, sizey), device=device)
    tempLupsh = torch.zeros((sizex, sizey), device=device)
    tempalbsh = torch.zeros((sizex, sizey), device=device)
    tempalbnosh = torch.zeros((sizex, sizey), device=device)

    pibyfour = torch.pi / 4
    threetimespibyfour = 3 * pibyfour
    fivetimespibyfour = 5 * pibyfour
    seventimespibyfour = 7 * pibyfour
    sinazimuth = torch.sin(azimuth)
    cosazimuth = torch.cos(azimuth)
    tanazimuth = torch.tan(azimuth)
    signsinazimuth = torch.sign(sinazimuth)
    signcosazimuth = torch.sign(cosazimuth)

    for n in torch.arange(0, second, device=device):
        if (pibyfour <= azimuth and azimuth < threetimespibyfour) or (fivetimespibyfour <= azimuth and azimuth < seventimespibyfour):
            dy = signsinazimuth * index
            dx = -1 * signcosazimuth * torch.abs(torch.round(index / tanazimuth))
        else:
            dy = signsinazimuth * torch.abs(torch.round(index * tanazimuth))
            dx = -1 * signcosazimuth * index

        absdx = torch.abs(dx)
        absdy = torch.abs(dy)

        xc1 = ((dx + absdx) / 2).int()
        xc2 = (sizex + (dx - absdx) / 2).int()
        yc1 = ((dy + absdy) / 2).int()
        yc2 = (sizey + (dy - absdy) / 2).int()

        xp1 = -((dx - absdx) / 2).int()
        xp2 = (sizex - (dx + absdx) / 2).int()
        yp1 = -((dy - absdy) / 2).int()
        yp2 = (sizey - (dy + absdy) / 2).int()

        tempbu[xp1:xp2, yp1:yp2] = buildings[xc1:xc2, yc1:yc2]
        tempsh[xp1:xp2, yp1:yp2] = shadow[xc1:xc2, yc1:yc2]
        tempLupsh[xp1:xp2, yp1:yp2] = Lup[xc1:xc2, yc1:yc2]
        tempalbsh[xp1:xp2, yp1:yp2] = albshadow[xc1:xc2, yc1:yc2]
        tempalbnosh[xp1:xp2, yp1:yp2] = alb[xc1:xc2, yc1:yc2]
        f = torch.min(f, tempbu)

        shadow2 = tempsh * f
        weightsumsh += shadow2

        Lupsh = tempLupsh * f
        weightsumLupsh += Lupsh

        albsh = tempalbsh * f
        weightsumalbsh += albsh

        albnosh = tempalbnosh * f
        weightsumalbnosh += albnosh

        tempwallsun[xp1:xp2, yp1:yp2] = sunwall[xc1:xc2, yc1:yc2]
        tempb = tempwallsun * f
        tempbwall = f * -1 + 1
        tempbub = ((tempb + tempbub) > 0).float()
        tempbubwall = ((tempbwall + tempbubwall) > 0).float()
        weightsumLwall += tempbub * Lwall
        weightsumalbwall += tempbub * albedo_b
        weightsumwall += tempbub
        weightsumalbwallnosh += tempbubwall * albedo_b

        ind = 1
        if (n + 1) <= first:
            weightsumwall_first = weightsumwall / ind
            weightsumsh_first = weightsumsh / ind
            wallsuninfluence_first = weightsumwall_first > 0
            weightsumLwall_first = weightsumLwall / ind
            weightsumLupsh_first = weightsumLupsh / ind

            weightsumalbwall_first = weightsumalbwall / ind
            weightsumalbsh_first = weightsumalbsh / ind
            weightsumalbwallnosh_first = weightsumalbwallnosh / ind
            weightsumalbnosh_first = weightsumalbnosh / ind
            wallinfluence_first = weightsumalbwallnosh_first > 0
            ind += 1
        index += 1

    wallsuninfluence_second = weightsumwall > 0
    wallinfluence_second = weightsumalbwallnosh > 0

    azilow = azimuth - torch.pi / 2
    azihigh = azimuth + torch.pi / 2
    if azilow >= 0 and azihigh < 2 * torch.pi:
        facesh = (torch.logical_or(aspect < azilow, aspect >= azihigh).float() - wallbol + 1)
    elif azilow < 0 and azihigh <= 2 * torch.pi:
        azilow = azilow + 2 * torch.pi
        facesh = torch.logical_or(aspect > azilow, aspect <= azihigh).float() * -1 + 1
    elif azilow > 0 and azihigh >= 2 * torch.pi:
        azihigh = azihigh - 2 * torch.pi
        facesh = torch.logical_or(aspect > azilow, aspect <= azihigh).float() * -1 + 1

    keep = (weightsumwall == second).float() - facesh
    keep[keep == -1] = 0

    gvf1 = ((weightsumwall_first + weightsumsh_first) / (first + 1)) * wallsuninfluence_first + \
           (weightsumsh_first) / first * (wallsuninfluence_first * -1 + 1)
    weightsumwall[keep == 1] = 0
    gvf2 = ((weightsumwall + weightsumsh) / (second + 1)) * wallsuninfluence_second + \
           (weightsumsh) / second * (wallsuninfluence_second * -1 + 1)

    gvf2[gvf2 > 1.] = 1.

    gvfLup1 = ((weightsumLwall_first + weightsumLupsh_first) / (first + 1)) * wallsuninfluence_first + \
              (weightsumLupsh_first) / first * (wallsuninfluence_first * -1 + 1)
    weightsumLwall[keep == 1] = 0
    gvfLup2 = ((weightsumLwall + weightsumLupsh) / (second + 1)) * wallsuninfluence_second + \
              (weightsumLupsh) / second * (wallsuninfluence_second * -1 + 1)

    gvfalb1 = ((weightsumalbwall_first + weightsumalbsh_first) / (first + 1)) * wallsuninfluence_first + \
              (weightsumalbsh_first) / first * (wallsuninfluence_first * -1 + 1)
    weightsumalbwall[keep == 1] = 0
    gvfalb2 = ((weightsumalbwall + weightsumalbsh) / (second + 1)) * wallsuninfluence_second + \
              (weightsumalbsh) / second * (wallsuninfluence_second * -1 + 1)

    gvfalbnosh1 = ((weightsumalbwallnosh_first + weightsumalbnosh_first) / (first + 1)) * wallinfluence_first + \
                  (weightsumalbnosh_first) / first * (wallinfluence_first * -1 + 1)
    gvfalbnosh2 = ((weightsumalbwallnosh + weightsumalbnosh) / second) * wallinfluence_second + \
                  (weightsumalbnosh) / second * (wallinfluence_second * -1 + 1)

    gvf = (gvf1 * 0.5 + gvf2 * 0.4) / 0.9
    gvfLup = (gvfLup1 * 0.5 + gvfLup2 * 0.4) / 0.9
    gvfLup = gvfLup + ((SBC * emis_grid * (Tg * shadow + Ta + 273.15) ** 4) - SBC * emis_grid * (Ta + 273.15) ** 4) * (
                buildings * -1 + 1)
    gvfalb = (gvfalb1 * 0.5 + gvfalb2 * 0.4) / 0.9
    gvfalb = gvfalb + alb_grid * (buildings * -1 + 1) * shadow
    gvfalbnosh = (gvfalbnosh1 * 0.5 + gvfalbnosh2 * 0.4) / 0.9
    gvfalbnosh = gvfalbnosh * buildings + alb_grid * (buildings * -1 + 1)

    del tempbu,tempsh,tempLupsh,tempalbsh,tempalbnosh,shadow2,Lupsh,albsh,albnosh,tempwallsun,tempb,tempbwall,tempbub,tempbubwall

    del weightsumLupsh ,weightsumLwall ,weightsumalbsh ,weightsumalbwall ,weightsumalbnosh ,weightsumalbwallnosh,weightsumsh ,weightsumwall

    return gvf, gvfLup, gvfalb, gvfalbnosh, gvf2


def gvf_2018a(wallsun, walls, buildings, scale, shadow, first, second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall,
              alb_grid, SBC, albedo_b, rows, cols, Twater, lc_grid, landcover):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    azimuthA = torch.arange(5, 359, 20, device=device, dtype=torch.float32)  # Search directions for Ground View Factors (GVF)

    gvfLup = torch.zeros((rows, cols), device=device)
    gvfalb = torch.zeros((rows, cols), device=device)
    gvfalbnosh = torch.zeros((rows, cols), device=device)
    gvfLupE = torch.zeros((rows, cols), device=device)
    gvfLupS = torch.zeros((rows, cols), device=device)
    gvfLupW = torch.zeros((rows, cols), device=device)
    gvfLupN = torch.zeros((rows, cols), device=device)
    gvfalbE = torch.zeros((rows, cols), device=device)
    gvfalbS = torch.zeros((rows, cols), device=device)
    gvfalbW = torch.zeros((rows, cols), device=device)
    gvfalbN = torch.zeros((rows, cols), device=device)
    gvfalbnoshE = torch.zeros((rows, cols), device=device)
    gvfalbnoshS = torch.zeros((rows, cols), device=device)
    gvfalbnoshW = torch.zeros((rows, cols), device=device)
    gvfalbnoshN = torch.zeros((rows, cols), device=device)
    gvfSum = torch.zeros((rows, cols), device=device)

    sunwall = ((wallsun / walls * buildings) == 1).float()

    for j in torch.arange(0, len(azimuthA), device=device):
        _, gvfLupi, gvfalbi, gvfalbnoshi, gvf2 = sunonsurface_2018a(
            azimuthA[j], scale, buildings, shadow, sunwall, first, second, dirwalls * torch.pi / 180, walls, Tg, Tgwall,
            Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, Twater, lc_grid, landcover
        )

        gvfLup += gvfLupi
        gvfalb += gvfalbi
        gvfalbnosh += gvfalbnoshi
        gvfSum += gvf2

        if 0 <= azimuthA[j] < 180:
            gvfLupE += gvfLupi
            gvfalbE += gvfalbi
            gvfalbnoshE += gvfalbnoshi

        if 90 <= azimuthA[j] < 270:
            gvfLupS += gvfLupi
            gvfalbS += gvfalbi
            gvfalbnoshS += gvfalbnoshi

        if 180 <= azimuthA[j] < 360:
            gvfLupW += gvfLupi
            gvfalbW += gvfalbi
            gvfalbnoshW += gvfalbnoshi

        if 270 <= azimuthA[j] or azimuthA[j] < 90:
            gvfLupN += gvfLupi
            gvfalbN += gvfalbi
            gvfalbnoshN += gvfalbnoshi

    gvfLup = gvfLup / len(azimuthA) + SBC * emis_grid * (Ta + 273.15) ** 4
    gvfalb = gvfalb / len(azimuthA)
    gvfalbnosh = gvfalbnosh / len(azimuthA)

    gvfLupE = gvfLupE / (len(azimuthA) / 2) + SBC * emis_grid * (Ta + 273.15) ** 4
    gvfLupS = gvfLupS / (len(azimuthA) / 2) + SBC * emis_grid * (Ta + 273.15) ** 4
    gvfLupW = gvfLupW / (len(azimuthA) / 2) + SBC * emis_grid * (Ta + 273.15) ** 4
    gvfLupN = gvfLupN / (len(azimuthA) / 2) + SBC * emis_grid * (Ta + 273.15) ** 4

    gvfalbE = gvfalbE / (len(azimuthA) / 2)
    gvfalbS = gvfalbS / (len(azimuthA) / 2)
    gvfalbW = gvfalbW / (len(azimuthA) / 2)
    gvfalbN = gvfalbN / (len(azimuthA) / 2)

    gvfalbnoshE = gvfalbnoshE / (len(azimuthA) / 2)
    gvfalbnoshS = gvfalbnoshS / (len(azimuthA) / 2)
    gvfalbnoshW = gvfalbnoshW / (len(azimuthA) / 2)
    gvfalbnoshN = gvfalbnoshN / (len(azimuthA) / 2)

    gvfNorm = gvfSum / len(azimuthA)
    gvfNorm[buildings == 0] = 1

    return (
        gvfLup, gvfalb, gvfalbnosh, gvfLupE, gvfalbE,
        gvfalbnoshE, gvfLupS, gvfalbS, gvfalbnoshS, gvfLupW,
        gvfalbW, gvfalbnoshW, gvfLupN, gvfalbN, gvfalbnoshN,
        gvfSum, gvfNorm
    )

def cylindric_wedge(zen, svfalfa, rows, cols):
    np.seterr(divide='ignore', invalid='ignore')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    beta = torch.tensor(zen, device=device, dtype=torch.float32)
    alfa = torch.zeros((rows, cols), device=device) + svfalfa

    xa = 1 - 2. / (torch.tan(alfa) * torch.tan(beta))
    ha = 2. / (torch.tan(alfa) * torch.tan(beta))
    ba = (1. / torch.tan(alfa))
    hkil = 2. * ba * ha

    qa = torch.zeros((rows, cols), device=device)
    qa[xa < 0] = torch.tan(beta) / 2

    Za = torch.zeros((rows, cols), device=device)
    Za[xa < 0] = (ba[xa < 0] ** 2 - (qa[xa < 0] ** 2) / 4) ** 0.5

    phi = torch.zeros((rows, cols), device=device)
    phi[xa < 0] = torch.atan(Za[xa < 0] / qa[xa < 0])

    A = torch.zeros((rows, cols), device=device)
    A[xa < 0] = (torch.sin(phi[xa < 0]) - phi[xa < 0] * torch.cos(phi[xa < 0])) / (1 - torch.cos(phi[xa < 0]))

    ukil = torch.zeros((rows, cols), device=device)
    ukil[xa < 0] = 2 * ba[xa < 0] * xa[xa < 0] * A[xa < 0]

    Ssurf = hkil + ukil

    F_sh = (2 * torch.pi * ba - Ssurf) / (2 * torch.pi * ba)

    del alfa, beta, hkil, ukil, phi, A, Ssurf, qa, Za

    return F_sh

def TsWaveDelay_2015a(gvfLup, firstdaytime, timeadd, timestepdec, Tgmap1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # gvfLup = torch.tensor(gvfLup, device=device)
    Tgmap0 = gvfLup  # current timestep

    if firstdaytime == 1:  # "first in morning"
        Tgmap1 = Tgmap0

    if timeadd >= (59 / 1440):  # more or equal to 59 min
        weight1 = torch.exp(-33.27 * torch.tensor(timeadd))  # surface temperature delay function - 1 step
        Tgmap1 = Tgmap0 * (1 - weight1) + Tgmap1 * weight1
        Lup = Tgmap1
        if timestepdec > (59 / 1440):
            timeadd = timestepdec
        else:
            timeadd = 0
    else:
        timeadd = timeadd + timestepdec
        weight1 = torch.exp(-33.27 * torch.tensor(timeadd))  # surface temperature delay function - 1 step
        Lup = (Tgmap0 * (1 - weight1) + Tgmap1 * weight1)

    return Lup, timeadd, Tgmap1

def Kup_veg_2015a(radI, radD, radG, altitude, svfbuveg, albedo_b, F_sh, gvfalb, gvfalbE, gvfalbS, gvfalbW, gvfalbN, gvfalbnosh, gvfalbnoshE, gvfalbnoshS, gvfalbnoshW, gvfalbnoshN):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    albedo_b = torch.tensor(albedo_b, device=device).clone().detach()
    Kup = (gvfalb * radI * torch.sin(altitude * (torch.pi / 180.))) + (radD * svfbuveg + albedo_b * (1 - svfbuveg) * (radG * (1 - F_sh) + radD * F_sh)) * gvfalbnosh
    KupE = (gvfalbE * radI * torch.sin(altitude * (torch.pi / 180.))) + (radD * svfbuveg + albedo_b * (1 - svfbuveg) * (radG * (1 - F_sh) + radD * F_sh)) * gvfalbnoshE
    KupS = (gvfalbS * radI * torch.sin(altitude * (torch.pi / 180.))) + (radD * svfbuveg + albedo_b * (1 - svfbuveg) * (radG * (1 - F_sh) + radD * F_sh)) * gvfalbnoshS
    KupW = (gvfalbW * radI * torch.sin(altitude * (torch.pi / 180.))) + (radD * svfbuveg + albedo_b * (1 - svfbuveg) * (radG * (1 - F_sh) + radD * F_sh)) * gvfalbnoshW
    KupN = (gvfalbN * radI * torch.sin(altitude * (torch.pi / 180.))) + (radD * svfbuveg + albedo_b * (1 - svfbuveg) * (radG * (1 - F_sh) + radD * F_sh)) * gvfalbnoshN

    return Kup, KupE, KupS, KupW, KupN

def Kvikt_veg(svf, svfveg, vikttot):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    viktwall = (vikttot - (63.227 * svf ** 6 - 161.51 * svf ** 5 + 156.91 * svf ** 4 - 70.424 * svf ** 3 + 16.773 * svf ** 2 - 0.4863 * svf)) / vikttot
    svfvegbu = (svfveg + svf - 1)
    viktveg = (vikttot - (63.227 * svfvegbu ** 6 - 161.51 * svfvegbu ** 5 + 156.91 * svfvegbu ** 4 - 70.424 * svfvegbu ** 3 + 16.773 * svfvegbu ** 2 - 0.4863 * svfvegbu)) / vikttot
    viktveg = viktveg - viktwall

    del svfvegbu

    return viktveg, viktwall


def shaded_or_sunlit(solar_altitude, solar_azimuth, patch_altitude, patch_azimuth, asvf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Patch azimuth in relation to sun azimuth
    patch_to_sun_azi = torch.abs(solar_azimuth - patch_azimuth)

    # Degrees to radians
    deg2rad = torch.pi / 180.0

    # Radians to degrees
    rad2deg = 180.0 / torch.pi
    xi = torch.cos(patch_to_sun_azi * deg2rad)
    yi = 2 * xi * torch.tan(solar_altitude * deg2rad)
    hsvf = torch.tan(asvf)
    yi_ = torch.where(yi > 0, 0.0, yi)
    tan_delta = hsvf + yi_

    # Degrees where below is in shade and above is sunlit
    sunlit_degrees = torch.atan(tan_delta) * rad2deg

    # Boolean for pixels where patch is sunlit
    sunlit_patches = sunlit_degrees < patch_altitude
    # Boolean for pixels where patch is shaded
    shaded_patches = sunlit_degrees > patch_altitude

    return sunlit_patches, shaded_patches

def Kside_veg_v2022a(radI, radD, radG, shadow, svfS, svfW, svfN, svfE, svfEveg, svfSveg, svfWveg, svfNveg,
                     azimuth, altitude, psi, t, albedo, F_sh, KupE, KupS, KupW, KupN, cyl, lv, anisotropic_diffuse,
                     diffsh, rows, cols, asvf, shmat, vegshmat, vbshvegshmat):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vikttot = 4.4897
    aziE = azimuth + t
    aziS = azimuth - 90 + t
    aziW = azimuth - 180 + t
    aziN = azimuth - 270 + t
    deg2rad = torch.pi / 180.0
    deg2rad = torch.tensor(deg2rad)

    KsideD = torch.zeros((rows, cols), device=device)
    Kref_sun = torch.zeros((rows, cols), device=device)
    Kref_sh = torch.zeros((rows, cols), device=device)
    Kref_veg = torch.zeros((rows, cols), device=device)
    Kside = torch.zeros((rows, cols), device=device)

    Kref_veg_n = torch.zeros((rows, cols), device=device)
    Kref_veg_s = torch.zeros((rows, cols), device=device)
    Kref_veg_e = torch.zeros((rows, cols), device=device)
    Kref_veg_w = torch.zeros((rows, cols), device=device)

    Kref_sh_n = torch.zeros((rows, cols), device=device)
    Kref_sh_s = torch.zeros((rows, cols), device=device)
    Kref_sh_e = torch.zeros((rows, cols), device=device)
    Kref_sh_w = torch.zeros((rows, cols), device=device)

    Kref_sun_n = torch.zeros((rows, cols), device=device)
    Kref_sun_s = torch.zeros((rows, cols), device=device)
    Kref_sun_e = torch.zeros((rows, cols), device=device)
    Kref_sun_w = torch.zeros((rows, cols), device=device)

    KeastRef = torch.zeros((rows, cols), device=device)
    KwestRef = torch.zeros((rows, cols), device=device)
    KnorthRef = torch.zeros((rows, cols), device=device)
    KsouthRef = torch.zeros((rows, cols), device=device)
    diffRadE = torch.zeros((rows, cols), device=device)
    diffRadS = torch.zeros((rows, cols), device=device)
    diffRadW = torch.zeros((rows, cols), device=device)
    diffRadN = torch.zeros((rows, cols), device=device)

    altitude = torch.tensor(altitude)
    if cyl == 1:
        KsideI = shadow * radI * torch.cos(altitude * deg2rad)
        KeastI = torch.zeros((rows, cols), device=device)
        KsouthI = torch.zeros((rows, cols), device=device)
        KwestI = torch.zeros((rows, cols), device=device)
        KnorthI = torch.zeros((rows, cols), device=device)
    else:
        KeastI = torch.where((azimuth > (360 - t)) | (azimuth <= (180 - t)),
                             radI * shadow * torch.cos(altitude * deg2rad) * torch.sin(aziE * deg2rad),
                             torch.zeros((rows, cols), device=device))
        KsouthI = torch.where((azimuth > (90 - t)) & (azimuth <= (270 - t)),
                              radI * shadow * torch.cos(altitude * deg2rad) * torch.sin(aziS * deg2rad),
                              torch.zeros((rows, cols), device=device))
        KwestI = torch.where((azimuth > (180 - t)) & (azimuth <= (360 - t)),
                             radI * shadow * torch.cos(altitude * deg2rad) * torch.sin(aziW * deg2rad),
                             torch.zeros((rows, cols), device=device))
        KnorthI = torch.where((azimuth <= (90 - t)) | (azimuth > (270 - t)),
                              radI * shadow * torch.cos(altitude * deg2rad) * torch.sin(aziN * deg2rad),
                              torch.zeros((rows, cols), device=device))
        KsideI = shadow * 0

    viktveg, viktwall = Kvikt_veg(svfE, svfEveg, vikttot)
    svfviktbuvegE = viktwall + (viktveg * (1 - psi))

    viktveg, viktwall = Kvikt_veg(svfS, svfSveg, vikttot)
    svfviktbuvegS = viktwall + (viktveg * (1 - psi))

    viktveg, viktwall = Kvikt_veg(svfW, svfWveg, vikttot)
    svfviktbuvegW = viktwall + (viktveg * (1 - psi))

    viktveg, viktwall = Kvikt_veg(svfN, svfNveg, vikttot)
    svfviktbuvegN = viktwall + (viktveg * (1 - psi))

    if anisotropic_diffuse == 1:
        anisotropic_sky = True

        patch_altitude = lv[:,0] 
        patch_azimuth = lv[:,1] 
        if anisotropic_sky:
            patch_luminance = lv[:,2] 
        else:
            patch_luminance = torch.ones((patch_altitude.shape[0]), device=device) / patch_altitude.shape[0]

        skyalt, skyalt_c = torch.unique(patch_altitude, return_counts=True)
        radTot = torch.zeros(1, device=device)
        steradian = torch.zeros((patch_altitude.shape[0]), device=device)
        for i in range(patch_altitude.shape[0]):
            if skyalt_c[skyalt == patch_altitude[i]] > 1:
                steradian[i] = ((360 / skyalt_c[skyalt == patch_altitude[i]]) * deg2rad) * (
                    torch.sin((patch_altitude[i] + patch_altitude[0]) * deg2rad) - torch.sin((patch_altitude[i] - patch_altitude[0]) * deg2rad))
            else:
                steradian[i] = ((360 / skyalt_c[skyalt == patch_altitude[i]]) * deg2rad) * (
                    torch.sin((patch_altitude[i]) * deg2rad) - torch.sin((patch_altitude[i - 1] + patch_altitude[0]) * deg2rad))

            radTot += (patch_luminance[i] * steradian[i] * torch.sin(patch_altitude[i] * deg2rad))

        lumChi = (patch_luminance * radD) / radTot

        if cyl == 1:
            for idx in range(patch_azimuth.shape[0]):
                anglIncC = torch.cos(patch_altitude[idx] * deg2rad) * torch.cos(torch.tensor(0.))
                KsideD += diffsh[:, :, idx] * lumChi[idx] * anglIncC * steradian[idx]

                sunlit_surface = ((albedo * (radI * torch.cos(altitude * deg2rad)) + (radD * 0.5)) / torch.pi)
                shaded_surface = ((albedo * radD * 0.5) / torch.pi)

                temp_vegsh = ((vegshmat[:,:,idx] == 0) | (vbshvegshmat[:,:,idx] == 0))
                Kref_veg += shaded_surface * temp_vegsh * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad)

                temp_vbsh = (1 - shmat[:,:,idx]) * vbshvegshmat[:,:,idx]
                temp_sh = (temp_vbsh == 1)

                sunlit_patches, shaded_patches = shaded_or_sunlit(altitude, azimuth, patch_altitude[idx], patch_azimuth[idx], asvf)
                Kref_sun += sunlit_surface * sunlit_patches * temp_sh * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad)
                Kref_sh += shaded_surface * shaded_patches * temp_sh * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad)

            Kside = KsideI + KsideD + Kref_sun + Kref_sh + Kref_veg

            Keast = KupE * 0.5
            Kwest = KupW * 0.5
            Knorth = KupN * 0.5
            Ksouth = KupS * 0.5

        else:
            for idx in range(patch_azimuth.shape[0]):
                if (patch_azimuth[idx] > 360) or (patch_azimuth[idx] <= 180):
                    anglIncE = torch.cos(patch_altitude[idx] * deg2rad) * torch.cos((90 - patch_azimuth[idx] + t) * deg2rad)
                    diffRadE += diffsh[:, :, idx] * lumChi[idx] * anglIncE * steradian[idx]

                if (patch_azimuth[idx] > 90) and (patch_azimuth[idx] <= 270):
                    anglIncS = torch.cos(patch_altitude[idx] * deg2rad) * torch.cos((180 - patch_azimuth[idx] + t) * deg2rad)
                    diffRadS += diffsh[:, :, idx] * lumChi[idx] * anglIncS * steradian[idx]

                if (patch_azimuth[idx] > 180) and (patch_azimuth[idx] <= 360):
                    anglIncW = torch.cos(patch_altitude[idx] * deg2rad) * torch.cos((270 - patch_azimuth[idx] + t) * deg2rad)
                    diffRadW += diffsh[:, :, idx] * lumChi[idx] * anglIncW * steradian[idx]

                if (patch_azimuth[idx] > 270) or (patch_azimuth[idx] <= 90):
                    anglIncN = torch.cos(patch_altitude[idx] * deg2rad) * torch.cos((0 - patch_azimuth[idx] + t) * deg2rad)
                    diffRadN += diffsh[:, :, idx] * lumChi[idx] * anglIncN * steradian[idx]

                sunlit_surface = ((albedo * (radI * torch.cos(altitude * deg2rad)) + (radD * 0.5)) / torch.pi)
                shaded_surface = ((albedo * radD * 0.5) / torch.pi)

                temp_vegsh = ((vegshmat[:,:,idx] == 0) | (vbshvegshmat[:,:,idx] == 0))
                Kref_veg += shaded_surface * temp_vegsh * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad)

                if (patch_azimuth[idx] > 360) or (patch_azimuth[idx] < 180):
                    Kref_veg_e += shaded_surface * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_vegsh * torch.cos((90 - patch_azimuth[idx] + t) * deg2rad)
                if (patch_azimuth[idx] > 90) and (patch_azimuth[idx] < 270):
                    Kref_veg_s += shaded_surface * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_vegsh * torch.cos((180 - patch_azimuth[idx] + t) * deg2rad)
                if (patch_azimuth[idx] > 180) and (patch_azimuth[idx] < 360):
                    Kref_veg_w += shaded_surface * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_vegsh * torch.cos((270 - patch_azimuth[idx] + t) * deg2rad)
                if (patch_azimuth[idx] > 270) or (patch_azimuth[idx] < 90):
                    Kref_veg_n += shaded_surface * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_vegsh * torch.cos((0 - patch_azimuth[idx] + t) * deg2rad)

                temp_vbsh = (1 - shmat[:,:,idx]) * vbshvegshmat[:,:,idx]
                temp_sh = (temp_vbsh == 1)
                azimuth_difference = torch.abs(azimuth - patch_azimuth[idx])

                if (azimuth_difference > 90) and (azimuth_difference < 270):
                    sunlit_patches, shaded_patches = shaded_or_sunlit(altitude, azimuth, patch_altitude[idx], patch_azimuth[idx], asvf)
                    Kref_sun += sunlit_surface * sunlit_patches * temp_sh * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad)
                    Kref_sh += shaded_surface * shaded_patches * temp_sh * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad)

                    if (patch_azimuth[idx] > 360) or (patch_azimuth[idx] < 180):
                        Kref_sun_e += sunlit_surface * sunlit_patches * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((90 - patch_azimuth[idx] + t) * deg2rad)
                        Kref_sh_e += shaded_surface * shaded_patches * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((90 - patch_azimuth[idx] + t) * deg2rad)
                    if (patch_azimuth[idx] > 90) and (patch_azimuth[idx] < 270):
                        Kref_sun_s += sunlit_surface * sunlit_patches * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((180 - patch_azimuth[idx] + t) * deg2rad)
                        Kref_sh_s += shaded_surface * shaded_patches * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((180 - patch_azimuth[idx] + t) * deg2rad)
                    if (patch_azimuth[idx] > 180) and (patch_azimuth[idx] < 360):
                        Kref_sun_w += sunlit_surface * sunlit_patches * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((270 - patch_azimuth[idx] + t) * deg2rad)
                        Kref_sh_w += shaded_surface * shaded_patches * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((270 - patch_azimuth[idx] + t) * deg2rad)
                    if (patch_azimuth[idx] > 270) or (patch_azimuth[idx] < 90):
                        Kref_sun_n += sunlit_surface * sunlit_patches * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((0 - patch_azimuth[idx] + t) * deg2rad)
                        Kref_sh_n += shaded_surface * shaded_patches * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((0 - patch_azimuth[idx] + t) * deg2rad)
                else:
                    Kref_sh += shaded_surface * temp_sh * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad)

                    if (patch_azimuth[idx] > 360) or (patch_azimuth[idx] < 180):
                        Kref_sh_e += shaded_surface * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((90 - patch_azimuth[idx] + t) * deg2rad)
                    if (patch_azimuth[idx] > 90) and (patch_azimuth[idx] < 270):
                        Kref_sh_s += shaded_surface * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((180 - patch_azimuth[idx] + t) * deg2rad)
                    if (patch_azimuth[idx] > 180) and (patch_azimuth[idx] < 360):
                        Kref_sh_w += shaded_surface * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((270 - patch_azimuth[idx] + t) * deg2rad)
                    if (patch_azimuth[idx] > 270) or (patch_azimuth[idx] < 90):
                        Kref_sh_n += shaded_surface * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((0 - patch_azimuth[idx] + t) * deg2rad)

            Keast = KeastI + diffRadE + Kref_sun_e + Kref_sh_e + Kref_veg_e + KupE * 0.5
            Kwest = KwestI + diffRadW + Kref_sun_w + Kref_sh_w + Kref_veg_w + KupW * 0.5
            Knorth = KnorthI + diffRadN + Kref_sun_n + Kref_sh_n + Kref_veg_n + KupN * 0.5
            Ksouth = KsouthI + diffRadS + Kref_sun_s + Kref_sh_s + Kref_veg_s + KupS * 0.5

    else:
        KeastDG = (radD * (1 - svfviktbuvegE) + albedo * (svfviktbuvegE * (radG * (1 - F_sh) + radD * F_sh)) + KupE) * 0.5
        Keast = KeastI + KeastDG

        KsouthDG = (radD * (1 - svfviktbuvegS) + albedo * (svfviktbuvegS * (radG * (1 - F_sh) + radD * F_sh)) + KupS) * 0.5
        Ksouth = KsouthI + KsouthDG

        KwestDG = (radD * (1 - svfviktbuvegW) + albedo * (svfviktbuvegW * (radG * (1 - F_sh) + radD * F_sh)) + KupW) * 0.5
        Kwest = KwestI + KwestDG

        KnorthDG = (radD * (1 - svfviktbuvegN) + albedo * (svfviktbuvegN * (radG * (1 - F_sh) + radD * F_sh)) + KupN) * 0.5
        Knorth = KnorthI + KnorthDG

    del temp_vegsh,temp_vbsh,temp_sh
    del Kref_sun ,Kref_sh ,Kref_veg
    del Kref_veg_n,Kref_veg_s,Kref_veg_e ,Kref_veg_w
    del Kref_sh_n ,Kref_sh_s ,Kref_sh_e ,Kref_sh_w
    del Kref_sun_n ,Kref_sun_s ,Kref_sun_e ,Kref_sun_w
    del KeastRef ,KwestRef ,KnorthRef,KsouthRef ,diffRadE ,diffRadS ,diffRadW ,diffRadN
    del KeastI, KsouthI, KwestI, KnorthI, viktveg,viktwall,svfviktbuvegE,svfviktbuvegS,svfviktbuvegW,svfviktbuvegN,KupW,KupN

    return Keast, Ksouth, Kwest, Knorth, KsideI, KsideD, Kside


def sun_distance(jday):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    b = 2. * torch.pi * jday / 365.
    D = torch.sqrt(1.00011 + 0.034221 * torch.cos(b) + 0.001280 * torch.sin(b) + 0.000719 * torch.cos(2. * b) + 0.000077 * torch.sin(2. * b))
    return D

def clearnessindex_2013b(zen, jday, Ta, RH, radG, location, P):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if P == -999.0:
        p = 1013.0  # Pressure in millibars
    else:
        p = P * 10.0  # Convert from hPa to millibars

    Itoa = 1370.0  # Effective solar constant
    D = sun_distance(jday)
    m = 35.0 * torch.cos(zen) * ((1224.0 * (torch.cos(zen) ** 2) + 1.0) ** (-1 / 2.0))  # optical air mass at p=1013
    Trpg = 1.021 - 0.084 * (m * (0.000949 * p + 0.051)) ** 0.5  # Transmission coefficient for Rayleigh scattering and permanent gases

    # empirical constant depending on latitude
    latitude = location['latitude']
    if latitude < 10.0:
        G = [3.37, 2.85, 2.80, 2.64]
    elif 10.0 <= latitude < 20.0:
        G = [2.99, 3.02, 2.70, 2.93]
    elif 20.0 <= latitude < 30.0:
        G = [3.60, 3.00, 2.98, 2.93]
    elif 30.0 <= latitude < 40.0:
        G = [3.04, 3.11, 2.92, 2.94]
    elif 40.0 <= latitude < 50.0:
        G = [2.70, 2.95, 2.77, 2.71]
    elif 50.0 <= latitude < 60.0:
        G = [2.52, 3.07, 2.67, 2.93]
    elif 60.0 <= latitude < 70.0:
        G = [1.76, 2.69, 2.61, 2.61]
    elif 70.0 <= latitude < 80.0:
        G = [1.60, 1.67, 2.24, 2.63]
    elif 80.0 <= latitude < 90.0:
        G = [1.11, 1.44, 1.94, 2.02]

    if jday > 335 or jday <= 60:
        G = G[0]
    elif 60 < jday <= 152:
        G = G[1]
    elif 152 < jday <= 244:
        G = G[2]
    elif 244 < jday <= 335:
        G = G[3]

    # dewpoint calculation
    a2 = torch.tensor(17.27,device=device)
    b2 = torch.tensor(237.7,device=device)
    Td = (b2 * (((a2 * Ta) / (b2 + Ta)) + torch.log(RH))) / (a2 - (((a2 * Ta) / (b2 + Ta)) + torch.log(RH)))
    Td = (Td * 1.8) + 32  # Dewpoint (F)
    u = torch.exp(0.1133 - torch.log(torch.tensor(G + 1.)) + 0.0393 * Td) 
    Tw = 1 - 0.077 * ((u * m) ** 0.3)  # Transmission coefficient for water vapor
    Tar = 0.935 ** m  # Transmission coefficient for aerosols
    I0 = Itoa * torch.cos(zen) * Trpg * Tw * D * Tar
    I0 = torch.where(torch.abs(zen) > torch.pi / 2, torch.tensor(0.0, device=device), I0)
    I0 = torch.where(torch.isnan(I0), torch.tensor(0.0, device=device), I0)
    corr = 0.1473 * torch.log(90 - (zen / torch.pi * 180)) + 0.3454  # 20070329
    CIuncorr = radG / I0
    CI = CIuncorr + (1 - corr)
    I0et = Itoa * torch.cos(zen) * D  # extra terrestrial solar radiation
    Kt = radG / I0et

    return I0, CI, Kt, I0et, CIuncorr

def diffusefraction(radG, altitude, Kt, Ta, RH):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Ta = ensure_tensor(Ta)
    RH = ensure_tensor(RH)
    alfa = altitude * (torch.pi / 180.0)
    alfa = ensure_tensor(alfa)
    if Ta <= -999.00 or RH <= -999.00 or torch.isnan(Ta) or torch.isnan(RH): 
        if Kt <= 0.3:
            radD = radG * (1.020 - 0.248 * Kt)
        elif 0.3 < Kt < 0.78:
            radD = radG * (1.45 - 1.67 * Kt)
        else:
            radD = radG * 0.147
    else:
        RH = RH / 100
        if Kt <= 0.3:
            radD = radG * (1 - 0.232 * Kt + 0.0239 * torch.sin(alfa) - 0.000682 * Ta + 0.0195 * RH)
        elif 0.3 < Kt < 0.78:
            radD = radG * (1.329 - 1.716 * Kt + 0.267 * torch.sin(alfa) - 0.00357 * Ta + 0.106 * RH)
        else:
            radD = radG * (0.426 * Kt - 0.256 * torch.sin(alfa) + 0.00349 * Ta + 0.0734 * RH)

    radI = (radG - radD) / torch.sin(alfa)

    # Corrections for low sun altitudes (20130307)
    radI = torch.where(radI < 0, torch.tensor(0.0, device=device), radI)
    radI = torch.where((altitude < 1) & (radI > radG), radG, radI)
    radD = torch.where(radD > radG, radG, radD)

    return radI, radD

def shadowingfunction_wallheight_13(a, azimuth, altitude, scale, walls, aspect):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not walls.size:
        pass
        # Add the implementation for creating walls if needed

    azimuth = torch.tensor(azimuth * (torch.pi / 180.0), device=device).clone().detach()
    altitude = torch.tensor(altitude * (torch.pi / 180.0), device=device).clone().detach()

    sizex = a.shape[0]
    sizey = a.shape[1]

    f = torch.tensor(a, device=device).clone().detach()
    dx = torch.tensor(0.0, device=device).clone().detach()
    dy = torch.tensor(0.0, device=device).clone().detach()
    dz = torch.tensor(0.0, device=device).clone().detach()
    temp = torch.zeros((sizex, sizey), device=device).clone().detach()
    wallbol = (walls > 0).float()

    amaxvalue = torch.max(a)
    pibyfour = torch.pi / 4
    threetimespibyfour = 3 * pibyfour
    fivetimespibyfour = 5 * pibyfour
    seventimespibyfour = 7 * pibyfour
    sinazimuth = torch.sin(azimuth)
    cosazimuth = torch.cos(azimuth)
    tanazimuth = torch.tan(azimuth)
    signsinazimuth = torch.sign(sinazimuth)
    signcosazimuth = torch.sign(cosazimuth)
    dssin = torch.abs(1 / sinazimuth)
    dscos = torch.abs(1 / cosazimuth)
    tanaltitudebyscale = torch.tan(altitude) / scale

    index = 1

    while (amaxvalue >= dz) and (torch.abs(dx) < sizex) and (torch.abs(dy) < sizey):
        if (pibyfour <= azimuth and azimuth < threetimespibyfour) or (fivetimespibyfour <= azimuth and azimuth < seventimespibyfour):
            dy = signsinazimuth * index
            dx = -1 * signcosazimuth * torch.abs(torch.round(index / tanazimuth))
            ds = dssin
        else:
            dy = signsinazimuth * torch.abs(torch.round(index * tanazimuth))
            dx = -1 * signcosazimuth * index
            ds = dscos

        dz = ds * index * tanaltitudebyscale
        temp[0:sizex, 0:sizey] = 0

        absdx = torch.abs(dx)
        absdy = torch.abs(dy)

        xc1 = int((dx + absdx) / 2)
        xc2 = int(sizex + (dx - absdx) / 2)
        yc1 = int((dy + absdy) / 2)
        yc2 = int(sizey + (dy - absdy) / 2)

        xp1 = int(-((dx - absdx) / 2))
        xp2 = int(sizex - (dx + absdx) / 2)
        yp1 = int(-((dy - absdy) / 2))
        yp2 = int(sizey - (dy + absdy) / 2)

        temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz
        f = torch.maximum(f, temp)

        index = index + 1

    azilow = azimuth - torch.pi / 2
    azihigh = azimuth + torch.pi / 2

    if azilow >= 0 and azihigh < 2 * torch.pi:
        facesh = (torch.logical_or(aspect < azilow, aspect >= azihigh).float() - wallbol + 1)
    elif azilow < 0 and azihigh <= 2 * torch.pi:
        azilow = azilow + 2 * torch.pi
        facesh = torch.logical_or(aspect > azilow, aspect <= azihigh).float() * -1 + 1
    elif azilow > 0 and azihigh >= 2 * torch.pi:
        azihigh = azihigh - 2 * torch.pi
        facesh = torch.logical_or(aspect > azilow, aspect <= azihigh).float() * -1 + 1

    sh = f - torch.tensor(a, device=device)
    facesun = torch.logical_and(facesh + wallbol == 1, walls > 0).float()
    wallsun = walls - sh
    wallsun[wallsun < 0] = 0
    wallsun[facesh == 1] = 0
    wallsh = walls - wallsun

    sh = torch.logical_not(torch.logical_not(sh)).float()
    sh = sh * -1 + 1

    del temp

    return sh, wallsh, wallsun, facesh, facesun


def shadowingfunction_wallheight_23(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush, walls, aspect):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    degrees = torch.tensor(np.pi / 180.0, device=device)
    azimuth = torch.tensor(azimuth, device=device) * degrees
    altitude = torch.tensor(altitude, device=device) * degrees

    sizex, sizey = a.shape

    # initialise parameters
    dx = torch.tensor(0.0, device=device)
    dy = torch.tensor(0.0, device=device)
    dz = torch.tensor(0.0, device=device)
    temp = torch.zeros((sizex, sizey), device=device)
    tempvegdem = torch.zeros((sizex, sizey), device=device)
    tempvegdem2 = torch.zeros((sizex, sizey), device=device)
    templastfabovea = torch.zeros((sizex, sizey), device=device)
    templastgabovea = torch.zeros((sizex, sizey), device=device)
    bushplant = bush > 1
    sh = torch.zeros((sizex, sizey), device=device) 
    vbshvegsh = torch.zeros((sizex, sizey), device=device) 
    vegsh = torch.zeros((sizex, sizey), device=device) + bushplant.float() 
    f = a 
    shvoveg = vegdem 
    wallbol = (walls > 0).float()

    pibyfour = torch.tensor(np.pi / 4.0, device=device)
    threetimespibyfour = 3 * pibyfour
    fivetimespibyfour = 5 * pibyfour
    seventimespibyfour = 7 * pibyfour
    sinazimuth = torch.sin(azimuth)
    cosazimuth = torch.cos(azimuth)
    tanazimuth = torch.tan(azimuth)
    signsinazimuth = torch.sign(sinazimuth)
    signcosazimuth = torch.sign(cosazimuth)
    dssin = torch.abs(1 / sinazimuth)
    dscos = torch.abs(1 / cosazimuth)
    tanaltitudebyscale = torch.tan(altitude) / scale

    index = 0
    dzprev = torch.tensor(0.0, device=device)

    while (amaxvalue >= dz) and (torch.abs(dx) < sizex) and (torch.abs(dy) < sizey):
        if ((pibyfour <= azimuth) and (azimuth < threetimespibyfour)) or ((fivetimespibyfour <= azimuth) and (azimuth < seventimespibyfour)):
            dy = signsinazimuth * index
            dx = -1 * signcosazimuth * torch.abs(torch.round(index / tanazimuth))
            ds = dssin
        else:
            dy = signsinazimuth * torch.abs(torch.round(index * tanazimuth))
            dx = -1 * signcosazimuth * index
            ds = dscos

        dz = (ds * index) * tanaltitudebyscale
        tempvegdem.zero_()
        tempvegdem2.zero_()
        temp.zero_()
        templastfabovea.zero_()
        templastgabovea.zero_()

        absdx = torch.abs(dx)
        absdy = torch.abs(dy)
        xc1 = int((dx + absdx) / 2)
        xc2 = int(sizex + (dx - absdx) / 2)
        yc1 = int((dy + absdy) / 2)
        yc2 = int(sizey + (dy - absdy) / 2)
        xp1 = -int((dx - absdx) / 2)
        xp2 = int(sizex - (dx + absdx) / 2)
        yp1 = -int((dy - absdy) / 2)
        yp2 = int(sizey - (dy + absdy) / 2)

        tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dz
        tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz
        temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz

        f = torch.maximum(f, temp) # Moving building shadow
        shvoveg = torch.maximum(shvoveg, tempvegdem) # moving vegetation shadow volume
        sh = torch.where(f > a, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
        fabovea = (tempvegdem > a).float()   # vegdem above DEM
        gabovea = (tempvegdem2 > a).float()   # vegdem2 above DEM

        templastfabovea[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dzprev
        templastgabovea[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dzprev
        lastfabovea = templastfabovea > a
        lastgabovea = templastgabovea > a
        dzprev = dz
        vegsh2 = fabovea + gabovea + lastfabovea.float() + lastgabovea.float()
        vegsh2 = torch.where(vegsh2 == 4, torch.tensor(0.0, device=device), vegsh2)
        vegsh2 = torch.where(vegsh2 > 0, torch.tensor(1.0, device=device), vegsh2)

        vegsh = torch.maximum(vegsh, vegsh2)
        vegsh = torch.where(vegsh * sh > 0, torch.tensor(0.0, device=device), vegsh)
        vbshvegsh = vbshvegsh + vegsh

        index += 1

    azilow = azimuth - torch.pi / 2
    azihigh = azimuth + torch.pi / 2
    if azilow >= 0 and azihigh < 2 * torch.pi:    # 90 to 270  (SHADOW)
        facesh = torch.logical_or(aspect < azilow, aspect >= azihigh).float() - wallbol + 1
    elif azilow < 0 and azihigh <= 2 * torch.pi:    # 0 to 90
        azilow += 2 * torch.pi
        facesh = torch.logical_or(aspect > azilow, aspect <= azihigh).float() * -1 + 1
    elif azilow > 0 and azihigh >= 2 * torch.pi:    # 270 to 360
        azihigh -= 2 * torch.pi
        facesh = torch.logical_or(aspect > azilow, aspect <= azihigh).float() * -1 + 1

    sh = 1 - sh
    vbshvegsh = torch.where(vbshvegsh > 0, torch.tensor(1.0, device=device), vbshvegsh)
    vbshvegsh = vbshvegsh - vegsh

    vegsh = torch.where(vegsh > 0, torch.tensor(1.0, device=device), vegsh)
    shvoveg = (shvoveg - a) * vegsh  # Vegetation shadow volume
    vegsh = 1 - vegsh
    vbshvegsh = 1 - vbshvegsh

    shvo = f - a   # building shadow volume
    facesun = torch.logical_and(facesh + wallbol == 1, walls > 0).float()
    wallsun = walls - shvo
    wallsun = torch.where(wallsun < 0, torch.tensor(0.0, device=device), wallsun)
    wallsun = torch.where(facesh == 1, torch.tensor(0.0, device=device), wallsun)
    wallsh = walls - wallsun

    wallshve = shvoveg * wallbol
    wallshve = wallshve - wallsh
    wallshve = torch.where(wallshve < 0, torch.tensor(0.0, device=device), wallshve)
    wallsun = wallsun - wallshve
    wallsun = torch.where(wallsun < 0, torch.tensor(0.0, device=device), wallsun)
    wallshve = torch.where(wallshve > walls, walls, wallshve)

    del fabovea,gabovea,lastfabovea,lastgabovea,vegsh2
    del tempvegdem,tempvegdem2,templastfabovea,templastgabovea,shvoveg,wallbol

    return vegsh, sh, vbshvegsh, wallsh, wallsun, wallshve, facesh, facesun


def Perez_v3(zen, azimuth, radD, radI, jday, patchchoice, patch_option):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    m_a1 = torch.tensor([1.3525, -1.2219, -1.1000, -0.5484, -0.6000, -1.0156, -1.0000, -1.0500], device=device)
    m_a2 = torch.tensor([-0.2576, -0.7730, -0.2515, -0.6654, -0.3566, -0.3670, 0.0211, 0.0289], device=device)
    m_a3 = torch.tensor([-0.2690, 1.4148, 0.8952, -0.2672, -2.5000, 1.0078, 0.5025, 0.4260], device=device)
    m_a4 = torch.tensor([-1.4366, 1.1016, 0.0156, 0.7117, 2.3250, 1.4051, -0.5119, 0.3590], device=device)
    m_b1 = torch.tensor([-0.7670, -0.2054, 0.2782, 0.7234, 0.2937, 0.2875, -0.3000, -0.3250], device=device)
    m_b2 = torch.tensor([0.0007, 0.0367, -0.1812, -0.6219, 0.0496, -0.5328, 0.1922, 0.1156], device=device)
    m_b3 = torch.tensor([1.2734, -3.9128, -4.5000, -5.6812, -5.6812, -3.8500, 0.7023, 0.7781], device=device)
    m_b4 = torch.tensor([-0.1233, 0.9156, 1.1766, 2.6297, 1.8415, 3.3750, -1.6317, 0.0025], device=device)
    m_c1 = torch.tensor([2.8000, 6.9750, 24.7219, 33.3389, 21.0000, 14.0000, 19.0000, 31.0625], device=device)
    m_c2 = torch.tensor([0.6004, 0.1774, -13.0812, -18.3000, -4.7656, -0.9999, -5.0000, -14.5000], device=device)
    m_c3 = torch.tensor([1.2375, 6.4477, -37.7000, -62.2500, -21.5906, -7.1406, 1.2438, -46.1148], device=device)
    m_c4 = torch.tensor([1.0000, -0.1239, 34.8438, 52.0781, 7.2492, 7.5469, -1.9094, 55.3750], device=device)
    m_d1 = torch.tensor([1.8734, -1.5798, -5.0000, -3.5000, -3.5000, -3.4000, -4.0000, -7.2312], device=device)
    m_d2 = torch.tensor([0.6297, -0.5081, 1.5218, 0.0016, -0.1554, -0.1078, 0.0250, 0.4050], device=device)
    m_d3 = torch.tensor([0.9738, -1.7812, 3.9229, 1.1477, 1.4062, -1.0750, 0.3844, 13.3500], device=device)
    m_d4 = torch.tensor([0.2809, 0.1080, -2.6204, 0.1062, 0.3988, 1.5702, 0.2656, 0.6234], device=device)
    m_e1 = torch.tensor([0.0356, 0.2624, -0.0156, 0.4659, 0.0032, -0.0672, 1.0468, 1.5000], device=device)
    m_e2 = torch.tensor([-0.1246, 0.0672, 0.1597, -0.3296, 0.0766, 0.4016, -0.3788, -0.6426], device=device)
    m_e3 = torch.tensor([-0.5718, -0.2190, 0.4199, -0.0876, -0.0656, 0.3017, -2.4517, 1.8564], device=device)
    m_e4 = torch.tensor([0.9938, -0.4285, -0.5562, -0.0329, -0.1294, -0.4844, 1.4656, 0.5636], device=device)

    acoeff = torch.stack([m_a1, m_a2, m_a3, m_a4], dim=1)
    bcoeff = torch.stack([m_b1, m_b2, m_b3, m_b4], dim=1)
    ccoeff = torch.stack([m_c1, m_c2, m_c3, m_c4], dim=1)
    dcoeff = torch.stack([m_d1, m_d2, m_d3, m_d4], dim=1)
    ecoeff = torch.stack([m_e1, m_e2, m_e3, m_e4], dim=1)

    deg2rad = torch.tensor(np.pi / 180, device=device).clone().detach()
    rad2deg = torch.tensor(180 / np.pi, device=device).clone().detach()
    altitude = 90 - zen
    zen = torch.tensor(zen, device=device) * deg2rad
    azimuth = torch.tensor(azimuth, device=device) * deg2rad
    altitude = torch.tensor(altitude, device=device) * deg2rad
    Idh = radD 
    Ibn = radI

    PerezClearness = ((Idh + Ibn) / (Idh + 1.041 * torch.pow(zen, 3))) / (1 + 1.041 * torch.pow(zen, 3))

    day_angle = jday * 2 * torch.pi / 365
    I0 = 1367 * (1.00011 + 0.034221 * torch.cos(torch.tensor(day_angle)) + 0.00128 * torch.sin(torch.tensor(day_angle)) + 0.000719 *
                 torch.cos(2 * torch.tensor(day_angle)) + 0.000077 * torch.sin(2 * torch.tensor(day_angle)))

    if altitude >= 10 * deg2rad:
        AirMass = 1 / torch.sin(altitude)
    elif altitude < 0:
        AirMass = 1 / torch.sin(altitude) + 0.50572 * torch.pow(180 * torch.complex(altitude, 0) / torch.pi + 6.07995, -1.6364)
    else:
        AirMass = 1 / torch.sin(altitude) + 0.50572 * torch.pow(180 * altitude / torch.pi + 6.07995, -1.6364)

    PerezBrightness = (AirMass * Idh) / I0
    if Idh <= 10:
        PerezBrightness = torch.tensor(0.0, device=device)

    if PerezClearness < 1.065:
        intClearness = 0
    elif PerezClearness < 1.230:
        intClearness = 1
    elif PerezClearness < 1.500:
        intClearness = 2
    elif PerezClearness < 1.950:
        intClearness = 3
    elif PerezClearness < 2.800:
        intClearness = 4
    elif PerezClearness < 4.500:
        intClearness = 5
    elif PerezClearness < 6.200:
        intClearness = 6
    else:
        intClearness = 7

    m_a = acoeff[intClearness, 0] + acoeff[intClearness, 1] * zen + PerezBrightness * (acoeff[intClearness, 2] + acoeff[intClearness, 3] * zen)
    m_b = bcoeff[intClearness, 0] + bcoeff[intClearness, 1] * zen + PerezBrightness * (bcoeff[intClearness, 2] + bcoeff[intClearness, 3] * zen)
    m_e = ecoeff[intClearness, 0] + ecoeff[intClearness, 1] * zen + PerezBrightness * (ecoeff[intClearness, 2] + ecoeff[intClearness, 3] * zen)

    if intClearness > 0:
        m_c = ccoeff[intClearness, 0] + ccoeff[intClearness, 1] * zen + PerezBrightness * (ccoeff[intClearness, 2] + ccoeff[intClearness, 3] * zen)
        m_d = dcoeff[intClearness, 0] + dcoeff[intClearness, 1] * zen + PerezBrightness * (dcoeff[intClearness, 2] + dcoeff[intClearness, 3] * zen)
    else:
        m_c = torch.exp(torch.pow(PerezBrightness * (ccoeff[intClearness, 0] + ccoeff[intClearness, 1] * zen), ccoeff[intClearness, 2])) - 1
        m_d = -torch.exp(PerezBrightness * (dcoeff[intClearness, 0] + dcoeff[intClearness, 1] * zen)) + dcoeff[intClearness, 2] + \
              PerezBrightness * dcoeff[intClearness, 3] * PerezBrightness

    if patchchoice == 2:
        skyvaultalt = torch.zeros((90, 361), device=device)
        skyvaultazi = torch.zeros((90, 361), device=device)
        for j in range(90):
            skyvaultalt[j, :] = 91 - j
            skyvaultazi[j, :] = torch.arange(361)

    elif patchchoice == 1:
        skyvaultalt, skyvaultazi, _, _, _, _, _ = create_patches(patch_option)

    skyvaultzen = (90 - skyvaultalt) * deg2rad
    skyvaultalt = skyvaultalt * deg2rad
    skyvaultazi = skyvaultazi * deg2rad

    cosSkySunAngle = torch.sin(skyvaultalt) * torch.sin(altitude) + \
                     torch.cos(altitude) * torch.cos(skyvaultalt) * torch.cos(torch.abs(skyvaultazi - azimuth))

    lv = (1 + m_a * torch.exp(m_b / torch.cos(skyvaultzen))) * ((1 + m_c * torch.exp(m_d * torch.arccos(cosSkySunAngle)) +
                                                                 m_e * cosSkySunAngle * cosSkySunAngle))

    lv = lv / torch.sum(lv)

    if patchchoice == 1:
        x = torch.transpose(torch.unsqueeze(skyvaultalt * rad2deg, 0), 0, 1)
        y = torch.transpose(torch.unsqueeze(skyvaultazi * rad2deg, 0), 0, 1)
        z = torch.transpose(torch.unsqueeze(lv, 0), 0, 1)
        lv = torch.cat((x, y, z), dim=1)
    return lv, PerezClearness, PerezBrightness

def model1(sky_patches, esky, Ta):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SBC = 5.67051e-8
    deg2rad = torch.tensor(np.pi / 180, device=device)

    skyalt, skyalt_c = torch.unique(sky_patches[:, 0], return_counts=True)
    skyzen = 90 - skyalt
    cosskyzen = torch.cos(skyzen * deg2rad)

    a_c = 0.67
    b_c = 0.094

    ln_u_prec = esky / b_c - a_c / b_c - 0.5
    u_prec = torch.exp(ln_u_prec)
    owp = u_prec / cosskyzen
    log_owp = torch.log(owp)

    esky_band = a_c + b_c * log_owp
    p_alt = sky_patches[:, 0]
    patch_emissivity = torch.zeros((p_alt.shape[0]), device=device)

    for idx in skyalt:
        temp_emissivity = esky_band[skyalt == idx]
        patch_emissivity[p_alt == idx] = temp_emissivity

    patch_emissivity_normalized = patch_emissivity / torch.sum(patch_emissivity)

    return patch_emissivity_normalized, esky_band

def model2(sky_patches, esky, Ta):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deg2rad = torch.tensor(np.pi / 180, device=device)

    skyalt, skyalt_c = torch.unique(sky_patches[:, 0], return_counts=True)
    skyzen = 90 - skyalt

    b_c = 0.308

    esky_band = 1 - (1 - esky) * torch.exp(b_c * (1.7 - (1 / torch.cos(skyzen * deg2rad))))
    p_alt = sky_patches[:, 0]
    patch_emissivity = torch.zeros((p_alt.shape[0]), device=device)

    for idx in skyalt:
        temp_emissivity = esky_band[skyalt == idx]
        patch_emissivity[p_alt == idx] = temp_emissivity

    patch_emissivity_normalized = patch_emissivity / torch.sum(patch_emissivity)

    return patch_emissivity_normalized, esky_band

def model3(sky_patches, esky, Ta):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deg2rad = torch.tensor(np.pi / 180, device=device)

    skyalt, skyalt_c = torch.unique(sky_patches[:, 0], return_counts=True)
    skyzen = 90 - skyalt

    b_c = 1.8

    esky_band = 1 - (1 - esky) ** (1 / (b_c * torch.cos(skyzen * deg2rad)))
    p_alt = sky_patches[:, 0]
    patch_emissivity = torch.zeros((p_alt.shape[0]), device=device)

    for idx in skyalt:
        temp_emissivity = esky_band[skyalt == idx]
        patch_emissivity[p_alt == idx] = temp_emissivity

    patch_emissivity_normalized = patch_emissivity / torch.sum(patch_emissivity)

    return patch_emissivity_normalized, esky_band


def define_patch_characteristics(solar_altitude, solar_azimuth,
                                 patch_altitude, patch_azimuth, steradian,
                                 asvf,
                                 shmat, vegshmat, vbshvegshmat,
                                 Lsky_down, Lsky_side, Lsky, Lup,
                                 Ta, Tgwall, ewall,
                                 rows, cols):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Stefan-Boltzmann's Constant
    SBC = torch.tensor(5.67051e-8, device=device)
    deg2rad = torch.tensor(np.pi / 180, device=device)

    # Define variables
    Ldown = torch.zeros((rows, cols), device=device)
    Ldown_sky = torch.zeros((rows, cols), device=device)
    Ldown_veg = torch.zeros((rows, cols), device=device)
    Ldown_sun = torch.zeros((rows, cols), device=device)
    Ldown_sh = torch.zeros((rows, cols), device=device)
    Ldown_ref = torch.zeros((rows, cols), device=device)

    Lside = torch.zeros((rows, cols), device=device)
    Lside_sky = torch.zeros((rows, cols), device=device)
    Lside_veg = torch.zeros((rows, cols), device=device)
    Lside_sun = torch.zeros((rows, cols), device=device)
    Lside_sh = torch.zeros((rows, cols), device=device)
    Lside_ref = torch.zeros((rows, cols), device=device)

    Least = torch.zeros((rows, cols), device=device)
    Lwest = torch.zeros((rows, cols), device=device)
    Lnorth = torch.zeros((rows, cols), device=device)
    Lsouth = torch.zeros((rows, cols), device=device)

    ewall = torch.tensor(ewall, device=device).clone().detach()

    for idx in range(patch_altitude.shape[0]):
        # Calculations for patches on sky, shmat = 1 = sky is visible
        temp_sky = ((shmat[:, :, idx] == 1) & (vegshmat[:, :, idx] == 1))

        # Longwave radiation from sky to vertical surface
        Ldown_sky += temp_sky * Lsky_down[idx, 2]

        # Longwave radiation from sky to horizontal surface
        Lside_sky += temp_sky * Lsky_side[idx, 2]

        # Calculations for patches that are vegetation, vegshmat = 0 = shade from vegetation
        temp_vegsh = ((vegshmat[:, :, idx] == 0) | (vbshvegshmat[:, :, idx] == 0))
        # Longwave radiation from vegetation surface (considered vertical)
        vegetation_surface = ((ewall * SBC * ((Ta + 273.15) ** 4)) / torch.tensor(np.pi, device=device))

        # Longwave radiation reaching a vertical surface
        Lside_veg += vegetation_surface * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_vegsh

        # Longwave radiation reaching a horizontal surface
        Ldown_veg += vegetation_surface * steradian[idx] * torch.sin(patch_altitude[idx] * deg2rad) * temp_vegsh

        # Portion into cardinal directions to be used for standing box or POI output
        if (patch_azimuth[idx] > 360) or (patch_azimuth[idx] < 180):
            Least += temp_sky * Lsky_side[idx, 2] * torch.cos((90 - patch_azimuth[idx]) * deg2rad)
            Least += vegetation_surface * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_vegsh * torch.cos((90 - patch_azimuth[idx]) * deg2rad)
        if (patch_azimuth[idx] > 90) and (patch_azimuth[idx] < 270):
            Lsouth += temp_sky * Lsky_side[idx, 2] * torch.cos((180 - patch_azimuth[idx]) * deg2rad)
            Lsouth += vegetation_surface * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_vegsh * torch.cos((180 - patch_azimuth[idx]) * deg2rad)
        if (patch_azimuth[idx] > 180) and (patch_azimuth[idx] < 360):
            Lwest += temp_sky * Lsky_side[idx, 2] * torch.cos((270 - patch_azimuth[idx]) * deg2rad)
            Lwest += vegetation_surface * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_vegsh * torch.cos((270 - patch_azimuth[idx]) * deg2rad)
        if (patch_azimuth[idx] > 270) or (patch_azimuth[idx] < 90):
            Lnorth += temp_sky * Lsky_side[idx, 2] * torch.cos((0 - patch_azimuth[idx]) * deg2rad)
            Lnorth += vegetation_surface * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_vegsh * torch.cos((0 - patch_azimuth[idx]) * deg2rad)

        # Calculations for patches that are buildings, shmat = 0 = shade from buildings
        temp_vbsh = (1 - shmat[:, :, idx]) * vbshvegshmat[:, :, idx]
        temp_sh = (temp_vbsh == 1)
        azimuth_difference = torch.abs(solar_azimuth - patch_azimuth[idx])

        # Longwave radiation from sunlit surfaces
        sunlit_surface = ((ewall * SBC * ((Ta + Tgwall + 273.15) ** 4)) / torch.tensor(np.pi, device=device))
        # Longwave radiation from shaded surfaces
        shaded_surface = ((ewall * SBC * ((Ta + 273.15) ** 4)) / torch.tensor(np.pi, device=device))
        if ((azimuth_difference > 90) and (azimuth_difference < 270) and (solar_altitude > 0)):
            # Calculate which patches defined as buildings that are sunlit or shaded
            sunlit_patches, shaded_patches = shaded_or_sunlit(solar_altitude, solar_azimuth, patch_altitude[idx], patch_azimuth[idx], asvf)

            # Calculate longwave radiation from sunlit walls to vertical surface
            Lside_sun += sunlit_surface * sunlit_patches * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh
            # Calculate longwave radiation from shaded walls to vertical surface
            Lside_sh += shaded_surface * shaded_patches * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh

            # Calculate longwave radiation from sunlit walls to horizontal surface
            Ldown_sun += sunlit_surface * sunlit_patches * steradian[idx] * torch.sin(patch_altitude[idx] * deg2rad) * temp_sh
            # Calculate longwave radiation from shaded walls to horizontal surface
            Ldown_sh += shaded_surface * shaded_patches * steradian[idx] * torch.sin(patch_altitude[idx] * deg2rad) * temp_sh

            # Portion into cardinal directions to be used for standing box or POI output
            if (patch_azimuth[idx] > 360) or (patch_azimuth[idx] < 180):
                Least += sunlit_surface * sunlit_patches * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((90 - patch_azimuth[idx]) * deg2rad)
                Least += shaded_surface * shaded_patches * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((90 - patch_azimuth[idx]) * deg2rad)
            if (patch_azimuth[idx] > 90) and (patch_azimuth[idx] < 270):
                Lsouth += sunlit_surface * sunlit_patches * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((180 - patch_azimuth[idx]) * deg2rad)
                Lsouth += shaded_surface * shaded_patches * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((180 - patch_azimuth[idx]) * deg2rad)
            if (patch_azimuth[idx] > 180) and (patch_azimuth[idx] < 360):
                Lwest += sunlit_surface * sunlit_patches * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((270 - patch_azimuth[idx]) * deg2rad)
                Lwest += shaded_surface * shaded_patches * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((270 - patch_azimuth[idx]) * deg2rad)
            if (patch_azimuth[idx] > 270) or (patch_azimuth[idx] < 90):
                Lnorth += sunlit_surface * sunlit_patches * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((0 - patch_azimuth[idx]) * deg2rad)
                Lnorth += shaded_surface * shaded_patches * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((0 - patch_azimuth[idx]) * deg2rad)

        else:
            # Calculate longwave radiation from shaded walls reaching a vertical surface
            Lside_sh += shaded_surface * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh

            # Calculate longwave radiation from shaded walls reaching a horizontal surface
            Ldown_sh += shaded_surface * steradian[idx] * torch.sin(patch_altitude[idx] * deg2rad) * temp_sh

            # Portion into cardinal directions to be used for standing box or POI output
            if (patch_azimuth[idx] > 360) or (patch_azimuth[idx] < 180):
                Least += shaded_surface * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((90 - patch_azimuth[idx]) * deg2rad)
            if (patch_azimuth[idx] > 90) and (patch_azimuth[idx] < 270):
                Lsouth += shaded_surface * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((180 - patch_azimuth[idx]) * deg2rad)
            if (patch_azimuth[idx] > 180) and (patch_azimuth[idx] < 360):
                Lwest += shaded_surface * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((270 - patch_azimuth[idx]) * deg2rad)
            if (patch_azimuth[idx] > 270) or (patch_azimuth[idx] < 90):
                Lnorth += shaded_surface * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((0 - patch_azimuth[idx]) * deg2rad)

    # Calculate reflected longwave in each patch
    reflected_on_surfaces = (((Ldown_sky + Lup) * (1 - ewall) * 0.5) / torch.tensor(np.pi, device=device))
    for idx in range(patch_altitude.shape[0]):
        temp_sh = ((shmat[:, :, idx] == 0) | (vegshmat[:, :, idx] == 0) | (vbshvegshmat[:, :, idx] == 0))

        # Reflected longwave radiation reaching vertical surfaces
        Lside_ref += reflected_on_surfaces * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh

        # Reflected longwave radiation reaching horizontal surfaces
        Ldown_ref += reflected_on_surfaces * steradian[idx] * torch.sin(patch_altitude[idx] * deg2rad) * temp_sh

        # Portion into cardinal directions to be used for standing box or POI output
        if (patch_azimuth[idx] > 360) or (patch_azimuth[idx] < 180):
            Least += reflected_on_surfaces * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((90 - patch_azimuth[idx]) * deg2rad)
        if (patch_azimuth[idx] > 90) and (patch_azimuth[idx] < 270):
            Lsouth += reflected_on_surfaces * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((180 - patch_azimuth[idx]) * deg2rad)
        if (patch_azimuth[idx] > 180) and (patch_azimuth[idx] < 360):
            Lwest += reflected_on_surfaces * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((270 - patch_azimuth[idx]) * deg2rad)
        if (patch_azimuth[idx] > 270) or (patch_azimuth[idx] < 90):
            Lnorth += reflected_on_surfaces * steradian[idx] * torch.cos(patch_altitude[idx] * deg2rad) * temp_sh * torch.cos((0 - patch_azimuth[idx]) * deg2rad)

    # Sum of all Lside components (sky, vegetation, sunlit and shaded buildings, reflected)
    Lside = Lside_sky + Lside_veg + Lside_sh + Lside_sun + Lside_ref

    # Sum of all Ldown components (sky, vegetation, sunlit and shaded buildings, reflected)
    Ldown = Ldown_sky + Ldown_veg + Ldown_sh + Ldown_sun + Ldown_ref


    del Ldown_sky ,Ldown_veg ,Ldown_sun ,Ldown_sh ,Ldown_ref

    del temp_vegsh, vegetation_surface, temp_vbsh, temp_sh

    return Ldown, Lside, Lside_sky, Lside_veg, Lside_sh, Lside_sun, Lside_ref, Least, Lwest, Lnorth, Lsouth


def Lcyl_v2022a(esky, sky_patches, Ta, Tgwall, ewall, Lup, shmat, vegshmat, vbshvegshmat, solar_altitude, solar_azimuth, rows, cols, asvf):

    # Device for GPU computation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Stefan-Boltzmann's Constant
    SBC = torch.tensor(5.67051e-8, device=device).clone().detach()

    # Sky longwave radiation from emissivity based on Prata (1996)
    Ldown_prata = (esky * SBC * ((Ta + 273.15) ** 4))

    # Degrees to radians
    deg2rad = torch.tensor(np.pi / 180, device=device).clone().detach()

    # Unique altitudes for patches
    sky_patches_cpu = sky_patches.cpu().numpy()
    skyalt, skyalt_c = np.unique(sky_patches_cpu[:, 0], return_counts=True)
    skyalt = torch.tensor(skyalt, device=device).clone().detach()
    skyalt_c = torch.tensor(skyalt_c, device=device).clone().detach()

    # Altitudes and azimuths of the Robinson & Stone patches
    patch_altitude = sky_patches[:,0] 
    patch_azimuth = sky_patches[:,1] 
    emis_m = 2

    # Unsworth & Monteith (1975)
    if emis_m == 1:
        patch_emissivity_normalized, esky_band = model1(sky_patches, esky, Ta)
    # Martin & Berdahl (1984)
    elif emis_m == 2:
        patch_emissivity_normalized, esky_band = model2(sky_patches, esky, Ta)
    # Bliss (1961)
    elif emis_m == 3:
        patch_emissivity_normalized, esky_band = model3(sky_patches, esky, Ta)

    # Calculation of steradian for each patch
    steradian = torch.zeros(patch_altitude.shape[0], device=device)
    for i in range(patch_altitude.shape[0]):
        # If there are more than one patch in a band
        if skyalt_c[skyalt == patch_altitude[i]] > 1:
            steradian[i] = ((360 / skyalt_c[skyalt == patch_altitude[i]]) * deg2rad) * (torch.sin((patch_altitude[i] + patch_altitude[0]) * deg2rad) \
            - torch.sin((patch_altitude[i] - patch_altitude[0]) * deg2rad))
        # If there is only one patch in band, i.e. 90 degrees
        else:
            steradian[i] = ((360 / skyalt_c[skyalt == patch_altitude[i]]) * deg2rad) * (torch.sin((patch_altitude[i]) * deg2rad) \
                - torch.sin((patch_altitude[i-1] + patch_altitude[0]) * deg2rad))

    # True = anisotropic sky, False = isotropic sky
    anisotropic_sky = True

    # Longwave based on spectral flux density (divide by pi)
    Ldown = torch.zeros(patch_altitude.shape[0], device=device)
    Lside = torch.zeros(patch_altitude.shape[0], device=device)
    Lnormal = torch.zeros(patch_altitude.shape[0], device=device)
    for altitude in skyalt:
        # Anisotropic sky
        if anisotropic_sky:
            temp_emissivity = esky_band[skyalt == altitude]
        # Isotropic sky but with patches (need to switch anisotropic_sky to False)
        else:
            temp_emissivity = esky
        # Estimate longwave radiation on a horizontal surface (Ldown), vertical surface (Lside) and perpendicular (Lnormal)
        Ldown[patch_altitude == altitude] = ((temp_emissivity * SBC * ((Ta + 273.15) ** 4)) / torch.tensor(np.pi, device=device)) * steradian[patch_altitude == altitude] * torch.sin(altitude * deg2rad)
        Lside[patch_altitude == altitude] = ((temp_emissivity * SBC * ((Ta + 273.15) ** 4)) / torch.tensor(np.pi, device=device)) * steradian[patch_altitude == altitude] * torch.cos(altitude * deg2rad)
        Lnormal[patch_altitude == altitude] = ((temp_emissivity * SBC * ((Ta + 273.15) ** 4)) / torch.tensor(np.pi, device=device)) * steradian[patch_altitude == altitude]

    Lsky_normal = torch.clone(sky_patches)
    Lsky_down = torch.clone(sky_patches)
    Lsky_side = torch.clone(sky_patches)

    Lsky_normal[:, 2] = Lnormal
    Lsky_down[:, 2] = Ldown
    Lsky_side[:, 2] = Lside

    # Estimate longwave radiation in each patch based on patch characteristics, i.e. sky, vegetation or building (shaded or sunlit)
    Ldown, Lside, Lside_sky, Lside_veg, Lside_sh, Lside_sun, Lside_ref, \
            Least_, Lwest_, Lnorth_, Lsouth_ = define_patch_characteristics(solar_altitude, solar_azimuth,
                                 patch_altitude, patch_azimuth, steradian,
                                 asvf,
                                 shmat, vegshmat, vbshvegshmat,
                                 Lsky_down, Lsky_side, Lsky_normal, Lup,
                                 Ta, Tgwall, ewall,
                                 rows, cols)

    del Lnormal, Lsky_normal, Lsky_down, Lsky_side

    return Ldown, Lside, Least_, Lwest_, Lnorth_, Lsouth_

def Lvikt_veg(svf, svfveg, svfaveg, vikttot):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    viktonlywall = (vikttot - (63.227 * svf ** 6 - 161.51 * svf ** 5 + 156.91 * svf ** 4 - 70.424 * svf ** 3 + 16.773 * svf ** 2 - 0.4863 * svf)) / vikttot
    viktaveg = (vikttot - (63.227 * svfaveg ** 6 - 161.51 * svfaveg ** 5 + 156.91 * svfaveg ** 4 - 70.424 * svfaveg ** 3 + 16.773 * svfaveg ** 2 - 0.4863 * svfaveg)) / vikttot
    viktwall = viktonlywall - viktaveg
    svfvegbu = (svfveg + svf - 1)  # Vegetation plus buildings
    viktsky = (63.227 * svfvegbu ** 6 - 161.51 * svfvegbu ** 5 + 156.91 * svfvegbu ** 4 - 70.424 * svfvegbu ** 3 + 16.773 * svfvegbu ** 2 - 0.4863 * svfvegbu) / vikttot
    viktrefl = (vikttot - (63.227 * svfvegbu ** 6 - 161.51 * svfvegbu ** 5 + 156.91 * svfvegbu ** 4 - 70.424 * svfvegbu ** 3 + 16.773 * svfvegbu ** 2 - 0.4863 * svfvegbu)) / vikttot
    viktveg = (vikttot - (63.227 * svfvegbu ** 6 - 161.51 * svfvegbu ** 5 + 156.91 * svfvegbu ** 4 - 70.424 * svfvegbu ** 3 + 16.773 * svfvegbu ** 2 - 0.4863 * svfvegbu)) / vikttot
    viktveg = viktveg - viktwall

    del viktonlywall,viktaveg,svfvegbu

    return viktveg, viktwall, viktsky, viktrefl


def Lside_veg_v2022a(svfS, svfW, svfN, svfE, svfEveg, svfSveg, svfWveg, svfNveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg, azimuth, altitude, Ta, Tw, SBC, ewall, Ldown, esky, t, F_sh, CI, LupE, LupS, LupW, LupN, anisotropic_longwave):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    azimuth = torch.tensor(azimuth, device=device)
    altitude = torch.tensor(altitude, device=device)
    ewall = torch.tensor(ewall, device=device)
    t = torch.tensor(t, device=device)
    anisotropic_longwave = torch.tensor(anisotropic_longwave, device=device)

    # Building height angle from svf
    svfalfaE = torch.arcsin(torch.exp((torch.log(1 - svfE)) / 2))
    svfalfaS = torch.arcsin(torch.exp((torch.log(1 - svfS)) / 2))
    svfalfaW = torch.arcsin(torch.exp((torch.log(1 - svfW)) / 2))
    svfalfaN = torch.arcsin(torch.exp((torch.log(1 - svfN)) / 2))

    vikttot = torch.tensor(4.4897, device=device)
    aziW = azimuth + t
    aziN = azimuth - 90 + t
    aziE = azimuth - 180 + t
    aziS = azimuth - 270 + t

    F_sh = 2 * F_sh - 1  # (cylindric_wedge scaled 0-1)

    c = 1 - CI
    Lsky_allsky = esky * SBC * ((Ta + 273.15) ** 4) * (1 - c) + c * SBC * ((Ta + 273.15) ** 4)

    ## Least
    viktveg, viktwall, viktsky, viktrefl = Lvikt_veg(svfE, svfEveg, svfEaveg, vikttot)

    if altitude > 0:  # daytime
        alfaB = torch.arctan(svfalfaE)
        betaB = torch.arctan(torch.tan(svfalfaE * F_sh))
        betasun = ((alfaB - betaB) / 2) + betaB
        if (azimuth > (180 - t)) and (azimuth <= (360 - t)):
            Lwallsun = SBC * ewall * ((Ta + 273.15 + Tw * torch.sin(aziE * (torch.tensor(np.pi, device=device) / 180))) ** 4) * viktwall * (1 - F_sh) * torch.cos(betasun) * 0.5
            Lwallsh = SBC * ewall * ((Ta + 273.15) ** 4) * viktwall * F_sh * 0.5
        else:
            Lwallsun = torch.tensor(0, device=device)
            Lwallsh = SBC * ewall * ((Ta + 273.15) ** 4) * viktwall * 0.5
    else:  # nighttime
        Lwallsun = torch.tensor(0, device=device)
        Lwallsh = SBC * ewall * ((Ta + 273.15) ** 4) * viktwall * 0.5

    # Longwave from ground (see Lcyl_v2022a for remaining fluxes)
    if anisotropic_longwave == 1:
        Lground = LupE * 0.5
        Least = Lground
    else:
        Lsky = ((svfE + svfEveg - 1) * Lsky_allsky) * viktsky * 0.5
        Lveg = SBC * ewall * ((Ta + 273.15) ** 4) * viktveg * 0.5
        Lground = LupE * 0.5
        Lrefl = (Ldown + LupE) * (viktrefl) * (1 - ewall) * 0.5
        Least = Lsky + Lwallsun + Lwallsh + Lveg + Lground + Lrefl

    ## Lsouth
    viktveg, viktwall, viktsky, viktrefl = Lvikt_veg(svfS, svfSveg, svfSaveg, vikttot)

    if altitude > 0:  # daytime
        alfaB = torch.arctan(svfalfaS)
        betaB = torch.arctan(torch.tan(svfalfaS * F_sh))
        betasun = ((alfaB - betaB) / 2) + betaB
        if (azimuth <= (90 - t)) or (azimuth > (270 - t)):
            Lwallsun = SBC * ewall * ((Ta + 273.15 + Tw * torch.sin(aziS * (torch.tensor(np.pi, device=device) / 180))) ** 4) * viktwall * (1 - F_sh) * torch.cos(betasun) * 0.5
            Lwallsh = SBC * ewall * ((Ta + 273.15) ** 4) * viktwall * F_sh * 0.5
        else:
            Lwallsun = torch.tensor(0, device=device)
            Lwallsh = SBC * ewall * ((Ta + 273.15) ** 4) * viktwall * 0.5
    else:  # nighttime
        Lwallsun = torch.tensor(0, device=device)
        Lwallsh = SBC * ewall * ((Ta + 273.15) ** 4) * viktwall * 0.5

    if anisotropic_longwave == 1:
        Lground = LupS * 0.5
        Lsouth = Lground
    else:
        Lsky = ((svfS + svfSveg - 1) * Lsky_allsky) * viktsky * 0.5
        Lveg = SBC * ewall * ((Ta + 273.15) ** 4) * viktveg * 0.5
        Lground = LupS * 0.5
        Lrefl = (Ldown + LupS) * (viktrefl) * (1 - ewall) * 0.5
        Lsouth = Lsky + Lwallsun + Lwallsh + Lveg + Lground + Lrefl

    ## Lwest
    viktveg, viktwall, viktsky, viktrefl = Lvikt_veg(svfW, svfWveg, svfWaveg, vikttot)

    if altitude > 0:  # daytime
        alfaB = torch.arctan(svfalfaW)
        betaB = torch.arctan(torch.tan(svfalfaW * F_sh))
        betasun = ((alfaB - betaB) / 2) + betaB
        if (azimuth > (360 - t)) or (azimuth <= (180 - t)):
            Lwallsun = SBC * ewall * ((Ta + 273.15 + Tw * torch.sin(aziW * (torch.tensor(np.pi, device=device) / 180))) ** 4) * viktwall * (1 - F_sh) * torch.cos(betasun) * 0.5
            Lwallsh = SBC * ewall * ((Ta + 273.15) ** 4) * viktwall * F_sh * 0.5
        else:
            Lwallsun = torch.tensor(0, device=device)
            Lwallsh = SBC * ewall * ((Ta + 273.15) ** 4) * viktwall * 0.5
    else:  # nighttime
        Lwallsun = torch.tensor(0, device=device)
        Lwallsh = SBC * ewall * ((Ta + 273.15) ** 4) * viktwall * 0.5

    if anisotropic_longwave == 1:
        Lground = LupW * 0.5
        Lwest = Lground
    else:
        Lsky = ((svfW + svfWveg - 1) * Lsky_allsky) * viktsky * 0.5
        Lveg = SBC * ewall * ((Ta + 273.15) ** 4) * viktveg * 0.5
        Lground = LupW * 0.5
        Lrefl = (Ldown + LupW) * (viktrefl) * (1 - ewall) * 0.5
        Lwest = Lsky + Lwallsun + Lwallsh + Lveg + Lground + Lrefl

    ## Lnorth
    viktveg, viktwall, viktsky, viktrefl = Lvikt_veg(svfN, svfNveg, svfNaveg, vikttot)

    if altitude > 0:  # daytime
        alfaB = torch.arctan(svfalfaN)
        betaB = torch.arctan(torch.tan(svfalfaN * F_sh))
        betasun = ((alfaB - betaB) / 2) + betaB
        if (azimuth > (90 - t)) and (azimuth <= (270 - t)):
            Lwallsun = SBC * ewall * ((Ta + 273.15 + Tw * torch.sin(aziN * (torch.tensor(np.pi, device=device) / 180))) ** 4) * viktwall * (1 - F_sh) * torch.cos(betasun) * 0.5
            Lwallsh = SBC * ewall * ((Ta + 273.15) ** 4) * viktwall * F_sh * 0.5
        else:
            Lwallsun = torch.tensor(0, device=device)
            Lwallsh = SBC * ewall * ((Ta + 273.15) ** 4) * viktwall * 0.5
    else:  # nighttime
        Lwallsun = torch.tensor(0, device=device)
        Lwallsh = SBC * ewall * ((Ta + 273.15) ** 4) * viktwall * 0.5

    if anisotropic_longwave == 1:
        Lground = LupN * 0.5
        Lnorth = Lground
    else:
        Lsky = ((svfN + svfNveg - 1) * Lsky_allsky) * viktsky * 0.5
        Lveg = SBC * ewall * ((Ta + 273.15) ** 4) * viktveg * 0.5
        Lground = LupN * 0.5
        Lrefl = (Ldown + LupN) * (viktrefl) * (1 - ewall) * 0.5
        Lnorth = Lsky + Lwallsun + Lwallsh + Lveg + Lground + Lrefl

    del LupE,LupS,LupW,LupN,svfalfaE,svfalfaS,svfalfaW,svfalfaN

    del viktveg, viktwall, viktsky, viktrefl

    return Least, Lsouth, Lwest, Lnorth


def Solweig_2022a_calc(i, dsm, scale, rows, cols, svf, svfN, svfW, svfE, svfS, svfveg, svfNveg, svfEveg, svfSveg,
                       svfWveg, svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg, vegdem, vegdem2, albedo_b, absK, absL,
                       ewall, Fside, Fup, Fcyl, altitude, azimuth, zen, jday, usevegdem, onlyglobal, buildings, location, psi,
                       landcover, lc_grid, dectime, altmax, dirwalls, walls, cyl, elvis, Ta, RH, radG, radD, radI, P,
                       amaxvalue, bush, Twater, TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST,
                       TmaxLST_wall, first, second, svfalfa, svfbuveg, firstdaytime, timeadd, timestepdec, Tgmap1,
                       Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, CI, TgOut1, diffsh, shmat, vegshmat, vbshvegshmat, anisotropic_sky, asvf, patch_option):

    t = 0.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert input data to torch tensors
    altitude = torch.tensor(altitude, device=device).clone().detach()
    azimuth = torch.tensor(azimuth, device=device).clone().detach()
    zen = torch.tensor(zen, device=device).clone().detach()
    #lc_grid = torch.tensor(lc_grid, device=device).clone().detach()
    dectime = torch.tensor(dectime, device=device).clone().detach()
    altmax = torch.tensor(altmax, device=device).clone().detach()
    Twater = torch.tensor(Twater, device=device).clone().detach()
    #TgK_wall = torch.tensor(TgK_wall, device=device).clone().detach()
    #Tstart_wall = torch.tensor(Tstart_wall, device=device).clone().detach()
    #TmaxLST = torch.tensor(TmaxLST, device=device).clone().detach()
    #TmaxLST_wall = torch.tensor(TmaxLST_wall, device=device).clone().detach()

    # Stefan Bolzmans Constant
    SBC = torch.tensor(5.67051e-8, device=device).clone().detach()

    # Find sunrise decimal hour - new from 2014a
    _, _, _, SNUP = daylen(torch.tensor(jday.item()), torch.tensor(location['latitude']))

    # Vapor pressure
    ea = 6.107 * 10 ** ((7.5 * Ta) / (237.3 + Ta)) * (RH / 100.)

    # Determination of clear - sky emissivity from Prata (1996)
    msteg = 46.5 * (ea / (Ta + 273.15))
    esky = (1 - (1 + msteg) * torch.exp(-((1.2 + 3.0 * msteg) ** 0.5))) + elvis  # -0.04 old error from Jonsson et al.2006

    if altitude > 0: # # # # # # DAYTIME # # # # # #
        # Clearness Index on Earth's surface after Crawford and Dunchon (1999) with a correction
        #  factor for low sun elevations after Lindberg et al.(2008)
        I0, CI, Kt, I0et, CIuncorr = clearnessindex_2013b(torch.tensor(zen.item()), torch.tensor(jday.item()), torch.tensor(Ta.item()), torch.tensor(RH.item()) / 100., torch.tensor(radG.item()), location, torch.tensor(P.item()))
        CI = min(CI, 1.0)

        # Estimation of radD and radI if not measured after Reindl et al.(1990)
        if onlyglobal == 1:
            I0, CI, Kt, I0et, CIuncorr = clearnessindex_2013b(torch.tensor(zen.item()), torch.tensor(jday.item()), Ta.item(), torch.tensor(RH.item()) / 100., torch.tensor(radG.item()), location, torch.tensor(P.item()))
            CI = min(CI, 1.0)

            radI, radD = diffusefraction(torch.tensor(radG.item()), torch.tensor(altitude.item()), Kt, torch.tensor(Ta.item()), torch.tensor(RH.item()))

        # Diffuse Radiation
        # Anisotropic Diffuse Radiation after Perez et al. 1993
        if anisotropic_sky == 1:
            patchchoice = 1
            zenDeg = zen * (180 / np.pi)
            # Relative luminance
            lv, pc_, pb_ = Perez_v3(zenDeg.item(), azimuth.item(), radD, radI, jday.item(), patchchoice, patch_option)
            # Total relative luminance from sky, i.e. from each patch, into each cell
            aniLum = torch.zeros((rows, cols), device=device)
            for idx in range(lv.shape[0]):
                aniLum += diffsh[:,:,idx] * lv[idx,2]

            dRad = aniLum * radD   # Total diffuse radiation from sky into each cell
        else:
            dRad = radD * svfbuveg
            patchchoice = 1
            lv = None

        # Shadow  images
        if usevegdem == 1:
            vegsh, sh, _, wallsh, wallsun, wallshve, _, facesun = shadowingfunction_wallheight_23(dsm, vegdem, vegdem2,
                                        azimuth.item(), altitude.item(), scale, amaxvalue.item(), bush, walls, dirwalls * np.pi / 180.)
            shadow = sh - (1 - vegsh) * (1 - psi)
        else:
            sh, wallsh, wallsun, facesh, facesun = shadowingfunction_wallheight_13(dsm, azimuth.item(), altitude.item(), scale,
                                                                                   walls, dirwalls * np.pi / 180.)
            shadow = sh

        # # # Surface temperature parameterisation during daytime # # # #
        Tgamp = TgK * altmax + Tstart # Fixed 2021
        Tgampwall = TgK_wall * altmax + Tstart_wall
        Tg = Tgamp * torch.sin((((dectime - torch.floor(dectime)) - SNUP / 24) / (TmaxLST / 24 - SNUP / 24)) * np.pi / 2) # 2015 a, based on max sun altitude
        Tgwall = Tgampwall * torch.sin((((dectime - torch.floor(dectime)) - SNUP / 24) / (TmaxLST_wall / 24 - SNUP / 24)) * np.pi / 2) # 2015a, based on max sun altitude

        Tgwall = torch.maximum(Tgwall, torch.tensor(0, device=device))

        radI0, _ = diffusefraction(I0, altitude.item(), 1., Ta.item(), RH.item())
        corr = 0.1473 * torch.log(90 - (zen / np.pi * 180)) + 0.3454  # 20070329 correction of lat, Lindberg et al. 2008
        CI_Tg = (radG / radI0) + (1 - corr)
        CI_Tg = min(CI_Tg, 1.0)

        deg2rad = np.pi / 180
        radG0 = radI0 * (torch.sin(altitude * deg2rad)) + _
        CI_TgG = (radG / radG0) + (1 - corr)
        CI_TgG = min(CI_TgG, 1.0)

        Tg = Tg * CI_TgG  # new estimation
        Tgwall = Tgwall * CI_TgG
        if landcover == 1:
            Tg = torch.maximum(Tg, torch.tensor(0, device=device))  # temporary for removing low Tg during morning 20130205

        # # # # Ground View Factors # # # #
        gvfLup, gvfalb, gvfalbnosh, gvfLupE, gvfalbE, gvfalbnoshE, gvfLupS, gvfalbS, gvfalbnoshS, gvfLupW, gvfalbW,\
        gvfalbnoshW, gvfLupN, gvfalbN, gvfalbnoshN, gvfSum, gvfNorm = gvf_2018a(wallsun, walls, buildings, scale, shadow, first,
                second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, rows, cols,
                                                                 Twater, lc_grid, landcover)

        # # # # Lup, daytime # # # #
        Lup, timeaddnotused, Tgmap1 = TsWaveDelay_2015a(gvfLup, firstdaytime, timeadd, timestepdec, Tgmap1)
        LupE, timeaddnotused, Tgmap1E = TsWaveDelay_2015a(gvfLupE, firstdaytime, timeadd, timestepdec, Tgmap1E)
        LupS, timeaddnotused, Tgmap1S = TsWaveDelay_2015a(gvfLupS, firstdaytime, timeadd, timestepdec, Tgmap1S)
        LupW, timeaddnotused, Tgmap1W = TsWaveDelay_2015a(gvfLupW, firstdaytime, timeadd, timestepdec, Tgmap1W)
        LupN, timeaddnotused, Tgmap1N = TsWaveDelay_2015a(gvfLupN, firstdaytime, timeadd, timestepdec, Tgmap1N)

        # # For Tg output in POIs
        TgTemp = Tg * shadow + Ta
        TgOut, timeadd, TgOut1 = TsWaveDelay_2015a(TgTemp, firstdaytime, timeadd, timestepdec, TgOut1) #timeadd only here v2021a

        # Building height angle from svf
        F_sh = cylindric_wedge(zen.item(), svfalfa, rows, cols)  # Fraction shadow on building walls based on sun alt and svf
        F_sh[torch.isnan(F_sh)] = 0.5

        # # # # # # # Calculation of shortwave daytime radiative fluxes # # # # # # #
        Kdown = radI * shadow * torch.sin(altitude * (np.pi / 180)) + dRad + albedo_b * (1 - svfbuveg) * \
                            (radG * (1 - F_sh) + radD * F_sh)

        Kup, KupE, KupS, KupW, KupN = Kup_veg_2015a(radI, radD, radG, altitude, svfbuveg, albedo_b, F_sh, gvfalb,
                    gvfalbE, gvfalbS, gvfalbW, gvfalbN, gvfalbnosh, gvfalbnoshE, gvfalbnoshS, gvfalbnoshW, gvfalbnoshN)

        Keast, Ksouth, Kwest, Knorth, KsideI, KsideD, Kside = Kside_veg_v2022a(radI, radD, radG, shadow, svfS, svfW, svfN, svfE,
                    svfEveg, svfSveg, svfWveg, svfNveg, azimuth.item(), altitude.item(), psi, t, albedo_b, F_sh, KupE, KupS, KupW,
                    KupN, cyl, lv, anisotropic_sky, diffsh, rows, cols, asvf, shmat, vegshmat, vbshvegshmat)

        firstdaytime = 0

    else:  # # # # # # # NIGHTTIME # # # # # # # #

        Tgwall = torch.tensor(0, device=device)

        # Nocturnal K fluxes set to 0
        Knight = torch.zeros((rows, cols), device=device)
        Kdown = torch.zeros((rows, cols), device=device)
        Kwest = torch.zeros((rows, cols), device=device)
        Kup = torch.zeros((rows, cols), device=device)
        Keast = torch.zeros((rows, cols), device=device)
        Ksouth = torch.zeros((rows, cols), device=device)
        Knorth = torch.zeros((rows, cols), device=device)
        KsideI = torch.zeros((rows, cols), device=device)
        KsideD = torch.zeros((rows, cols), device=device)
        F_sh = torch.zeros((rows, cols), device=device)
        Tg = torch.zeros((rows, cols), device=device)
        shadow = torch.zeros((rows, cols), device=device)
        CI_Tg = deepcopy(CI)
        CI_TgG = deepcopy(CI)

        dRad = torch.zeros((rows,cols), device=device)

        Kside = torch.zeros((rows,cols), device=device)

        # # # # Lup # # # #
        Lup = SBC * emis_grid * ((Knight + Ta + Tg + 273.15) ** 4)
        if landcover == 1:
            Lup[lc_grid == 3] = (SBC * 0.98 * (Twater + 273.15) ** 4).float()  # nocturnal Water temp

        LupE = Lup
        LupS = Lup
        LupW = Lup
        LupN = Lup

        # # For Tg output in POIs
        TgOut = Ta + Tg

        I0 = 0
        timeadd = 0
        firstdaytime = 1

    # # # # Ldown # # # #
    # Anisotropic sky longwave radiation
    if anisotropic_sky == 1:
        if 'lv' not in locals():
            # Creating skyvault of patches of constant radians (Tregeneza and Sharples, 1993)
            skyvaultalt, skyvaultazi, _, _, _, _, _ = create_patches(patch_option)

            patch_emissivities = torch.zeros(skyvaultalt.shape[0], device=device)

            x = torch.transpose(torch.atleast_2d(skyvaultalt), 0, 1)
            y = torch.transpose(torch.atleast_2d(skyvaultazi), 0, 1)
            z = torch.transpose(torch.atleast_2d(patch_emissivities), 0, 1)

            L_patches = torch.cat((x, y, z), dim=1)

        else:
            L_patches = deepcopy(lv)

        if altitude < 0:
            CI = deepcopy(CI)

        if CI < 0.95:
            esky_c = CI * esky + (1 - CI) * 1.
            esky = esky_c

        Ldown, Lside, Least_, Lwest_, Lnorth_, Lsouth_ \
                  = Lcyl_v2022a(esky, L_patches, Ta, Tgwall, ewall, Lup, shmat, vegshmat, vbshvegshmat,
                                altitude, azimuth, rows, cols, asvf)

    else:
        Ldown = (svf + svfveg - 1) * esky * SBC * ((Ta + 273.15) ** 4) + (2 - svfveg - svfaveg) * ewall * SBC * \
                    ((Ta + 273.15) ** 4) + (svfaveg - svf) * ewall * SBC * ((Ta + 273.15 + Tgwall) ** 4) + \
                    (2 - svf - svfveg) * (1 - ewall) * esky * SBC * ((Ta + 273.15) ** 4)  # Jonsson et al.(2006)

        Lside = torch.zeros((rows, cols), device=device)
        L_patches = None

        if CI < 0.95:  # non - clear conditions
            c = 1 - CI
            Ldown = Ldown * (1 - c) + c * ((svf + svfveg - 1) * SBC * ((Ta + 273.15) ** 4) + (2 - svfveg - svfaveg) *
                    ewall * SBC * ((Ta + 273.15) ** 4) + (svfaveg - svf) * ewall * SBC * ((Ta + 273.15 + Tgwall) ** 4) +
                    (2 - svf - svfveg) * (1 - ewall) * SBC * ((Ta + 273.15) ** 4))

    # # # # Lside # # # #
    Least, Lsouth, Lwest, Lnorth = Lside_veg_v2022a(svfS, svfW, svfN, svfE, svfEveg, svfSveg, svfWveg, svfNveg,
                    svfEaveg, svfSaveg, svfWaveg, svfNaveg, azimuth.item(), altitude.item(), Ta, Tgwall, SBC, ewall, Ldown,
                                                      esky, t, F_sh, CI, LupE, LupS, LupW, LupN, anisotropic_sky)

    # Box and anisotropic longwave
    if cyl == 0 and anisotropic_sky == 1:
        Least += Least_
        Lwest += Lwest_
        Lnorth += Lnorth_
        Lsouth += Lsouth_

    # # # # Calculation of radiant flux density and Tmrt # # # #
    # Human body considered as a cylinder with isotropic all-sky diffuse
    if cyl == 1 and anisotropic_sky == 0:
        Sstr = absK * (KsideI * Fcyl + (Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) + absL * \
                        ((Ldown + Lup) * Fup + (Lnorth + Least + Lsouth + Lwest) * Fside)
    # Human body considered as a cylinder with Perez et al. (1993) (anisotropic sky diffuse)
    # and Martin and Berdahl (1984) (anisotropic sky longwave)
    elif cyl == 1 and anisotropic_sky == 1:
        Sstr = absK * (Kside * Fcyl + (Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) + absL * \
                        ((Ldown + Lup) * Fup + Lside * Fcyl + (Lnorth + Least + Lsouth + Lwest) * Fside)
    # Knorth = nan Ksouth = nan Kwest = nan Keast = nan
    else: # Human body considered as a standing cube
        Sstr = absK * ((Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) + absL * \
                        ((Ldown + Lup) * Fup + (Lnorth + Least + Lsouth + Lwest) * Fside)

    Tmrt = torch.sqrt(torch.sqrt((Sstr / (absL * SBC)))) - 273.2

    # Add longwave to cardinal directions for output in POI
    if (cyl == 1) and (anisotropic_sky == 1):
        Least += Least_
        Lwest += Lwest_
        Lnorth += Lnorth_
        Lsouth += Lsouth_

    return Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, shadow, firstdaytime, timestepdec, \
           timeadd, Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, Keast, Ksouth, Kwest, Knorth, Least, \
           Lsouth, Lwest, Lnorth, KsideI, TgOut1, TgOut, radI, radD, \
               Lside, L_patches, CI_Tg, CI_TgG, KsideD, dRad, Kside

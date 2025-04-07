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

def ensure_tensor(x, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, device=device)
    return x

def shadow(amaxvalue, a, vegdem, vegdem2, bush, azimuth, altitude, scale):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    degrees = torch.pi / 180.
    if azimuth == 0.0:
        azimuth = 1e-12
    azimuth = ensure_tensor(azimuth)
    altitutde = ensure_tensor(altitude)
    azimuth = azimuth * degrees #torch.tensor(azimuth * degrees, device=a.device)
    altitude = altitude * degrees #torch.tensor(altitude * degrees, device=a.device)

    dx = 0.
    dy = 0.
    dz = 0.
    sizex = a.shape[0]
    sizey = a.shape[1]

    device = a.device

    dx = torch.tensor(dx, device=device)
    dy = torch.tensor(dy, device=device)
    dz = torch.tensor(dz, device=device)

    temp = torch.zeros((sizex, sizey), device=device)
    tempvegdem = torch.zeros((sizex, sizey), device=device)
    tempvegdem2 = torch.zeros((sizex, sizey), device=device)
    sh = torch.zeros((sizex, sizey), device=device)
    vbshvegsh = torch.zeros((sizex, sizey), device=device)
    tempbush = torch.zeros((sizex, sizey), device=device)

    f = a.clone()
    g = torch.zeros((sizex, sizey), device=device)
    bushplant = bush > 1.
    vegsh = torch.zeros((sizex, sizey), device=device) + bushplant.float()

    pibyfour = torch.pi / 4.
    threetimespibyfour = 3. * pibyfour
    fivetimespibyfour = 5. * pibyfour
    seventimespibyfour = 7. * pibyfour
    sinazimuth = torch.sin(azimuth)
    cosazimuth = torch.cos(azimuth)
    tanazimuth = torch.tan(azimuth)
    signsinazimuth = torch.sign(sinazimuth)
    signcosazimuth = torch.sign(cosazimuth)
    dssin = torch.abs((1. / sinazimuth))
    dscos = torch.abs((1. / cosazimuth))
    tanaltitudebyscale = torch.tan(altitude) / scale

    index = 1
    #print(type(amaxvalue))
    #%%amaxvalue =  torch.tensor(amaxvalue,  device=device)
    while (amaxvalue >= dz and torch.abs(dx) < sizex and torch.abs(dy) < sizey):
        if (pibyfour <= azimuth < threetimespibyfour or fivetimespibyfour <= azimuth < seventimespibyfour):
            dy = signsinazimuth * index
            dx = -1. * signcosazimuth * torch.abs(torch.round(index / tanazimuth))
            ds = dssin
        else:
            dy = signsinazimuth * torch.abs(torch.round(index * tanazimuth))
            dx = -1. * signcosazimuth * index
            ds = dscos

        dz = ds * index * tanaltitudebyscale

        tempvegdem.zero_()
        tempvegdem2.zero_()
        temp.zero_()
        absdx = torch.abs(dx)
        absdy = torch.abs(dy)
        xc1 = int((dx + absdx) / 2.)
        xc2 = int(sizex + (dx - absdx) / 2.)
        yc1 = int((dy + absdy) / 2.)
        yc2 = int(sizey + (dy - absdy) / 2.)
        xp1 = int(-((dx - absdx) / 2.))
        xp2 = int(sizex - (dx + absdx) / 2.)
        yp1 = int(-((dy - absdy) / 2.))
        yp2 = int(sizey - (dy + absdy) / 2.)

        tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dz
        tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz
        temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz

        f = torch.max(f, temp)
        sh[f > a] = 1.
        sh[f <= a] = 0.

        fabovea = tempvegdem > a
        gabovea = tempvegdem2 > a
        vegsh2 = fabovea.float() - gabovea.float()

        vegsh = torch.max(vegsh, vegsh2)
        vegsh[(vegsh * sh > 0.)] = 0.

        vbshvegsh = vegsh + vbshvegsh

        if index == 1.:
            firstvegdem = tempvegdem - temp
            firstvegdem[firstvegdem <= 0.] = 1000.
            vegsh[firstvegdem < dz] = 1.
            vegsh = vegsh * (vegdem2 > a).float()
            vbshvegsh.zero_()

        if bush.max() > 0. and torch.max(fabovea * bush) > 0.:
            tempbush.zero_()
            tempbush[int(xp1)-1:int(xp2), int(yp1)-1:int(yp2)] = bush[int(xc1)-1:int(xc2), int(yc1)-1:int(yc2)] - dz
            g = torch.max(g, tempbush)
            g *= bushplant

        index += 1.

    sh = 1. - sh
    vbshvegsh[vbshvegsh > 0.] = 1.
    vbshvegsh = vbshvegsh - vegsh

    if bush.max() > 0.:
        g = g - bush
        g[g > 0.] = 1.
        g[g < 0.] = 0.
        vegsh = vegsh - bushplant + g
        vegsh[vegsh < 0.] = 0.

    vegsh[vegsh > 0.] = 1.
    vegsh = 1. - vegsh
    vbshvegsh = 1. - vbshvegsh

    # Changed here
    del tempvegdem, tempvegdem2, temp, tempbush, fabovea, gabovea, vegsh2
    torch.cuda.empty_cache()
    # Changed here
    return sh, vegsh, vbshvegsh

def annulus_weight(altitude, aziinterval, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n = torch.tensor(90.0, device=device)
    altitude = torch.tensor(altitude, device=device)
    #%%aziinterval = torch.tensor(aziinterval, device=device)

    steprad = (360.0 / aziinterval) * (torch.pi / 180.0)
    annulus = 91.0 - altitude
    w = (1.0 / (2.0 * torch.pi)) * torch.sin(torch.pi / (2.0 * n)) * torch.sin((torch.pi * (2.0 * annulus - 1.0)) / (2.0 * n))
    weight = steprad * w

    return weight

def create_patches(patch_option):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deg2rad = torch.pi / 180
    skyvaultalt = torch.tensor([], device=device)
    skyvaultazi = torch.tensor([], device=device)

    if patch_option == 1:
        annulino = torch.tensor([0, 12, 24, 36, 48, 60, 72, 84, 90], device=device)
        skyvaultaltint = torch.tensor([6, 18, 30, 42, 54, 66, 78, 90], device=device)
        azistart = torch.tensor([0, 4, 2, 5, 8, 0, 10, 0], device=device)
        patches_in_band = torch.tensor([30, 30, 24, 24, 18, 12, 6, 1], device=device)
    elif patch_option == 2:
        annulino = torch.tensor([0, 12, 24, 36, 48, 60, 72, 84, 90], device=device)
        skyvaultaltint = torch.tensor([6, 18, 30, 42, 54, 66, 78, 90], device=device)
        azistart = torch.tensor([0, 4, 2, 5, 8, 0, 10, 0], device=device)
        patches_in_band = torch.tensor([31, 30, 28, 24, 19, 13, 7, 1], device=device)
    elif patch_option == 3:
        annulino = torch.tensor([0, 12, 24, 36, 48, 60, 72, 84, 90], device=device)
        skyvaultaltint = torch.tensor([6, 18, 30, 42, 54, 66, 78, 90], device=device)
        azistart = torch.tensor([0, 4, 2, 5, 8, 0, 10, 0], device=device)
        patches_in_band = torch.tensor([31*2, 30*2, 28*2, 24*2, 19*2, 13*2, 7*2, 1], device=device)
    elif patch_option == 4:
        annulino = torch.tensor([0, 4.5, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 90], device=device)
        skyvaultaltint = torch.tensor([3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 90], device=device)
        patches_in_band = torch.tensor([31*2, 31*2, 30*2, 30*2, 28*2, 28*2, 24*2, 24*2, 19*2, 19*2, 13*2, 13*2, 7*2, 7*2, 1], device=device)
        azistart = torch.tensor([0, 0, 4, 4, 2, 2, 5, 5, 8, 8, 0, 0, 10, 10, 0], device=device)

    skyvaultaziint = 360 / patches_in_band

    for j in range(skyvaultaltint.shape[0]):
        for k in range(patches_in_band[j]):
            skyvaultalt = torch.cat((skyvaultalt, torch.tensor([skyvaultaltint[j]], device=device)))
            skyvaultazi = torch.cat((skyvaultazi, torch.tensor([k * skyvaultaziint[j] + azistart[j]], device=device)))

    return skyvaultalt, skyvaultazi, annulino, skyvaultaltint, patches_in_band, skyvaultaziint, azistart


def svf_calculator(patch_option,amaxvalue, a, vegdem, vegdem2, bush, scale):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = a.device
    rows = a.shape[0]
    cols = a.shape[1]

    svf = torch.zeros([rows, cols], device=device)
    svfE = torch.zeros([rows, cols], device=device)
    svfS = torch.zeros([rows, cols], device=device)
    svfW = torch.zeros([rows, cols], device=device)
    svfN = torch.zeros([rows, cols], device=device)
    svfveg = torch.zeros((rows, cols), device=device)
    svfEveg = torch.zeros((rows, cols), device=device)
    svfSveg = torch.zeros((rows, cols), device=device)
    svfWveg = torch.zeros((rows, cols), device=device)
    svfNveg = torch.zeros((rows, cols), device=device)
    svfaveg = torch.zeros((rows, cols), device=device)
    svfEaveg = torch.zeros((rows, cols), device=device)
    svfSaveg = torch.zeros((rows, cols), device=device)
    svfWaveg = torch.zeros((rows, cols), device=device)
    svfNaveg = torch.zeros((rows, cols), device=device)

    skyvaultalt, skyvaultazi, annulino, skyvaultaltint, aziinterval, skyvaultaziint, azistart = create_patches(patch_option)
    skyvaultaziint = torch.tensor([360 / patches for patches in aziinterval], device=device)
    iazimuth = torch.zeros((1, torch.sum(aziinterval).item()), device=device)

    shmat = torch.zeros((rows, cols, sum(aziinterval)), device=device)
    vegshmat = torch.zeros((rows, cols, sum(aziinterval)), device=device)
    vbshvegshmat = torch.zeros((rows, cols, sum(aziinterval)), device=device)

    index = 0
    for j in range(skyvaultaltint.shape[0]):
        for k in range(int(360 / skyvaultaziint[j])):
            iazimuth[0, index] = k * skyvaultaziint[j] + azistart[j]
            if iazimuth[0, index] > 360.:
                iazimuth[0, index] = iazimuth[0, index] - 360.
            index += 1

    aziintervalaniso = torch.ceil(aziinterval / 2.0)

    index = 0
    for i in range(skyvaultaltint.shape[0]):
        for j in range(aziinterval[i].int()):
            altitude = skyvaultaltint[i]
            azimuth = iazimuth[0, index]
            sh, vegsh, vbshvegsh = shadow(amaxvalue, a, vegdem, vegdem2, bush, azimuth, altitude, scale)

            vegshmat[:, :, index] = vegsh
            vbshvegshmat[:, :, index] = vbshvegsh
            shmat[:, :, index] = sh

            for k in range(annulino[i]+1, annulino[i+1]+1):
                weight = annulus_weight(k, aziinterval[i], device) * sh
                svf = svf + weight
                weight = annulus_weight(k, aziintervalaniso[i], device) * sh
                if 0 <= azimuth < 180:
                    svfE = svfE + weight
                if 90 <= azimuth < 270:
                    svfS = svfS + weight
                if 180 <= azimuth < 360:
                    svfW = svfW + weight
                if 270 <= azimuth < 90:
                    svfN = svfN + weight

                weight = annulus_weight(k, aziinterval[i], device)
                svfveg = svfveg + weight * vegsh
                svfaveg = svfaveg + weight * vbshvegsh
                weight = annulus_weight(k, aziintervalaniso[i], device)
                if 0 <= azimuth < 180:
                    svfEveg = svfEveg + weight * vegsh
                    svfEaveg = svfEaveg + weight * vbshvegsh
                if 90 <= azimuth < 270:
                    svfSveg = svfSveg + weight * vegsh
                    svfSaveg = svfSaveg + weight * vbshvegsh
                if 180 <= azimuth < 360:
                    svfWveg = svfWveg + weight * vegsh
                    svfWaveg = svfWaveg + weight * vbshvegsh
                if 270 <= azimuth < 90:
                    svfNveg = svfNveg + weight * vegsh
                    svfNaveg = svfNaveg + weight * vbshvegsh

            index += 1
    svfS = svfS + 3.0459e-004
    svfW = svfW + 3.0459e-004
    svf[svf > 1.] = 1.
    svfE[svfE > 1.] = 1.
    svfS[svfS > 1.] = 1.
    svfW[svfW > 1.] = 1.
    svfN[svfN > 1.] = 1.

    last = torch.zeros((rows, cols), device=device)
    last[vegdem2 == 0.] = 3.0459e-004
    svfSveg = svfSveg + last
    svfWveg = svfWveg + last
    svfSaveg = svfSaveg + last
    svfWaveg = svfWaveg + last
    svfveg[svfveg > 1.] = 1.
    svfEveg[svfEveg > 1.] = 1.
    svfSveg[svfSveg > 1.] = 1.
    svfWveg[svfWveg > 1.] = 1.
    svfNveg[svfNveg > 1.] = 1.
    svfaveg[svfaveg > 1.] = 1.
    svfEaveg[svfEaveg > 1.] = 1.
    svfSaveg[svfSaveg > 1.] = 1.
    svfWaveg[svfWaveg > 1.] = 1.
    svfNaveg[svfNaveg > 1.] = 1.

    #Changed here
    del sh, vegsh,vbshvegsh, last, weight
    torch.cuda.empty_cache()
    #Changed here

    return svf, svfaveg, svfE, svfEaveg, svfEveg, svfN, svfNaveg, svfNveg, svfS, svfSaveg, svfSveg, svfveg, svfW, svfWaveg, svfWveg, vegshmat, vbshvegshmat, shmat

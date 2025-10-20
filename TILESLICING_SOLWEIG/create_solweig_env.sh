#!/bin/bash
set -e

# Nome dell'ambiente
ENV_NAME=solweig_env

echo ">>> Creazione nuovo environment: $ENV_NAME"
conda create -y -n $ENV_NAME \
    python=3.10 \
    pytorch=2.5.* \
    pytorch-cuda=12.4 \
    torchvision \
    torchaudio \
    cudnn \
    numpy \
    pandas \
    scipy \
    gdal \
    rasterio \
    shapely \
    geopandas \
    pyproj \
    xarray \
    netcdf4 \
    timezonefinder \
    --override-channels -c conda-forge -c nvidia -c pytorch

echo ">>> Attivo environment e installo pacchetti pip..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

pip install earthengine-api

echo ">>> Ambiente $ENV_NAME creato e configurato!"


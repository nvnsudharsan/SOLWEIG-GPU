---
title: 'SOLWEIG-GPU: GPU-Accelerated Thermal Comfort Modeling Framework for Urban Digital Twins'
tags:
  - Python
  - Thermal Comfort
  - Heat Stress
  - GPU
  - Urban
authors:
  - name: Harsh G. Kamath
    orcid: 0000-0002-5210-8369
    equal-contrib: true
    corresponding: true
    affiliation: 1 
  - name: Naveen Sudharsan
    orcid: 0000-0002-1328-110X
    equal-contrib: true
    corresponding: true
    affiliation: 1
  - name: Manmeet Singh
    orcid: 0000-0002-3374-7149
    affiliation: 1
  - name: Nils Wallenberg
    orcid: 0000-0003-1359-658X
    affiliation: 2
  - name: Fredrik Lindberg
    orcid: 0000-0002-9648-4542
    affiliation: 2
  - name: Dev Niyogi
    orcid: 0000-0002-1848-5080
    corresponding: true
    affiliation: "1, 3" 
affiliations:
 - name: Jackson School of Geosciences, The University of Texas at Austin, USA
   index: 1
 - name: Department of Earth Sciences, University of Gothenburg, Sweden
   index: 2
 - name: Cockrell School of Engineering, The University of Texas at Austin, USA
   index: 3
date: 3 September 2025
bibliography: paper.bib
---

# Summary

We present Solar and LongWave Environmental Irradiance Geometry-Graphics Processing Unit (SOLWEIG-GPU) as a Python package that provides a single-line code, command-line interface (CLI) and a graphical user interface (GUI) for executing the SOLWEIG model version 2022a on GPUs. The package facilitates GPU parallelized human thermal comfort modeling at meter-scale resolution across city-scale domains by computing key variables such as sky-view factor (SVF), mean radiant temperature (T<sub>MRT</sub>), ground shading, and the universal thermal climate index (UTCI). Sample data and model instructions can be found in the repository.

# Statement of need

The original SOLWEIG model [@lindberg2008solweig] was developed to calculate T<sub>MRT</sub> over small geographical areas in cities. At this spatial scale, based on the time required for computation, the model could be run on CPUs [@kamath2023human]. However, for city-scale thermal comfort estimation, the model can be accelerated using a GPU. Specifically, the calculation of the SVF (which is the most time-consuming step), T<sub>MRT</sub> and UTCI can be processed on a GPU. Currently, there is a tool that computes SVF on GPU but it uses Python to read the inputs and write the output rasters, while the SVF calculation on GPU is done by interfacing with a C-language code [@li2021gpu]. Thus, there was a need for a Python-based end-to-end framework to run SOLWEIG on a GPU. Our framework is implemented in PyTorch, which enables automatic selection of the GPU when available. 


# Functionality 
SOLWEIG-GPU requires the following inputs: (i) Building digital surface model (DSM) that includes both buildings and terrain height (*m*), (ii) Digital elevation model (DEM) that is the bare ground elevation (*m*), (iii). Tree or vegetation DSM that only represents the vegetation height (*m*) (iv). UMEP [@lindberg2018umep] style ground cover (optional), and (v) meteorological forcing. Allowed land cover types are asphalt or paved, buildings, grass, bare soil, and water but users can add their own land cover types provided they know the thermal properties for the land surface (e.g., albedo and emissivity). All the input datasets must be of the same spatial resolution, projection, and spatial extent. The recommended projection is the Universal Transverse Mercator (UTM). For example, EPSG: 32614. Necessary meteorological variables for SOLWEIG are: (i) 2-meter air temperature (*℃*), (ii) relative humidity (*%*), (iii) barometric pressure (*kPa*), (iv) downwelling shortwave radiation (*W/m<sup>2</sup>*). Additionally, near-surface wind speed (*m/s*) is required for UTCI computation and longwave radiation is estimated using 2-meter air temperature, relative humidity and pressure. 

![Different steps involved in calculation of `thermal comfort` using SOLWEIG-GPU: CPU and GPU based calculations in SOLWEIG-GPU are shown.](figures/figure1.png)

Figure 1 shows the workflow of SOLWEIG-GPU and detailed description of functionalities are as follows:
  1. SOLWEIG-GPU can divide a larger geographical domain into smaller tiles and create separate meteorological forcing for each of the tiles. 
  2. If **torch.cuda.is_available()** returns **‘True’**, the simulations will be performed on a GPU.
  3. The calculations of wall height and aspect (wall directional orientation) are faster on CPU as they use an *i* and *j* indices to loop through the input rasters. Thus, we have parallelized this operation and calculations are performed on multiple CPUs at once.
  4. SVF, T<sub>MRT</sub>, UTCI and ground shading are computed on the GPU, if available. Additionally, the model can output shortwave and longwave radiations in both upward and downward directions. The users can select the required output variables from SOLWEIG-GPU but UTCI is outputted by default. Note that calculation of UTCI uses wind speed from the meteorological forcing but more advanced methods to calculate wind speeds in urban areas are available [@bernard2023urock]. On clear, hot days T<sub>MRT</sub> is a dominant driver, yet wind can still change UTCI by several kelvin, so relying on grid-averaged wind may not always have a minimal impact.
  5. Ground cover classes can be optionally provided and they are used to set-up the grid for the surface thermal properties [@lindberg2016ground].
  6. SOLWEIG-GPU can only work with hourly meteorological data at present. However, the time period of simulation is based on the meteorological forcing data provided. The model can accept the meteorological forcing in three ways: (i) Output from MetProcessor tool in UMEP (Lindberg et al., 2018), (ii) Gridded ERA-5 reanalysis and (iii) Gridded output from Weather Research and Forecasting (WRF) model. 


# Usage 

SOLWEIG-GPU can be run using python code, CLI or GUI depending on the code adaptation. We explain how to run SOLWEIG-GPU in python below. Examples for running the model using all three methods are provided in the repository. 

![A single line Python code to call the function `thermal_comfort`.](figures/figure2.png)

In the example shown in Figure 2, *selected_ date_str* is the day of the simulation, landcover raster is used if *landcover_filename* is provided (else set to **None**), tile_size is set to 3600 meaning the tiles will be 3600 x 3600 pixels, *overlap* is set to 100 meaning there will be 100 pixels that overlap between the tiles for shadow transfer between the tiles, meteorological data created using UMEP is used (*thus use_own_met* = **True**), *use_own_file* is the path to the UMEP meteorological forcing, and lastly only UTCI rasteres are saved (default). The rest of the inputs shown in Figure 2 are optional if using *use_own_met* = **True**. However, if gridded datasets are to be used, *use_own_met* = **False**. Additionally, the source of the gridded files must be mentioned (*ERA5* or *wrfout*) and the folder path where gridded data is stored should be provided in *data_folder*. The *start_time* and *end_time* are the first and last hour (in UTC) of meteorological data in the gridded datasets. UTC to local time conversion is done within the program.

When ERA-5 is chosen for meteorological forcing, the model expects two files in the *data_folder* that have instantaneous and accumulated data variables: 
  1. Instantaneous variables: 2-meter temperature (*t2m*), 2-meter dew point temperature (*d2m*),  winds in meridional and zonal directions (*u10* and *v10*) and surface pressure (*sp*).
  2. Accumulated variables: Surface downwelling shortwave radiation (*ssrd*) and surface downwelling thermal radiation (*strd*).
     
Model outputs will be stored by creating a new folder *Outputs*. In this folder, sub-folders will be created for each of the tile outputs. For T<sub>MRT</sub>, UTCI, shortwave and longwave radiations in up and down directions and shadows, a raster will be created that will have n bands, where n is the number of hours of simulation. For SVF, a single band raster will be created. 

# Comparison with SOLWEIG - CPU
UTCI simulations were run with different tile sizes both on CPU and GPU. The CPU-based simulations were run on a Windows 11 machine with an 10th generation Intel Core i7 (i7-10700) and 16 Gigabytes (GB) of Random-Access Memory (RAM). The GPU-based simulations were run on an Ubuntu machine with NVIDIA A6000 GPU with 48 GB vRAM. Table 1 below reports the average time taken for the UTCI calculations (mean of 4-5 simulations with the same tile size). Note that Table 1 only reports the time for SVF and UTCI calculations as the calculations for wall height and aspect are CPU-based.

Table 1. Comparison of time taken for UTCI computation using CPU and GPU-based machines for different tile sizes.

| Tile size     | CPU-based               | GPU-based | GPU acceleration |
|---------------|-------------------------|-----------|------------------|
| 1000 x 1000   | 1187 seconds (0.33 hrs) | 47 seconds| ~25×             |
| 1500 x 1500   | 3322 seconds (0.92 hrs) | 105 seconds| ~32×            |
| 2000 x 2000   | 6487 seconds (1.8 hrs)  | 158 seconds| ~41×            |


# Funding

This research is supported by National Oceanic and Atmospheric Administration (NOAA) National Integrated Heat Health Information System (NIHHIS) [NA21OAR4310146], National Science Foundation (NSF) Workshop on Atmospheric and Urban Digital Twins [2324744],  National Aeronautics and Space Administration (NASA) Austin Cares [80NSSC25K7417] and National Institute of Standards and Technology [60NANB24D235]

# References

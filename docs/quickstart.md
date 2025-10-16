># Quickstart

This guide will walk you through the process of running your first thermal comfort simulation using SOLWEIG-GPU. We will use the sample data provided with the package to get you started as quickly as possible.

## 1. Download Sample Data

The first step is to download the sample dataset. This dataset contains the required input rasters and meteorological forcing data. You can download it from the following Zenodo repository:

<a href="https://doi.org/10.5281/zenodo.17048978"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.17048978.svg" alt="DOI"></a>

Once downloaded, extract the contents of the zip file to a location of your choice. For this guide, we will assume you have extracted it to `/path/to/sample_data`.

## 2. Running a Simulation with the CLI

The easiest way to run a simulation is by using the command-line interface (CLI). Here is an example of how to run a simulation using the sample ERA5 data.

Open your terminal, activate the `solweig` conda environment, and run the following command. Make sure to replace `/path/to/sample_data` with the actual path to your extracted sample data.

```bash
conda activate solweig

thermal_comfort --base_path '/path/to/sample_data/Input_raster' ^
                --date '2020-08-13' ^
                --building_dsm 'Building_DSM.tif' ^
                --dem 'DEM.tif' ^
                --trees 'Trees.tif' ^
                --tile_size 1000 ^
                --landcover  'Landcover2.tif' ^
                --overlap 100 ^
                --use_own_met False ^
                --data_source_type 'ERA5' ^
                --data_folder '/path/to/sample_data/Forcing_data/ERA5' ^
                --start '2020-08-13 06:00:00' ^
                --end '2020-08-13 23:00:00' ^
                --save_tmrt True
```

## 3. Running a Simulation with the GUI

If you prefer a graphical interface, you can use the SOLWEIG-GPU GUI.

1.  **Launch the GUI**

    ```bash
    conda activate solweig
    solweig_gpu_gui
    ```

2.  **Configure the Simulation**

    -   **Base Path**: Select the `Input_raster` directory from the sample data.
    -   **Rasters**: Select the `Building_DSM.tif`, `DEM.tif`, and `Trees.tif` files.
    -   **Tile Size**: Set the tile size to `1000`.
    -   **Meteorological Source**: Select `ERA5` and point to the `Forcing_data/ERA5` directory.
    -   **Time**: Set the start and end times as in the CLI example.
    -   **Outputs**: Check the outputs you want to generate (e.g., Tmrt).

3.  **Run the Simulation**

    Click the "Run" button to start the simulation.

## 4. Checking the Outputs

After the simulation is complete, you will find the output files in the `Outputs` directory created within your base path. The outputs will be organized in subdirectories for each tile (e.g., `tile_0_0`).

The output files are multi-band GeoTIFFs, which can be opened with GIS software such as QGIS or ArcGIS to visualize the results.


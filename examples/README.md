># SOLWEIG-GPU Usage Examples

This directory contains example scripts and tutorials to help you get started with SOLWEIG-GPU. The examples demonstrate how to run simulations with different types of meteorological data.

## Available Examples

-   `run_with_wrf.py`: Demonstrates how to run a simulation using meteorological data from WRF (Weather Research and Forecasting) model output files.
-   `run_with_era5.py`: Shows how to use ERA5 reanalysis data as the meteorological forcing for a simulation.
-   `run_with_custom_met_file.py`: Provides an example of how to use your own custom meteorological data in the UMEP text file format.
-   `Example_ERA5.ipynb`, `Example_ownmetfile.ipynb`, `Example_wrfout.ipynb`: Jupyter notebooks for ERA5, custom met file, and WRF forcing (also available under [docs/notebooks](../docs/notebooks)).

## Before You Begin

Before running these examples, make sure you have:

1.  **Installed SOLWEIG-GPU**: Follow the instructions in the [installation guide](../docs/installation.md).
2.  **Downloaded the Sample Data**: The examples are designed to work with the official sample dataset. You can download it from [Zenodo](https://doi.org/10.5281/zenodo.18283037) (DOI: 10.5281/zenodo.18283037).

## How to Run the Examples

1.  **Activate your Conda environment**:

    ```bash
    conda activate solweig
    ```

2.  **Edit the script**: Open the example script you want to run and update the file paths to point to your sample data directories. For example, in `run_with_wrf.py`, you will need to set the `base_path` and `data_folder` variables.

3.  **Run the script**:

    ```bash
    python run_with_wrf.py
    ```

## Jupyter Notebooks

To run the example notebooks, install Jupyter in your environment:

```bash
conda install jupyterlab
```

Launch JupyterLab and open one of the example notebooks (`Example_ERA5.ipynb`, `Example_ownmetfile.ipynb`, or `Example_wrfout.ipynb`):

```bash
jupyter lab
```

Follow the instructions in each notebook to run a simulation and visualize the outputs.


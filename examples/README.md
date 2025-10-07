># SOLWEIG-GPU Usage Examples

This directory contains example scripts and tutorials to help you get started with SOLWEIG-GPU. The examples demonstrate how to run simulations with different types of meteorological data.

## Available Examples

-   `run_with_wrf.py`: Demonstrates how to run a simulation using meteorological data from WRF (Weather Research and Forecasting) model output files.
-   `run_with_era5.py`: Shows how to use ERA5 reanalysis data as the meteorological forcing for a simulation.
-   `run_with_custom_met_file.py`: Provides an example of how to use your own custom meteorological data in the UMEP text file format.
-   `notebooks/interactive_simulation.ipynb`: A Jupyter notebook that provides an interactive, step-by-step guide to running a simulation and visualizing the results.

## Before You Begin

Before running these examples, make sure you have:

1.  **Installed SOLWEIG-GPU**: Follow the instructions in the [installation guide](../docs/installation.md).
2.  **Downloaded the Sample Data**: The examples are designed to work with the official sample dataset. You can download it from [Zenodo](https://doi.org/10.5281/zenodo.17048978).

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

## Jupyter Notebook

To run the interactive notebook, you will need to have Jupyter Notebook or JupyterLab installed in your Conda environment:

```bash
conda install jupyterlab
```

Then, you can launch JupyterLab and open the `interactive_simulation.ipynb` notebook:

```bash
jupyter lab
```

Follow the instructions within the notebook to run the simulation and visualize the outputs.


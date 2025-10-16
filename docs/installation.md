# Installation

This guide provides instructions for installing SOLWEIG-GPU and its dependencies. As the package relies on specific versions of CUDA and GDAL, it is highly recommended to use Conda for environment management.

## Prerequisites

- **NVIDIA GPU**: A CUDA-enabled NVIDIA GPU is required for GPU acceleration. The model will automatically fall back to CPU if no compatible GPU is found.
- **NVIDIA Drivers**: Ensure you have the latest NVIDIA drivers installed for your GPU.
- **Conda**: It is recommended to use either Anaconda or Miniconda to manage the environment and dependencies.

## Installation Steps

1.  **Create a Conda Environment**

    First, create a new Conda environment with Python 3.10. This ensures that the package dependencies do not conflict with other Python projects on your system.

    ```bash
    conda create -n solweig python=3.10
    ```

2.  **Activate the Environment**

    Activate the newly created environment before proceeding with the installation.

    ```bash
    conda activate solweig
    ```

3.  **Install Dependencies**

    Install the required dependencies using Conda. This step is crucial as it installs the correct versions of GDAL, PyTorch, and other libraries that are compatible with each other.

    ```bash
    conda install -c conda-forge gdal pytorch timezonefinder matplotlib sip
    pip install pyqt5
    conda install -c conda-forge cudnn # for GPU
    ```

4.  **Install SOLWEIG-GPU**

    Finally, install the SOLWEIG-GPU package from PyPI using pip.

    ```bash
    pip install solweig-gpu
    ```

## Verifying the Installation

To verify that the package has been installed correctly, you can run the following command to display the help message for the command-line interface:

```bash
thermal_comfort --help
```

If the installation was successful, you should see a list of available command-line options. You can also launch the GUI to confirm that the graphical interface is working:

```bash
solweig_gpu_gui
```

If the GUI window appears, the installation is complete and you are ready to start using SOLWEIG-GPU.


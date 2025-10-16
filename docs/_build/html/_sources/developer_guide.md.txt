># Developer Guide

This guide is intended for developers who want to understand the architecture of SOLWEIG-GPU and contribute to its development.

## Architecture Overview

SOLWEIG-GPU is a modular package that separates the different stages of the thermal comfort modeling process. The main components are:

-   **Data Preprocessing**: Handles the validation, tiling, and extraction of input data.
-   **Geometry Processing**: Calculates wall heights and aspects from the input rasters.
-   **Core SOLWEIG Model**: The main radiation and thermal comfort calculation engine, accelerated with PyTorch.
-   **Interfaces**: Provides both a command-line interface (CLI) and a graphical user interface (GUI) for user interaction.

## Module Interactions

The `solweig_gpu.py` module acts as the main orchestrator, calling the other modules in the correct sequence. The `utci_process.py` module is the core of the computation, bringing together the various components to calculate the final UTCI values.

## Contributing to SOLWEIG-GPU

We welcome contributions to the SOLWEIG-GPU project. To contribute, please follow these steps:

1.  **Fork the Repository**: Create your own fork of the [SOLWEIG-GPU repository](https://github.com/nvnsudharsan/SOLWEIG-GPU) on GitHub.
2.  **Create a Feature Branch**: Create a new branch for your feature or bug fix.
3.  **Commit Your Changes**: Make your changes and commit them with clear and concise commit messages.
4.  **Push to Your Branch**: Push your changes to your forked repository.
5.  **Create a Pull Request**: Open a pull request from your branch to the `main` branch of the original repository.

Please ensure that your pull requests are small and focused on a single feature or bug fix. This makes it easier to review and merge your contributions.


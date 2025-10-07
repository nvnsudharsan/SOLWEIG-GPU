# Welcome to SOLWEIG-GPU

<p align="center">
  <img src="https://raw.githubusercontent.com/nvnsudharsan/solweig-gpu/main/Logo_solweig.jpg" alt="SOLWEIG Logo" width="400"/>
</p>

**SOLWEIG-GPU** is a powerful Python package for GPU-accelerated urban thermal comfort modeling. It enables high-resolution microclimate analysis by computing key variables such as Sky View Factor (SVF), Mean Radiant Temperature (Tmrt), and the Universal Thermal Climate Index (UTCI).

## Overview

SOLWEIG-GPU is designed to provide researchers, urban planners, and climate scientists with a fast and efficient tool for analyzing urban thermal environments. The package leverages GPU acceleration through PyTorch to process large urban areas quickly, making it practical for city-scale thermal comfort assessments.

The model was originally developed by Dr. Fredrik Lindberg's group and has been extended with GPU acceleration capabilities. This implementation maintains compatibility with the original SOLWEIG model while providing significant performance improvements through parallel computing.

## Key Features

**GPU Acceleration** — The package leverages PyTorch for high-performance computation on NVIDIA GPUs, with automatic fallback to CPU when GPU is not available.

**Comprehensive Modeling** — Calculates SVF, Tmrt, UTCI, and detailed shortwave and longwave radiation fluxes for complete thermal environment characterization.

**Flexible Data Input** — Supports multiple meteorological data sources including custom text files (UMEP format), ERA5 reanalysis data, and WRF model outputs.

**Intelligent Tiling** — Automatically divides large study areas into manageable tiles with configurable overlap to handle memory constraints and ensure seamless results.

**Multiple Interfaces** — Offers both a command-line interface for scripting and automation, and a graphical user interface for interactive use.

**Parallel Processing** — CPU-based computations (wall heights and aspects) are parallelized across multiple cores for optimal performance.

## Quick Links

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Get started by installing SOLWEIG-GPU and its dependencies

    [:octicons-arrow-right-24: Installation Guide](installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quickstart**

    ---

    Run your first thermal comfort simulation in minutes

    [:octicons-arrow-right-24: Quickstart Guide](quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **User Guide**

    ---

    Learn how to prepare data and configure simulations

    [:octicons-arrow-right-24: User Guide](user_guide.md)

-   :material-code-braces:{ .lg .middle } **API Reference**

    ---

    Detailed documentation of functions and parameters

    [:octicons-arrow-right-24: API Reference](api_reference.md)

-   :material-file-code:{ .lg .middle } **Examples**

    ---

    Explore example scripts for different use cases

    [:octicons-arrow-right-24: Examples](examples.md)

-   :material-account-group:{ .lg .middle } **Developer Guide**

    ---

    Contribute to the project and understand the architecture

    [:octicons-arrow-right-24: Developer Guide](developer_guide.md)

</div>

## Citation

If you use SOLWEIG-GPU in your research, please cite:

**Original SOLWEIG Model:**
> Lindberg, F., Holmer, B. & Thorsson, S. SOLWEIG 1.0 – Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings. *Int J Biometeorol* 52, 697–713 (2008). [https://doi.org/10.1007/s00484-008-0162-7](https://doi.org/10.1007/s00484-008-0162-7)

**UMEP Framework:**
> Lindberg, F., Grimmond, C.S.B., Gabey, A., et al. Urban Multi-scale Environmental Predictor (UMEP): An integrated tool for city-based climate services. *Environmental Modelling & Software*, 99, pp.70-87 (2018). [https://doi.org/10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)

## License

SOLWEIG-GPU is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/main/LICENSE) file for details.


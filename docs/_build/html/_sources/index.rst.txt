SOLWEIG-GPU Documentation
=========================

GPU-accelerated SOLWEIG model for urban thermal comfort simulation.

**SOLWEIG-GPU** is a high-performance implementation of the Solar and Longwave Environmental Irradiance Geometry (SOLWEIG) model, 
designed for calculating mean radiant temperature (Tmrt) and Universal Thermal Climate Index (UTCI) in complex urban environments.

Features
--------

* **GPU Acceleration**: Leverages CUDA/PyTorch for 10-100x speedup over CPU
* **Large Domain Support**: Handles city-scale simulations through intelligent tiling
* **Multiple Data Sources**: Supports ERA5, WRF, and custom meteorological inputs
* **Complete 3D Geometry**: Accounts for buildings, vegetation, and terrain
* **Parallel Processing**: Multi-core CPU processing for wall calculations
* **High Accuracy**: Implements latest SOLWEIG 2022a algorithms with anisotropic radiation

Quick Start
-----------

Installation
^^^^^^^^^^^^

.. code-block:: bash

   # Using conda (recommended)
   conda create -n solweig python=3.10
   conda activate solweig
   conda install -c conda-forge gdal pytorch
   pip install solweig-gpu

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from solweig_gpu import thermal_comfort
   
   thermal_comfort(
       base_path='/path/to/data',
       selected_date_str='2020-08-13',
       tile_size=1000,
       use_own_met=True,
       own_met_file='met_data.txt'
   )

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   input_data
   configuration
   outputs
   examples

.. toctree::
   :maxdepth: 2
   :caption: Additional Resources:

   testing
   developer_guide

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


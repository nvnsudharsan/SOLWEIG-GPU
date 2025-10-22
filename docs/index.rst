SOLWEIG-GPU: GPU-Accelerated Thermal Comfort Modeling Framework for Urban Digital Twins
=========================

.. image:: https://readthedocs.org/projects/solweig-gpu/badge/?version=latest
    :target: https://solweig-gpu.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/solweig-gpu.svg
    :target: https://pypi.org/project/solweig-gpu/
    :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/solweig-gpu.svg
    :target: https://pypi.org/project/solweig-gpu/
    :alt: Python versions

.. image:: https://img.shields.io/github/stars/nvnsudharsan/solweig-gpu?style=social
    :target: https://github.com/nvnsudharsan/solweig-gpu
    :alt: GitHub stars


**SOLWEIG-GPU** is a high-performance implementation of the Solar and Longwave Environmental Irradiance Geometry (SOLWEIG) model, 
designed for calculating Sky View Factor (SVF), mean radiant temperature (Tmrt), Universal Thermal Climate Index (UTCI), shadows, and short and long-wave radiation in urban environments. 

The original `SOLWEIG model <https://umep-docs.readthedocs.io/en/latest/OtherManuals/SOLWEIG.html>`_ was developed to calculate TMRT over small geographical areas in cities. At this spatial scale, given the time required for computation, the model can be run on CPUs. However, for city-scale thermal comfort estimation, the model should be accelerated using a GPU. Specifically, the calculation of the SVF (the most time-consuming step), TMRT, and UTCI can be processed on a GPU. 

Currently, a tool exists that computes SVF on GPU (`gpusvf <https://link.springer.com/article/10.1007/s00704-021-03692-z>`_). However, it utilizes Python to read the inputs and write the output rasters, whereas the SVF calculation on the GPU is performed by interfacing with a C-language code. Thus, there was a need for a Python-based end-to-end framework to run SOLWEIG on a GPU. Our framework is implemented in PyTorch, which enables automatic selection of the GPU when available.

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
   # Install dependencies via conda
   conda install -c conda-forge gdal pytorch timezonefinder matplotlib sip 
   pip install PyQt5
   conda install -c conda-forge cudnn #If GPU is available
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
   :caption: User Guide:

   input_data
   configuration
   outputs

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   notebooks
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


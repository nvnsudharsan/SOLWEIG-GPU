# Installation

## Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, but recommended for performance)
- GDAL library (libgdal) with matching Python bindings

> **Note:** The GDAL Python bindings (`osgeo`) must be built against the same
> `libgdal` version available on the system. Installing GDAL via `pip` in
> environments with older system GDAL (common on HPC / institutional systems)
> may result in import errors such as `_gdal` or `_gdal_array`.

## Using Conda (Recommended)

Conda handles the complex GDAL and PyTorch dependencies automatically:

```bash
# Create a new environment
conda create -n solweig python=3.10
conda activate solweig

# Install dependencies via conda
conda install -c conda-forge gdal pytorch timezonefinder matplotlib sip 
pip install PyQt5
conda install -c conda-forge cudnn #If GPU is available

# Install SOLWEIG-GPU
pip install solweig-gpu

# If you have installed an older version
pip install --upgrade solweig-gpu
```

## Using pip with system GDAL

If you have GDAL and Pytorch installed system-wide:

```bash
# Install SOLWEIG-GPU
pip install solweig-gpu
```

## Development Installation

For contributing or development:

```bash
# Clone the repository instead of 'pip install solweig-gpu'
git clone https://github.com/nvnsudharsan/SOLWEIG-GPU.git
cd SOLWEIG-GPU

# Install in editable mode
pip install -e .
```

## Verify Installation

```python
import solweig_gpu
print(solweig_gpu.__version__)

# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")

# Verify GDAL NumPy support
python -c "from osgeo import gdal_array"
```

## GPU Setup

### CUDA Requirements

- CUDA 11.0 or higher
- Compatible NVIDIA GPU 
- Sufficient GPU memory (4GB minimum, 8GB+ recommended)

### CPU-Only Mode

SOLWEIG-GPU automatically falls back to CPU if no GPU is detected, though performance will be significantly slower.

## Common Issues


### GDAL Import Error

Errors such as:

- `ModuleNotFoundError: No module named '_gdal'`
- `ModuleNotFoundError: No module named '_gdal_array'`

usually indicate that the GDAL Python bindings do not match the system
`libgdal`, or that NumPy support (`gdal_array`) is missing.

**Verification:**
```bash
python -c "from osgeo import gdal_array"
```
### PyTorch GPU Not Detected

Verify CUDA installation:

```bash
nvidia-smi  # Check GPU driver
python -c "import torch; print(torch.cuda.is_available())"
```

If False, reinstall PyTorch with CUDA:

```bash
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Input Data Preparation](input_data.md)
- [API Reference](api.rst)

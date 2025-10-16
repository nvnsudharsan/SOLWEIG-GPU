# Installation

## Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, but recommended for performance)
- GDAL library

## Method 1: Using Conda (Recommended)

Conda handles the complex GDAL and PyTorch dependencies automatically:

```bash
# Create a new environment
conda create -n solweig python=3.10
conda activate solweig

# Install dependencies via conda
conda install -c conda-forge gdal pytorch

# Install SOLWEIG-GPU
pip install solweig-gpu
```

## Method 2: Using pip with system GDAL

If you have GDAL installed system-wide:

```bash
# Install SOLWEIG-GPU
pip install solweig-gpu
```

## Method 3: Development Installation

For contributing or development:

```bash
# Clone the repository
git clone https://github.com/your-username/SOLWEIG-GPU.git
cd SOLWEIG-GPU

# Create conda environment
conda env create -f environment.yml
conda activate solweig

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
```

## GPU Setup

### CUDA Requirements

- CUDA 11.0 or higher
- Compatible NVIDIA GPU (compute capability 3.5+)
- Sufficient GPU memory (4GB minimum, 8GB+ recommended)

### CPU-Only Mode

SOLWEIG-GPU automatically falls back to CPU if no GPU is detected, though performance will be significantly slower.

## Common Issues

### GDAL Import Error

If you get `ModuleNotFoundError: No module named '_gdal'`:

```bash
# Uninstall and reinstall GDAL via conda
conda uninstall gdal
conda install -c conda-forge gdal
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

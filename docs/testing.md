# Testing Guide

This page provides information about testing SOLWEIG-GPU. For installing the package and running a quick verification, see [Installation](installation.md#verify-with-test-suite).

## Running Tests

### Full Test Suite

```bash
pytest tests/
```

### With Coverage Report

```bash
pytest --cov=solweig_gpu --cov-report=html tests/
```

View the coverage report by opening `htmlcov/index.html` in a browser.

### Specific Test Modules

```bash
# Test UTCI calculations
pytest tests/test_calculate_utci.py

# Test preprocessing
pytest tests/test_preprocessor.py

# Test solar position
pytest tests/test_sun_position.py

# Test CLI
pytest tests/test_cli.py
```

### Example: Testing a Utility Function

```python
import pytest
from solweig_gpu.preprocessor import saturation_vapor_pressure

def test_saturation_vapor_pressure():
    """Test SVP calculation at known temperature."""
    # Water boils at 100°C, SVP = 101.325 kPa
    svp = saturation_vapor_pressure(100.0)
    assert abs(svp - 101.325) < 0.1
    
    # Freezing point
    svp = saturation_vapor_pressure(0.0)
    assert abs(svp - 0.6113) < 0.01
```

### Example: Testing with Fixtures

```python
import pytest
from pathlib import Path

@pytest.fixture
def temp_workspace(tmp_path):
    """Create temporary workspace structure (mirrors real layout: output_folder under base_path)."""
    workspace = tmp_path / "solweig_test"
    workspace.mkdir()
    (workspace / "Input_rasters").mkdir()
    (workspace / "output_folder").mkdir()
    return workspace

def test_file_creation(temp_workspace):
    """Test that files are created in correct location."""
    output_file = temp_workspace / "output_folder" / "test.tif"
    # ... test code ...
    assert output_file.exists()
```

## Continuous Integration

Tests run automatically on GitHub Actions for:
- **Python versions:** 3.10, 3.11, 3.12
- **Operating systems:** Linux, macOS
- **On every:** Push, pull request

See [.github/workflows/tests.yml](https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/main/.github/workflows/tests.yml)

## Manual Testing

For GPU-dependent features, manual testing is required:

### Test with Sample Data

Use the [official sample dataset on Zenodo](https://zenodo.org/records/18283037) (see also [Input Data](input_data.md) and [Installation](installation.md)):

```bash
# After downloading and extracting the sample data to e.g. ./sample_data

# Run simulation
python -m solweig_gpu.cli \
    --base_path ./sample_data \
    --date 2020-08-13 \
    --tile_size 500
```

### Verify Outputs

Check that outputs match expected values:
- UTCI range: typically 20-45°C for summer
- Tmrt range: typically 30-70°C in direct sun
- SVF range: 0-1 (0 = fully obstructed, 1 = open sky)

## Benchmark Tests

Compare with known results from published studies:

1. **Lindberg et al. (2008)** - Original SOLWEIG validation
2. **Lindberg & Grimmond (2011)** - Tmrt validation
3. **Your specific use cases** - Local validation data


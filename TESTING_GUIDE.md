# SOLWEIG-GPU Testing Guide

## Overview

SOLWEIG-GPU includes a comprehensive test suite with automated CI/CD testing. This guide explains how to run tests locally and how the CI system works.

## Test Suite Structure

The test suite is organized into the following test modules:

### Core Functionality Tests
- **test_preprocessor.py** - Raster validation, tiling, and WRF filename parsing
- **test_calculate_utci.py** - UTCI (Universal Thermal Climate Index) calculations
- **test_sun_position.py** - Solar position and day length calculations
- **test_walls_aspect.py** - Wall height and aspect calculations

### Integration Tests
- **test_integration.py** - End-to-end workflow with test data
- **test_integration_end_to_end.py** - Full pipeline tests
- **test_era5_processing.py** - ERA5 data processing with `valid_time` coordinate
- **test_tiling_and_metfiles.py** - Tiling and meteorological file generation

### Special Tests
- **test_cli.py** - Command-line interface tests
- **test_gpu_smoke.py** - GPU availability tests (auto-skipped without CUDA)
- **test_wrf_and_gui.py** - WRF parsing and GUI import tests

## Running Tests Locally

### Prerequisites

Install test dependencies:

```bash
conda create -n solweig python=3.10
conda activate solweig
conda install -c conda-forge gdal pytorch timezonefinder matplotlib sip
pip install PyQt5
pip install pytest coverage pytest-cov
pip install -e .
```

### Basic Test Commands

Run all tests:
```bash
pytest
```

Run with verbose output:
```bash
pytest -v
```

Run specific test file:
```bash
pytest tests/test_preprocessor.py
```

Run specific test:
```bash
pytest tests/test_preprocessor.py::TestRasterValidation::test_matching_dimensions
```

Run tests matching a pattern:
```bash
pytest -k "utci"
```

### Coverage Reports

Generate coverage report:
```bash
pytest --cov=solweig_gpu --cov-report=term-missing
```

Generate HTML coverage report:
```bash
pytest --cov=solweig_gpu --cov-report=html
# Open htmlcov/index.html in browser
```

Generate XML coverage (for CI):
```bash
pytest --cov=solweig_gpu --cov-report=xml
```

### Test Markers

Tests are marked with special markers for selective running:

- **@pytest.mark.slow** - Slow tests (skip with `-m "not slow"`)
- **@pytest.mark.integration** - Integration tests
- **@pytest.mark.gpu** - GPU-specific tests (auto-skip without CUDA)

Run only fast tests:
```bash
pytest -m "not slow"
```

Run only integration tests:
```bash
pytest -m integration
```

Skip GPU tests:
```bash
pytest -m "not gpu"
```

## Continuous Integration (CI)

### GitHub Actions Workflow

Tests run automatically on:
- Every push to `main`, `master`, or `Updates` branches
- Every pull request to these branches

### CI Test Matrix

- **Operating Systems**: Ubuntu (Linux), macOS
- **Python Versions**: 3.10, 3.11, 3.12
- **Note**: Windows excluded due to PyTorch DLL compatibility issues in CI

### CI Steps

1. Checkout code
2. Setup Miniconda environment
3. Install dependencies via conda-forge (GDAL, PyTorch, etc.)
4. Install package in editable mode
5. Run pytest with coverage
6. Upload coverage to Codecov

### Viewing CI Results

- Check the **Actions** tab on GitHub
- Look for the green checkmark ✅ or red X ❌ on commits/PRs
- Click on a workflow run to see detailed logs

## Coverage Requirements

- **Minimum coverage**: 5% (realistic for GPU-heavy code)
- **Coverage tracking**: Codecov integration
- **Configuration**: See `.coveragerc`

## Writing New Tests

### Test File Naming

- Place tests in `tests/` directory
- Name files `test_<module>.py`
- Name test functions `test_<description>`
- Name test classes `Test<Description>`

### Using Fixtures

Shared fixtures are available in `tests/conftest.py`:

```python
def test_example(temp_workspace):
    # temp_workspace is automatically cleaned up after test
    test_file = os.path.join(temp_workspace, 'test.tif')
    # ... your test code
```

### Example Test Structure

```python
import unittest
import numpy as np

class TestMyFeature(unittest.TestCase):
    def setUp(self):
        """Run before each test"""
        self.test_data = np.zeros((10, 10))
    
    def tearDown(self):
        """Run after each test"""
        pass
    
    def test_something(self):
        """Test description"""
        from solweig_gpu import my_function
        result = my_function(self.test_data)
        self.assertEqual(result.shape, (10, 10))
```

## Troubleshooting

### Common Issues

**Import errors**: Make sure package is installed with `pip install -e .`

**GDAL errors**: Install GDAL from conda-forge: `conda install -c conda-forge gdal`

**PyTorch DLL errors (Windows)**: Install PyTorch via conda: `conda install pytorch`

**GPU tests fail**: GPU tests auto-skip without CUDA - this is expected

### Debug Failed Tests

Run with full traceback:
```bash
pytest -v --tb=long
```

Run single failing test:
```bash
pytest tests/test_preprocessor.py::test_failing_test -v
```

Print output during tests:
```bash
pytest -s
```

## Test Data

Tests use synthetic data generated on-the-fly:
- **Rasters**: Created with GDAL in memory
- **ERA5 files**: Generated with xarray
- **Met files**: Simple text files
- **No external data dependencies**

## Contributing Tests

When adding new features:
1. Add corresponding tests
2. Run tests locally before pushing
3. Ensure CI passes on GitHub
4. Aim for reasonable coverage of new code

## Additional Resources

- pytest documentation: https://docs.pytest.org/
- Coverage.py: https://coverage.readthedocs.io/
- GitHub Actions: https://docs.github.com/en/actions


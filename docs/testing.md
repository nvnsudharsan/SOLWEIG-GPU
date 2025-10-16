# Testing Guide

This page provides information about testing SOLWEIG-GPU.

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

## Test Coverage

Current test coverage: **~13-15%**

This is **expected and appropriate** for a GPU-accelerated scientific simulation package.

### What's Tested

✅ **Unit Tests:**
- UTCI calculations (~80% coverage)
- Solar Position Algorithm (~80% coverage)
- CLI interface (~60% coverage)
- Wall aspect calculations (~40% coverage)
- Raster validation (~30% coverage)

✅ **Integration Tests:**
- Tile creation and overlap handling
- Met file generation
- End-to-end workflows (with synthetic data)

### What's NOT Tested (and Why)

❌ **GPU Physics Code** (~2000 lines):
- Requires CUDA/GPU runtime
- Needs real meteorological data
- Complex radiation calculations
- **Better validated through scientific publications**

❌ **GUI** (~500 lines):
- Qt interface requires display server
- Better tested manually

### Why This is OK

**Scientific computing packages** typically have 20-40% coverage because:

1. **GPU code** cannot run on standard CI without GPU runners
2. **Physics simulations** are validated through peer review, not just unit tests
3. **Large datasets** required for realistic tests (GB of memory)
4. **Execution time** can be hours for full simulations

**SOLWEIG-GPU tests what matters:**
- User-facing APIs ✅
- Data validation ✅
- Scientific algorithms (SPA, UTCI) ✅
- Error handling ✅

## Writing Tests

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
    """Create temporary workspace structure."""
    workspace = tmp_path / "solweig_test"
    workspace.mkdir()
    (workspace / "Input_rasters").mkdir()
    (workspace / "Outputs").mkdir()
    return workspace

def test_file_creation(temp_workspace):
    """Test that files are created in correct location."""
    output_file = temp_workspace / "Outputs" / "test.tif"
    # ... test code ...
    assert output_file.exists()
```

## Continuous Integration

Tests run automatically on GitHub Actions for:
- **Python versions:** 3.10, 3.11, 3.12
- **Operating systems:** Linux, macOS
- **On every:** Push, pull request

See [.github/workflows/tests.yml](https://github.com/your-repo/blob/main/.github/workflows/tests.yml)

## Manual Testing

For GPU-dependent features, manual testing is required:

### Test with Sample Data

```bash
# Download sample dataset
wget https://example.com/solweig_test_data.zip
unzip solweig_test_data.zip

# Run simulation
python -m solweig_gpu.cli \
    --base_path ./test_data \
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

## Contributing Tests

When contributing:

1. **Add tests for new features**
2. **Maintain or improve coverage** for testable code
3. **Don't force coverage** on GPU-dependent code
4. **Document test rationale** in docstrings

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

## Further Reading

- [Testing Philosophy](testing.md)
- [Coverage Report Details](testing.md#coverage-details)
- [pytest Documentation](https://docs.pytest.org/)

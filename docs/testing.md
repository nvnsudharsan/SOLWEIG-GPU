# Testing

This guide provides information about the test suite for SOLWEIG-GPU and how to run and contribute tests.

## Test Suite Overview

The SOLWEIG-GPU test suite is designed to ensure the reliability and correctness of the package. Tests are organized into several modules, each focusing on a specific component of the package.

## Test Structure

The test suite is located in the `tests/` directory and includes the following modules:

### Unit Tests

**`test_preprocessor.py`** — Tests for data preprocessing functionality including raster validation, tiling logic, and meteorological data extraction from different sources (ERA5, WRF, custom files).

**`test_sun_position.py`** — Tests for solar position calculations, day length computations, and solar geometry accuracy.

**`test_calculate_utci.py`** — Tests for UTCI calculation accuracy under various meteorological conditions.

**`test_walls_aspect.py`** — Tests for wall height and aspect (orientation) calculations from building DSM data.

### Integration Tests

**`test_integration.py`** — End-to-end tests that verify the complete workflow from input data to final outputs, including tile boundary handling and different meteorological data sources.

## Running Tests

### Prerequisites

Install the testing dependencies:

```bash
pip install pytest pytest-cov
```

### Running All Tests

To run the entire test suite:

```bash
pytest
```

### Running Specific Test Modules

To run tests from a specific module:

```bash
pytest tests/test_preprocessor.py
```

To run a specific test class:

```bash
pytest tests/test_preprocessor.py::TestRasterValidation
```

To run a specific test function:

```bash
pytest tests/test_preprocessor.py::TestRasterValidation::test_matching_dimensions
```

### Verbose Output

For detailed output showing each test:

```bash
pytest -v
```

### Test Markers

The test suite uses markers to categorize tests:

**`slow`** — Tests that take a long time to run (e.g., full integration tests)

```bash
# Run only fast tests
pytest -m "not slow"

# Run only slow tests
pytest -m "slow"
```

**`integration`** — Integration tests that test multiple components together

```bash
pytest -m "integration"
```

**`gpu`** — Tests that require a GPU to run

```bash
# Skip GPU tests on systems without GPU
pytest -m "not gpu"
```

## Test Coverage

To generate a test coverage report:

```bash
pytest --cov=solweig_gpu --cov-report=html
```

This will create an HTML coverage report in the `htmlcov/` directory. Open `htmlcov/index.html` in a browser to view detailed coverage information.

To see coverage in the terminal:

```bash
pytest --cov=solweig_gpu --cov-report=term
```

## Current Test Status

!!! warning "Work in Progress"
    The current test suite contains many placeholder tests that need to be fully implemented. The test structure is in place, but the actual test logic needs to be completed for many tests.

**Implemented Tests:**

- Raster validation (dimension and pixel size checking)
- WRF filename parsing
- Day length calculations

**Placeholder Tests:**

- Solar position accuracy tests
- UTCI calculation accuracy tests
- Wall height and aspect calculation tests
- Full integration tests

## Writing New Tests

When adding new functionality to SOLWEIG-GPU, please include corresponding tests. Follow these guidelines:

### Test Structure

```python
import unittest

class TestFeatureName(unittest.TestCase):
    """Test description."""

    def setUp(self):
        """Set up test fixtures before each test."""
        # Create temporary data, initialize objects, etc.
        pass

    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary files, close connections, etc.
        pass

    def test_specific_behavior(self):
        """Test a specific behavior or edge case."""
        # Arrange: Set up test data
        # Act: Call the function being tested
        # Assert: Verify the results
        pass
```

### Test Naming Conventions

- Test files: `test_<module_name>.py`
- Test classes: `Test<FeatureName>`
- Test methods: `test_<specific_behavior>`

### Using Assertions

```python
# Equality
self.assertEqual(actual, expected)
self.assertNotEqual(actual, unexpected)

# Boolean
self.assertTrue(condition)
self.assertFalse(condition)

# Comparisons
self.assertGreater(a, b)
self.assertLess(a, b)
self.assertGreaterEqual(a, b)
self.assertLessEqual(a, b)

# Approximate equality (for floats)
self.assertAlmostEqual(actual, expected, places=7)
self.assertAlmostEqual(actual, expected, delta=0.001)

# Exceptions
with self.assertRaises(ValueError):
    function_that_should_raise()

# Containment
self.assertIn(item, container)
self.assertNotIn(item, container)
```

### Testing with Temporary Files

```python
import tempfile
import os

class TestFileOperations(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temporary directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_file_creation(self):
        filepath = os.path.join(self.temp_dir, 'test.txt')
        # Test file operations
        pass
```

### Marking Tests

Use decorators to mark tests:

```python
import pytest

@pytest.mark.slow
def test_long_running_operation():
    """This test takes a long time."""
    pass

@pytest.mark.integration
def test_full_workflow():
    """This is an integration test."""
    pass

@pytest.mark.gpu
def test_gpu_acceleration():
    """This test requires a GPU."""
    pass
```

### Testing GPU Code

For GPU-dependent code, check if GPU is available:

```python
import torch
import unittest

class TestGPUFeature(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "GPU not available")
    def test_gpu_computation(self):
        """Test that only runs if GPU is available."""
        pass
```

## Continuous Integration

### GitHub Actions

A GitHub Actions workflow can be set up to automatically run tests on every commit and pull request. Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11']

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest --cov=solweig_gpu --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Best Practices

!!! tip "Test-Driven Development"
    Consider writing tests before implementing new features. This helps clarify requirements and ensures good test coverage from the start.

!!! tip "Test Independence"
    Each test should be independent and not rely on the state from other tests. Use `setUp()` and `tearDown()` to ensure clean state.

!!! tip "Meaningful Test Names"
    Test names should clearly describe what is being tested. A good test name explains the expected behavior.

!!! tip "Test Edge Cases"
    Don't just test the happy path. Test edge cases, boundary conditions, and error handling.

## Contributing Tests

When contributing tests to SOLWEIG-GPU:

1. Ensure all tests pass before submitting a pull request
2. Add tests for any new features or bug fixes
3. Maintain or improve test coverage
4. Follow the existing test structure and naming conventions
5. Document complex test setups or assertions

## Getting Help

If you need help with testing:

- Review existing tests for examples
- Check the pytest documentation: [https://docs.pytest.org/](https://docs.pytest.org/)
- Open an issue on GitHub for questions or suggestions

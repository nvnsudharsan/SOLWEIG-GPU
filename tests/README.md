># Testing Guide for SOLWEIG-GPU

This directory contains the test suite for the SOLWEIG-GPU package. The tests are organized into several modules, each focusing on a specific component of the package.

## Test Structure

The test suite is organized as follows:

-   `test_preprocessor.py`: Tests for data preprocessing, including raster validation, tiling, and meteorological data extraction.
-   `test_sun_position.py`: Tests for solar position calculations.
-   `test_calculate_utci.py`: Tests for UTCI calculation.
-   `test_walls_aspect.py`: Tests for wall height and aspect calculations.
-   `test_integration.py`: Integration tests that verify the entire workflow.

## Running the Tests

To run the entire test suite, use pytest:

```bash
pytest
```

To run a specific test module:

```bash
pytest tests/test_preprocessor.py
```

To run tests with verbose output:

```bash
pytest -v
```

To run only fast tests (excluding slow integration tests):

```bash
pytest -m "not slow"
```

## Test Coverage

To generate a test coverage report, install `pytest-cov` and run:

```bash
pip install pytest-cov
pytest --cov=solweig_gpu --cov-report=html
```

This will generate an HTML coverage report in the `htmlcov` directory.

## Writing New Tests

When adding new functionality to SOLWEIG-GPU, please include corresponding tests. Follow these guidelines:

1.  **Unit Tests**: Test individual functions in isolation. Use mocking when necessary to avoid dependencies on external resources.
2.  **Integration Tests**: Test the interaction between multiple components. Mark these tests with the `@pytest.mark.integration` decorator.
3.  **GPU Tests**: If a test requires a GPU, mark it with `@pytest.mark.gpu` so it can be skipped on systems without GPUs.

## Current Test Status

**Note**: The current test suite contains placeholder tests that need to be fully implemented. Many tests are marked as `pass` and require actual test logic to be added. This is a starting point for building a comprehensive test suite.

## Contributing

When contributing tests, please ensure that:

-   All tests pass before submitting a pull request.
-   New features include corresponding tests.
-   Tests are well-documented with clear descriptions of what they test.

># Documentation, Tests, and Examples for SOLWEIG-GPU

This document provides an overview of the documentation, test suite, and examples that have been created for the SOLWEIG-GPU package.

## Documentation

Comprehensive documentation has been created in the `docs/` directory. The documentation is organized into several sections to help users and developers understand and use the package effectively.

### Documentation Structure

The documentation includes the following files:

-   `index.md`: The main entry point for the documentation, providing an overview and links to other sections.
-   `installation.md`: Detailed installation instructions, including prerequisites and step-by-step setup.
-   `quickstart.md`: A quick-start guide to help users run their first simulation.
-   `user_guide.md`: Comprehensive user guide covering input data, configuration, and outputs.
-   `api_reference.md`: Detailed API reference for the `thermal_comfort()` function.
-   `developer_guide.md`: Information for developers who want to contribute to the project.
-   `examples.md`: Overview of usage examples.

### Key Features of the Documentation

The documentation provides clear explanations of how to prepare input data, configure simulations, and interpret outputs. It covers all three types of meteorological forcing (custom text files, ERA5, and WRF) and includes examples for each. The API reference documents all parameters of the main `thermal_comfort()` function, making it easy for users to understand what each parameter does.

## Test Suite

A comprehensive test suite has been created in the `tests/` directory. The test suite is organized into multiple modules, each focusing on a specific component of the package.

### Test Structure

The test suite includes the following test modules:

-   `test_preprocessor.py`: Tests for data preprocessing functionality, including raster validation, tiling, and meteorological data extraction.
-   `test_sun_position.py`: Tests for solar position calculations.
-   `test_calculate_utci.py`: Tests for UTCI calculation.
-   `test_walls_aspect.py`: Tests for wall height and aspect calculations.
-   `test_integration.py`: Integration tests that verify the entire workflow.

### Test Configuration

The test suite uses pytest as the testing framework. A `pytest.ini` configuration file has been created to set up test discovery and markers. The configuration includes markers for slow tests, integration tests, and GPU-dependent tests, allowing users to selectively run tests based on their environment.

### Running the Tests

To run the entire test suite:

```bash
pytest
```

To run a specific test module:

```bash
pytest tests/test_preprocessor.py
```

To run only fast tests:

```bash
pytest -m "not slow"
```

### Current Status

The test suite provides a solid foundation for testing the package. Many tests are currently implemented as placeholders and will need to be fully developed as the package evolves. The test structure is in place, making it easy to add new tests as features are added or bugs are discovered.

## Examples

A collection of example scripts has been created in the `examples/` directory. These examples demonstrate how to use SOLWEIG-GPU for different scenarios.

### Example Scripts

The examples include:

-   `run_with_wrf.py`: Demonstrates how to run a simulation using WRF data.
-   `run_with_era5.py`: Shows how to use ERA5 data for a simulation.
-   `run_with_custom_met_file.py`: Provides an example of using a custom meteorological file.

Each example script is fully commented and includes configuration sections where users can easily update paths to their own data.

### Example Documentation

A `README.md` file in the `examples/` directory provides instructions on how to run the examples, including prerequisites and step-by-step guidance.

## Next Steps

With the documentation, tests, and examples in place, the SOLWEIG-GPU package is now better equipped for users and developers. Here are some recommendations for next steps:

1.  **Complete Test Implementation**: Many tests are currently placeholders. Implementing the full test logic will ensure the package is robust and reliable.

2.  **Add More Examples**: Consider adding more advanced examples, such as batch processing multiple dates or visualizing outputs with Python libraries like matplotlib or folium.

3.  **Create Tutorial Notebooks**: Jupyter notebooks can provide an interactive way for users to learn about the package. Consider creating notebooks that walk through the entire workflow with visualizations.

4.  **Set Up Continuous Integration**: Implementing CI/CD with GitHub Actions or similar tools will ensure that tests are run automatically on every commit, helping to catch bugs early.

5.  **Generate API Documentation**: Consider using tools like Sphinx to automatically generate API documentation from docstrings in the code.

## Summary

This comprehensive documentation, test suite, and collection of examples will significantly improve the usability and maintainability of the SOLWEIG-GPU package. Users will find it easier to get started, and developers will have a solid foundation for contributing to the project.

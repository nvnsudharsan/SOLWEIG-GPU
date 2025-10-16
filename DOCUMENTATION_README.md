# SOLWEIG-GPU Documentation, Tests, and Examples

This package includes comprehensive documentation, a test suite, and usage examples for SOLWEIG-GPU.

## What's Included

### 📚 Documentation (GitHub Pages Ready)

A complete documentation website built with MkDocs and the Material theme, ready to be deployed to GitHub Pages.

**Documentation Structure:**

- **Home** (`docs/index.md`): Welcome page with overview and quick links
- **Getting Started**:
  - Installation guide with prerequisites and step-by-step instructions
  - Quickstart guide for running your first simulation
- **User Guide**:
  - Input Data: Detailed information on preparing geospatial rasters and meteorological data
  - Configuration: Complete parameter reference and optimization tips
  - Outputs: Guide to understanding and visualizing results
- **API Reference** (`docs/api_reference.md`): Detailed documentation of the `thermal_comfort()` function
- **Examples** (`docs/examples.md`): Usage examples for different meteorological data sources
- **Developer Guide** (`docs/developer_guide.md`): Architecture overview and contribution guidelines
- **Testing** (`docs/testing.md`): Guide to running and writing tests

**Key Features:**

- Modern, responsive design with Material for MkDocs theme
- Dark/light mode toggle
- Full-text search
- Code syntax highlighting
- Admonitions (notes, tips, warnings)
- Mobile-friendly navigation
- Automatic deployment via GitHub Actions

### 🧪 Test Suite

A comprehensive test framework using pytest, organized into multiple modules:

**Test Modules:**

- `test_preprocessor.py`: Tests for raster validation, tiling, and meteorological data extraction
- `test_sun_position.py`: Tests for solar position and day length calculations
- `test_calculate_utci.py`: Tests for UTCI calculation accuracy
- `test_walls_aspect.py`: Tests for wall height and aspect calculations
- `test_integration.py`: End-to-end integration tests

**Test Configuration:**

- `pytest.ini`: Test configuration with markers for slow, integration, and GPU tests
- `tests/README.md`: Testing guide with instructions

**Current Status:** Test structure is in place with some implemented tests and many placeholders ready for full implementation.

### 💡 Examples

Practical example scripts demonstrating different use cases:

- `run_with_wrf.py`: Using WRF meteorological data
- `run_with_era5.py`: Using ERA5 reanalysis data
- `run_with_custom_met_file.py`: Using custom meteorological files
- `examples/README.md`: Instructions for running examples

## Setting Up GitHub Pages

### Quick Start

1. **Push to GitHub:**

   ```bash
   git add .
   git commit -m "Add documentation, tests, and examples"
   git push origin main
   ```

2. **Enable GitHub Pages:**

   - Go to repository **Settings** → **Pages**
   - Source: Deploy from a branch
   - Branch: `gh-pages` / `/ (root)`
   - Click **Save**

3. **Configure Workflow Permissions:**

   - Settings → **Actions** → **General**
   - Workflow permissions: **Read and write permissions**
   - Check **Allow GitHub Actions to create and approve pull requests**
   - Click **Save**

4. **Wait for Deployment:**

   - Go to **Actions** tab
   - Wait for "Deploy Documentation" workflow to complete
   - Documentation will be available at: `https://<username>.github.io/<repository>/`

For detailed instructions, see `GITHUB_PAGES_SETUP.md`.

## Local Development

### Preview Documentation Locally

1. **Install dependencies:**

   ```bash
   pip install -r docs/requirements.txt
   ```

2. **Serve documentation:**

   ```bash
   mkdocs serve
   ```

3. **Open in browser:** `http://127.0.0.1:8000/`

### Run Tests

1. **Install test dependencies:**

   ```bash
   pip install pytest pytest-cov
   ```

2. **Run all tests:**

   ```bash
   pytest
   ```

3. **Run with coverage:**

   ```bash
   pytest --cov=solweig_gpu --cov-report=html
   ```

### Run Examples

1. **Edit example script** to update file paths
2. **Run the script:**

   ```bash
   python examples/run_with_era5.py
   ```

## File Structure

```
SOLWEIG-GPU-main/
├── docs/                           # Documentation source files
│   ├── index.md                    # Home page
│   ├── installation.md             # Installation guide
│   ├── quickstart.md               # Quickstart guide
│   ├── input_data.md               # Input data guide
│   ├── configuration.md            # Configuration reference
│   ├── outputs.md                  # Outputs guide
│   ├── api_reference.md            # API documentation
│   ├── examples.md                 # Examples overview
│   ├── developer_guide.md          # Developer guide
│   ├── testing.md                  # Testing guide
│   └── requirements.txt            # Documentation dependencies
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── test_preprocessor.py
│   ├── test_sun_position.py
│   ├── test_calculate_utci.py
│   ├── test_walls_aspect.py
│   ├── test_integration.py
│   └── README.md                   # Testing guide
├── examples/                       # Usage examples
│   ├── run_with_wrf.py
│   ├── run_with_era5.py
│   ├── run_with_custom_met_file.py
│   └── README.md                   # Examples guide
├── .github/
│   └── workflows/
│       └── docs.yml                # GitHub Actions workflow
├── mkdocs.yml                      # MkDocs configuration
├── pytest.ini                      # Pytest configuration
├── GITHUB_PAGES_SETUP.md           # Detailed setup guide
├── DOCUMENTATION_OVERVIEW.md       # Overview of deliverables
└── DOCUMENTATION_README.md         # This file
```

## Customization

### Update Documentation Theme

Edit `mkdocs.yml` to customize colors, features, and navigation:

```yaml
theme:
  name: material
  palette:
    primary: teal  # Change primary color
    accent: amber  # Change accent color
```

### Add New Documentation Pages

1. Create a new `.md` file in `docs/`
2. Add it to the navigation in `mkdocs.yml`:

```yaml
nav:
  - Your New Page: new_page.md
```

### Add More Tests

1. Create test functions in existing test modules or create new modules
2. Follow the naming convention: `test_<feature>.py`
3. Use pytest markers for categorization (`@pytest.mark.slow`, etc.)

## Deployment

### Automatic Deployment

The documentation is automatically deployed to GitHub Pages when you push to the `main` branch. The GitHub Actions workflow (`.github/workflows/docs.yml`) handles:

- Installing dependencies
- Building the documentation
- Deploying to the `gh-pages` branch

### Manual Deployment

To manually deploy:

```bash
mkdocs gh-deploy
```

This builds the documentation and pushes it to the `gh-pages` branch.

## Troubleshooting

### Documentation Not Building

- Check for syntax errors in Markdown files
- Verify all referenced files exist
- Check the GitHub Actions logs for errors

### Tests Failing

- Ensure all dependencies are installed
- Check that test data paths are correct
- Review test output for specific errors

### GitHub Pages Not Updating

- Verify the workflow completed successfully
- Check that `gh-pages` branch was created
- Clear browser cache
- Wait a few minutes for propagation

## Next Steps

1. **Complete Test Implementation**: Many tests are placeholders and need full implementation
2. **Add More Examples**: Consider adding Jupyter notebooks or advanced examples
3. **Enhance API Documentation**: Add docstrings to functions and use mkdocstrings to auto-generate API docs
4. **Add Tutorials**: Create step-by-step tutorials for common workflows
5. **Set Up CI/CD**: Add automated testing to the GitHub Actions workflow

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [Pytest Documentation](https://docs.pytest.org/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)

## Support

For questions or issues:

- Open an issue on the GitHub repository
- Check the documentation at the GitHub Pages site
- Review the setup guides included in this package

# SOLWEIG-GPU Documentation

This directory contains the Sphinx documentation for SOLWEIG-GPU, which can be built locally or hosted on Read the Docs.

## Building Documentation Locally

### Prerequisites

```bash
pip install -r requirements.txt
```

### Build HTML Documentation

```bash
cd docs
make html
```

The generated HTML will be in `_build/html/`. Open `_build/html/index.html` in a browser.

### Build PDF Documentation

```bash
make latexpdf
```

### Clean Build Files

```bash
make clean
```

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation index
├── api.rst              # Auto-generated API reference
├── installation.md      # Installation guide
├── quickstart.md        # Quick start tutorial
├── testing.md           # Testing guide
├── requirements.txt     # Sphinx dependencies
├── Makefile            # Build commands
└── guide/              # User guides (to be created)
    ├── input_data.md
    ├── meteorological_forcing.md
    └── ...
```

## Read the Docs Setup

### 1. Connect Repository

1. Go to https://readthedocs.org/
2. Sign in with GitHub
3. Import your SOLWEIG-GPU repository
4. The `.readthedocs.yml` file will be automatically detected

### 2. Configuration

The build is configured via `.readthedocs.yml` in the repository root:
- Python 3.11
- Sphinx builder
- PDF and ePub output
- Automatic builds on commits

### 3. Custom Domain (Optional)

Set up a custom domain in Read the Docs admin panel:
- Go to Admin → Domains
- Add your domain (e.g., docs.solweig-gpu.org)
- Configure DNS CNAME record

### 4. Badge

Add to your README.md:

```markdown
[![Documentation Status](https://readthedocs.org/projects/solweig-gpu/badge/?version=latest)](https://solweig-gpu.readthedocs.io/en/latest/?badge=latest)
```

## Documentation Guidelines

### Docstring Format

Use NumPy/Google style docstrings:

```python
def function_name(param1, param2):
    """
    Brief description of function.
    
    Longer description with details about what the function does,
    algorithms used, and any important notes.
    
    Args:
        param1 (type): Description of param1
        param2 (type): Description of param2
    
    Returns:
        return_type: Description of return value
    
    Example:
        >>> function_name(1, 2)
        3
    
    Notes:
        - Important note 1
        - Important note 2
    
    References:
        Author et al. (Year). Title. Journal.
    """
    pass
```

### Adding New Pages

1. Create Markdown or RST file in `docs/`
2. Add to `toctree` in `index.rst`
3. Rebuild documentation

### Adding Examples

Create code examples in docstrings or separate files:

```python
"""
Example:
    Basic usage::
    
        from solweig_gpu import thermal_comfort
        
        thermal_comfort(
            base_path='/path/to/data',
            selected_date_str='2020-08-13'
        )
"""
```

## Auto-Generated API Documentation

The API documentation is automatically generated from docstrings using Sphinx autodoc.

To update:
1. Write/update docstrings in source code
2. Rebuild documentation: `make html`
3. API reference will be automatically updated

## Contributing to Documentation

1. Fork the repository
2. Create a feature branch
3. Add/update documentation
4. Build locally to verify
5. Submit pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

## Troubleshooting

### Import Errors

If Sphinx cannot import modules:

```python
# In conf.py, add to autodoc_mock_imports:
autodoc_mock_imports = ['torch', 'gdal', 'osr', 'ogr', 'netCDF4', 'PyQt5']
```

### Build Warnings

Fix warnings by:
- Checking RST syntax
- Verifying all references exist
- Ensuring all modules can be imported

### Read the Docs Build Failures

Check:
- `.readthedocs.yml` syntax
- `docs/requirements.txt` has all dependencies
- Python version compatibility
- Build logs on Read the Docs dashboard

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Read the Docs Guide](https://docs.readthedocs.io/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [MyST Parser (Markdown)](https://myst-parser.readthedocs.io/)


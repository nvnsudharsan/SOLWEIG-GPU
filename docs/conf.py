# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'SOLWEIG-GPU'
copyright = '2022-2025, Harsh Kamath and Naveen Sudharsan'
author = 'Harsh Kamath and Naveen Sudharsan'

# Try to get version from package, fallback to static version
try:
    from solweig_gpu import __version__
    release = __version__
except ImportError:
    release = '1.2.15'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'myst_parser',  # For Markdown support
    'nbsphinx',  # For Jupyter notebooks
]

# Napoleon settings (for NumPy/Google style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_typehints = 'description'
autodoc_mock_imports = [
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'gdal', 
    'osgeo',
    'osgeo.gdal',
    'osgeo.osr',
    'osgeo.ogr',
    '_gdal',
    'osr', 
    'ogr', 
    'netCDF4', 
    'PyQt5',
    'PyQt5.QtWidgets',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'scipy',
    'scipy.ndimage',
    'scipy.spatial',
    'numpy',
    'pandas',
    'xarray',
    'shapely',
    'shapely.geometry',
    'timezonefinder',
    'pytz',
    'matplotlib',
    'matplotlib.path',
    'tqdm',  # Progress bar library
]

# Additional autodoc settings to handle import errors
autodoc_inherit_docstrings = True
autodoc_class_signature = "separated"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'  # Read the Docs theme
html_static_path = ['_static']

# Logo
html_logo = '_static/logo.jpg'

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'display_version': True,
    # Uncomment to show logo without text:
    # 'logo_only': True,
    # 'style_nav_header_background': '#2980B9',  # Custom header color
}

# The master toctree document.
master_doc = 'index'

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# MyST parser settings (for Markdown)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# nbsphinx configuration
nbsphinx_execute = 'never'  # Don't execute notebooks during build (faster, safer)
nbsphinx_allow_errors = True  # Continue build even if notebook has errors
nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}

.. note::
   This page was generated from a Jupyter notebook. 
   You can download it here: :download:`{{ docname }}`
"""


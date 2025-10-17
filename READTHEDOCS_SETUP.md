# 📚 Read the Docs Setup Guide for SOLWEIG-GPU

This guide walks you through setting up professional documentation on Read the Docs for your SOLWEIG-GPU package.

## ✅ What's Already Done

I've created the complete documentation structure for you:

```
SOLWEIG-GPU/
├── .readthedocs.yml          # RTD configuration
├── docs/
│   ├── conf.py               # Sphinx configuration
│   ├── index.rst             # Main documentation page
│   ├── api.rst               # Auto-generated API docs
│   ├── installation.md       # Installation guide
│   ├── quickstart.md         # Quick start tutorial
│   ├── testing.md            # Testing documentation
│   ├── requirements.txt      # Sphinx dependencies
│   ├── Makefile             # Build commands
│   └── README.md            # Documentation README
```

All your **docstrings** (81 functions, 77.1% coverage) are ready to be auto-generated into beautiful API documentation!

---

## 🚀 Step-by-Step Setup

### Step 1: Install Sphinx Locally (Optional but Recommended)

Test the documentation build on your machine first:

```bash
cd /Users/geo-ns36752/Documents/GitHub/SOLWEIG-GPU

# Install Sphinx and dependencies
pip install -r docs/requirements.txt

# Build the documentation
cd docs
make html

# View the result
open _build/html/index.html  # macOS
# or
xdg-open _build/html/index.html  # Linux
```

If there are any errors, fix them before proceeding to Read the Docs.

### Step 2: Commit and Push Documentation Files

```bash
cd /Users/geo-ns36752/Documents/GitHub/SOLWEIG-GPU

# Add all documentation files
git add .readthedocs.yml
git add docs/

# Commit
git commit -m "Add Sphinx documentation for Read the Docs

- Configure Sphinx with autodoc and Napoleon
- Create main index and API reference
- Add installation and quickstart guides
- Set up Read the Docs configuration
- All 81 functions with docstrings will auto-generate API docs"

# Push to GitHub
git push origin main  # or your branch name
```

### Step 3: Sign Up / Log In to Read the Docs

1. Go to https://readthedocs.org/
2. Click **"Sign Up"** (or "Log in" if you have an account)
3. **Sign in with GitHub** (recommended for automatic integration)
4. Authorize Read the Docs to access your repositories

### Step 4: Import Your Project

1. Click **"Import a Project"** from your dashboard
2. You'll see a list of your GitHub repositories
3. Find **"SOLWEIG-GPU"** and click the **"+"** button next to it
4. Or click **"Import Manually"** and enter:
   - Name: `SOLWEIG-GPU`
   - Repository URL: `https://github.com/YOUR_USERNAME/SOLWEIG-GPU`
   - Repository type: `Git`

5. Click **"Next"**

### Step 5: Configure Project Settings

Read the Docs will automatically detect `.readthedocs.yml` and use those settings.

**Default settings (from .readthedocs.yml):**
- ✅ Python 3.11
- ✅ Sphinx builder
- ✅ Auto-build on commits
- ✅ PDF and ePub output

You can customize in **Admin → Settings** if needed.

### Step 6: Build Documentation

1. Read the Docs will automatically trigger a build
2. Go to **"Builds"** tab to watch progress
3. First build takes ~2-5 minutes

**If build succeeds:** 🎉 Your docs are live!

**If build fails:** Check the build log for errors and fix them.

### Step 7: View Your Documentation

Your documentation will be available at:

```
https://solweig-gpu.readthedocs.io/en/latest/
```

Or with your username:
```
https://YOUR-USERNAME-solweig-gpu.readthedocs.io/
```

---

## 🎨 Customization Options

### Add More Content

Create additional documentation pages:

```bash
cd docs

# Create new guide
cat > guide/input_data.md << 'EOF'
# Input Data Guide

Details about preparing input rasters...
EOF

# Add to index.rst toctree
# Then rebuild
make html
```

### Change Theme

Edit `docs/conf.py`:

```python
# Current: Read the Docs theme
html_theme = 'sphinx_rtd_theme'

# Alternative themes:
# html_theme = 'alabaster'
# html_theme = 'sphinx_book_theme'
# html_theme = 'pydata_sphinx_theme'
```

### Add Logo

1. Add logo image to `docs/_static/logo.png`
2. Edit `docs/conf.py`:

```python
html_logo = '_static/logo.png'
html_theme_options = {
    'logo_only': True,
    'display_version': True,
}
```

### Custom Domain

Set up a custom domain (e.g., docs.solweig-gpu.org):

1. Go to **Admin → Domains** in Read the Docs
2. Click **"Add Domain"**
3. Enter your domain
4. Configure DNS CNAME record:
   ```
   docs.solweig-gpu.org  CNAME  solweig-gpu.readthedocs.io
   ```

---

## 📌 Adding Documentation Badge to README

Add this to your main `README.md`:

```markdown
[![Documentation Status](https://readthedocs.org/projects/solweig-gpu/badge/?version=latest)](https://solweig-gpu.readthedocs.io/en/latest/?badge=latest)
```

Example placement:

```markdown
# SOLWEIG-GPU

[![PyPI version](https://badge.fury.io/py/solweig-gpu.svg)](https://badge.fury.io/py/solweig-gpu)
[![Documentation Status](https://readthedocs.org/projects/solweig-gpu/badge/?version=latest)](https://solweig-gpu.readthedocs.io/en/latest/?badge=latest)
[![CI Status](https://github.com/YOUR_USERNAME/SOLWEIG-GPU/workflows/Tests/badge.svg)](https://github.com/YOUR_USERNAME/SOLWEIG-GPU/actions)
[![codecov](https://codecov.io/gh/YOUR_USERNAME/SOLWEIG-GPU/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/SOLWEIG-GPU)

GPU-accelerated SOLWEIG model for urban thermal comfort simulation.

📚 **[Documentation](https://solweig-gpu.readthedocs.io/)** | ...
```

---

## 🔧 Troubleshooting

### Build Fails with Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'torch'`

**Solution:** These are already mocked in `conf.py`:
```python
autodoc_mock_imports = ['torch', 'gdal', 'osr', 'ogr', 'netCDF4', 'PyQt5']
```

If you need to add more:
```python
autodoc_mock_imports = ['torch', 'gdal', 'new_module_name']
```

### Build Fails with Syntax Errors

**Symptom:** `WARNING: document isn't included in any toctree`

**Solution:** Make sure all `.md` and `.rst` files are listed in `index.rst`:

```rst
.. toctree::
   :maxdepth: 2
   
   installation
   quickstart
   api
   testing
```

### API Documentation Not Showing

**Symptom:** API pages are empty or missing

**Solution:** Check that:
1. Module paths are correct in `api.rst`
2. Docstrings are present in your code
3. Modules can be imported (check autodoc_mock_imports)

### Documentation Not Updating

**Symptom:** Changes don't appear after commit

**Solution:**
1. Go to your project on Read the Docs
2. Click **"Builds"** tab
3. Check if auto-build is triggered
4. If not, click **"Build Version"** manually
5. Enable webhooks: **Admin → Integrations**

---

## 📖 Documentation Content Recommendations

### Essential Pages (Already Created)
- ✅ Installation guide
- ✅ Quick start tutorial
- ✅ API reference (auto-generated)
- ✅ Testing guide

### Recommended Additions

1. **User Guide**
   - Input data preparation
   - Meteorological forcing options
   - Configuration parameters
   - Output interpretation
   - Troubleshooting

2. **Examples Gallery**
   - Urban park analysis
   - Street canyon study
   - Campus-scale simulation
   - Multi-day runs

3. **Scientific Background**
   - SOLWEIG model overview
   - Solar position algorithm (SPA)
   - Radiation balance equations
   - UTCI calculation method
   - References

4. **Developer Guide**
   - Architecture overview
   - Contributing guidelines
   - Code style
   - GPU optimization tips

---

## ✨ Next Steps

1. **Build locally** to test: `cd docs && make html`
2. **Commit and push** documentation files
3. **Import on Read the Docs** (takes 5 minutes)
4. **Add badge** to README.md
5. **Share** your beautiful docs with the world!

Your documentation will look like this:

```
📚 SOLWEIG-GPU Documentation
   ├── 🏠 Home (index.rst)
   ├── 💾 Installation
   ├── ⚡ Quick Start
   ├── 📖 API Reference
   │   ├── Main Entry Point (thermal_comfort)
   │   ├── Data Preprocessing (16 functions)
   │   ├── Radiation Calculations (24 functions)
   │   ├── Solar Position (17 functions)
   │   ├── Shadow & SVF (5 functions)
   │   ├── UTCI (8 functions)
   │   ├── Wall Geometry (6 functions)
   │   ├── CLI (2 functions)
   │   └── Utilities (2 functions)
   ├── 🧪 Testing
   └── 🔍 Search
```

**All 81 documented functions will have beautiful, searchable API documentation automatically generated from your docstrings!** 🎉

---

## 📚 Resources

- [Read the Docs Tutorial](https://docs.readthedocs.io/en/stable/tutorial/)
- [Sphinx Documentation](https://www.sphinx-doc.org/en/master/)
- [RST Syntax Guide](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Example Scientific Project](https://scikit-learn.org/stable/) - For inspiration

---

**Questions?** Check `docs/README.md` for more details or open an issue on GitHub.


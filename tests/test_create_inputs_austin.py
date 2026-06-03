"""
Test create_inputs workflow for Austin, Texas.

Requires optional dependencies: earthengine-api, geemap, geopandas, rasterio, osmnx.
Earth Engine must be authenticated (e.g. ee.Authenticate() or service account).

Run as script (no pytest) from repo root:
  python tests/test_create_inputs_austin.py
  Outputs are written to the project folder: <repo>/austin_create_inputs/Austin_for_solweig/
  (so you can inspect Building_DSM.tif, DEM.tif, Trees.tif, Landuse.tif, era5_*.nc, etc.)

Run with pytest (skips if deps missing):
  pytest tests/test_create_inputs_austin.py -v
"""

import os
import sys
import tempfile
import shutil

# Prefer local repo when run from project root (so create_inputs is found before installed pkg)
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Austin, Texas
AUSTIN_LAT = 30.2672
AUSTIN_LON = -97.7431


def _check_deps():
    """Return True if create_inputs optional deps are available."""
    try:
        import importlib.util
        if importlib.util.find_spec("ee") is None:
            return False
        if importlib.util.find_spec("geemap") is None:
            return False
        if importlib.util.find_spec("rasterio") is None:
            return False
        return True
    except Exception:
        return False


def run_austin_test(base_folder=None):
    """Run create_inputs for Austin, TX. Returns output path or raises."""
    try:
        from solweig_gpu import create_inputs
    except ImportError as e:
        if "create_inputs" in str(e):
            raise ImportError(
                "create_inputs not in this install. Run this test from the repo root so the "
                "local package is used, or reinstall in editable mode: pip install -e ."
            ) from e
        raise

    if base_folder is None:
        base_folder = tempfile.mkdtemp(prefix="solweig_austin_")
    out_path = create_inputs(
        lat=AUSTIN_LAT,
        lon=AUSTIN_LON,
        city="Austin",
        km_buffer=2.0,
        km_reduced_lat=0.5,
        km_reduced_lon=0.5,
        year_start=2024,
        year_end=2024,
        base_folder=base_folder,
        resolution=10.0,
    )
    return out_path


def main():
    if not _check_deps():
        print("SKIP: Optional deps missing (earthengine-api, geemap, rasterio).")
        print("Python in use:", sys.executable)
        print("Install for this Python with:")
        print("  python -m pip install earthengine-api geemap rasterio")
        sys.exit(0)
    print("Running create_inputs for Austin, TX ...")
    # Write to project folder so you can analyse the outputs (not a temp dir)
    base = os.path.join(_repo_root, "austin_create_inputs")
    os.makedirs(base, exist_ok=True)
    print(f"Output base: {os.path.abspath(base)}")
    try:
        out_path = run_austin_test(base_folder=base)
        out_path = os.path.abspath(out_path)
        print(f"SOLWEIG inputs dir: {out_path}")
        assert os.path.isdir(out_path)
        expected = ["Building_DSM.tif", "DEM.tif", "Trees.tif", "Landuse.tif"]
        for name in expected:
            fp = os.path.join(out_path, name)
            if os.path.isfile(fp):
                print(f"  OK {name}")
            else:
                print(f"  MISSING {name}")
        ncs = [f for f in os.listdir(out_path) if f.endswith(".nc")]
        print(f"  NetCDF: {len(ncs)} file(s)")
        print("PASS: Austin create_inputs test completed.")
        print(f"You can analyse the rasters and met file in: {out_path}")
    except Exception as e:
        err_msg = str(e)
        if "PROJ" in err_msg and ("DATABASE.LAYOUT" in err_msg or "proj.db" in err_msg):
            print("FAIL: PROJ database version conflict in your environment.")
            print("Fix: update PROJ in the same env (e.g. conda):")
            print("  conda install -c conda-forge proj")
            print("  or: conda update proj")
            print("Then rerun this test.")
            sys.exit(3)
        if "Earth Engine" in err_msg and ("Authenticate" in err_msg or "not initialized" in err_msg or "project" in err_msg):
            print("FAIL: Earth Engine:", err_msg[:200])
            if "project" in err_msg.lower() and "no project" in err_msg.lower():
                print("Set your Google Cloud project ID, then rerun:")
                print("  export EE_PROJECT=your-gcp-project-id")
                print("(Create a project at https://console.cloud.google.com and enable Earth Engine API.)")
            else:
                creds = os.path.expanduser("~/.config/earthengine/credentials")
                if os.path.isfile(creds):
                    print(f"Credentials exist at {creds}. If you see 'no project found', run:")
                    print("  export EE_PROJECT=your-gcp-project-id")
                else:
                    print('Run: python -c "import ee; ee.Authenticate()" and complete browser sign-in.')
            print("Then rerun this test.")
            sys.exit(2)
        print(f"FAIL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


# --- pytest entry (skips if deps missing) ---
import pytest

@pytest.fixture
def temp_base():
    d = tempfile.mkdtemp(prefix="solweig_austin_test_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_create_inputs_austin_smoke(temp_base):
    """Run create_inputs for Austin, TX (small area, one year). Requires EE auth."""
    pytest.importorskip("ee", reason="earthengine-api not installed")
    pytest.importorskip("geemap", reason="geemap not installed")
    pytest.importorskip("rasterio", reason="rasterio not installed")

    out_path = run_austin_test(base_folder=temp_base)

    assert os.path.isdir(out_path), f"Output dir not created: {out_path}"
    assert "Austin" in out_path or "austin" in out_path.lower()

    expected_rasters = [
        "Building_DSM.tif",
        "DEM.tif",
        "Trees.tif",
        "Landuse.tif",
    ]
    for name in expected_rasters:
        fp = os.path.join(out_path, name)
        assert os.path.isfile(fp), f"Missing expected raster: {name}"

    nc_files = [f for f in os.listdir(out_path) if f.endswith(".nc")]
    assert len(nc_files) >= 1, f"Expected at least one .nc file in {out_path}"

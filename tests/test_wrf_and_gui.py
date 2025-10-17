import os
import sys
import pytest


def test_wrf_filename_parsing_variants():
    from solweig_gpu.preprocessor import extract_datetime_strict
    # Underscore format HH_MM_SS
    dt, dom = extract_datetime_strict('wrfout_d01_2020-08-13_06_00_00')
    assert (dt.year, dt.month, dt.day, dt.hour) == (2020, 8, 13, 6)
    assert dom == 1

    # Colon format HH:MM:SS
    dt, dom = extract_datetime_strict('wrfout_d02_2020-08-13_12:30:45')
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second) == (2020, 8, 13, 12, 30, 45)
    assert dom == 2

    # Hour-only format HH
    dt, dom = extract_datetime_strict('wrfout_d03_2020-08-13_23')
    assert (dt.year, dt.month, dt.day, dt.hour) == (2020, 8, 13, 23)
    assert dom == 3

    # Invalid should raise
    with pytest.raises(ValueError):
        extract_datetime_strict('wrfout_invalid_name')


@pytest.mark.skipif(
    os.environ.get('DISPLAY') is None and sys.platform != 'win32',
    reason='GUI smoke test skipped on headless runners without DISPLAY'
)
def test_gui_entrypoint_smoke():
    # Importing and constructing the app would require Qt loop; just ensure module imports.
    # This verifies the entrypoint can be located.
    import importlib
    mod = importlib.import_module('solweig_gpu.solweig_gpu_gui')
    assert hasattr(mod, '__file__')



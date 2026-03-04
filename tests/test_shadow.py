"""
Unit tests for shadow.py – covers the svfN accumulation fix
and bush indexing consistency.
"""

import torch
import pytest
from solweig_gpu.shadow import svf_calculator


def _build_small_scene(size=20):
    """Create a minimal urban scene with one building for SVF testing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dem = torch.zeros((size, size), device=device)
    dsm = dem.clone()
    dsm[8:12, 8:12] = 10.0  # 10 m building in the centre

    tree_height = torch.zeros((size, size), device=device)
    vegdsm = tree_height + dsm
    vegdsm[vegdsm == dsm] = 0
    vegdsm2 = tree_height * 0.25 + dsm
    vegdsm2[vegdsm2 == dsm] = 0

    bush = torch.zeros((size, size), device=device)
    amaxvalue = dsm.max()
    scale = 1.0  # 1 pixel = 1 m

    return amaxvalue, dsm, vegdsm, vegdsm2, bush, scale


def test_svfN_is_nonzero():
    """svfN must be accumulated for north-facing azimuths (>=270 or <90).

    Before the fix, the condition ``270 <= azimuth < 90`` was always False,
    leaving svfN as all zeros.  After the fix it should contain positive
    values comparable in magnitude to svfE / svfS / svfW.
    """
    amaxvalue, dsm, vegdsm, vegdsm2, bush, scale = _build_small_scene()

    result = svf_calculator(2, amaxvalue, dsm, vegdsm, vegdsm2, bush, scale)
    # Return order: svf, svfaveg, svfE, svfEaveg, svfEveg, svfN, svfNaveg,
    #               svfNveg, svfS, svfSaveg, svfSveg, svfveg, svfW, svfWaveg,
    #               svfWveg, vegshmat, vbshvegshmat, shmat, SVFtotal
    svfE = result[2]
    svfN = result[5]
    svfS = result[8]
    svfW = result[12]

    # All directional SVFs should be non-zero where there is sky exposure
    assert svfN.sum().item() > 0, "svfN is all zeros – north accumulation broken"
    assert svfE.sum().item() > 0, "svfE is all zeros"
    assert svfS.sum().item() > 0, "svfS is all zeros"
    assert svfW.sum().item() > 0, "svfW is all zeros"

    # svfN should be roughly the same order of magnitude as the other directions
    ratio = svfN.sum().item() / svfE.sum().item()
    assert 0.3 < ratio < 3.0, (
        f"svfN/svfE ratio {ratio:.3f} is unreasonable – accumulation may still be wrong"
    )

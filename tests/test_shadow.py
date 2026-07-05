import numpy as np
import torch

from solweig_gpu.shadow import SVF_CACHE_VERSION, is_current_svf_cache, shadow


def _isolated_tree_scene(size=31):
    dsm = torch.zeros((size, size), dtype=torch.float32)
    canopy = torch.zeros_like(dsm)
    trunk = torch.zeros_like(dsm)
    bush = torch.zeros_like(dsm)

    center = size // 2
    canopy[center, center] = 10.0
    trunk[center, center] = 2.5
    return dsm, canopy, trunk, bush


def test_shadow_tracks_previous_ray_step_for_thin_canopies():
    dsm, canopy, trunk, bush = _isolated_tree_scene()

    _, vegetation_visible, _ = shadow(
        torch.tensor(10.0),
        dsm,
        canopy,
        trunk,
        bush,
        torch.tensor(5.0),
        torch.tensor(42.0),
        1.0,
    )

    shaded = set(map(tuple, torch.nonzero(vegetation_visible == 0).tolist()))
    assert shaded == {
        (18, 15),
        (19, 15),
        (20, 15),
        (21, 14),
        (22, 14),
        (23, 14),
        (24, 14),
        (25, 14),
        (26, 14),
        (27, 14),
    }


def test_zenith_patch_does_not_cast_horizontal_building_shadows():
    dsm = torch.zeros((9, 9), dtype=torch.float32)
    dsm[4, 4] = 10.0
    vegetation = torch.zeros_like(dsm)

    building_visible, _, _ = shadow(
        torch.tensor(10.0),
        dsm,
        vegetation,
        vegetation,
        vegetation,
        torch.tensor(0.0),
        torch.tensor(90.0),
        1.0,
    )

    assert torch.all(building_visible == 1)


def test_pre_pergola_shadow_cache_is_invalidated(tmp_path):
    old_cache = tmp_path / "old.npz"
    current_cache = tmp_path / "current.npz"
    np.savez_compressed(old_cache, shadowmat=np.ones((1, 1, 1)))
    np.savez_compressed(current_cache, cache_version=np.asarray(SVF_CACHE_VERSION))

    assert not is_current_svf_cache(old_cache)
    assert is_current_svf_cache(current_cache)

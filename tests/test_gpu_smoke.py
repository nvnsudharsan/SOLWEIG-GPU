import pytest


@pytest.mark.gpu
@pytest.mark.skipif(__import__('torch').cuda.is_available() is False, reason='CUDA not available')
def test_cuda_tensor_ops():
    torch = __import__('torch')
    device = torch.device('cuda')
    a = torch.tensor([1.0, 2.0, 3.0], device=device)
    b = torch.tensor([4.0, 5.0, 6.0], device=device)
    c = a + b
    assert torch.allclose(c, torch.tensor([5.0, 7.0, 9.0], device=device))


@pytest.mark.gpu
def test_package_import_without_cuda_side_effects():
    # Ensure importing the package modules does not require CUDA implicitly
    # and does not error when CUDA is unavailable.
    try:
        __import__('solweig_gpu')
        __import__('solweig_gpu.shadow')
    except Exception as e:
        pytest.fail(f'Package import failed: {e}')



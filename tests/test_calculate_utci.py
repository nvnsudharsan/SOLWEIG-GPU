"""
Test suite for the calculate_utci module.

This module tests the UTCI (Universal Thermal Climate Index) calculation.
"""

import unittest
import numpy as np
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestUTCICalculation(unittest.TestCase):
    """Test UTCI calculation functionality."""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason='PyTorch required for UTCI')
    def test_utci_comfortable_conditions(self):
        """Test UTCI calculation for comfortable conditions."""
        from solweig_gpu.calculate_utci import utci_calculator
        import torch
        
        # Comfortable conditions: Ta = 25°C, Tmrt = 25°C, wind = 1 m/s, RH = 50%
        # Expected UTCI should be close to 25°C
        # Signature: utci_calculator(Ta, RH, Tmrt, va10m)
        ta = torch.tensor([25.0], dtype=torch.float32)
        rh = torch.tensor([50.0], dtype=torch.float32)
        tmrt = torch.tensor([25.0], dtype=torch.float32)
        wind = torch.tensor([1.0], dtype=torch.float32)
        
        utci = utci_calculator(ta, rh, tmrt, wind)
        
        # UTCI should be close to air temp in comfortable conditions
        utci_val = utci.item() if hasattr(utci, 'item') else float(utci[0])
        self.assertTrue(23.0 < utci_val < 27.0, f"UTCI {utci_val} out of expected range")

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason='PyTorch required for UTCI')
    def test_utci_hot_conditions(self):
        """Test UTCI calculation for hot conditions."""
        from solweig_gpu.calculate_utci import utci_calculator
        import torch
        
        # Hot conditions: high temperature and high radiation
        # Signature: utci_calculator(Ta, RH, Tmrt, va10m)
        ta = torch.tensor([35.0], dtype=torch.float32)
        rh = torch.tensor([40.0], dtype=torch.float32)
        tmrt = torch.tensor([50.0], dtype=torch.float32)
        wind = torch.tensor([0.5], dtype=torch.float32)
        
        utci = utci_calculator(ta, rh, tmrt, wind)
        
        # UTCI should be higher than air temperature due to radiation
        utci_val = utci.item() if hasattr(utci, 'item') else float(utci[0])
        self.assertTrue(utci_val > 35.0, f"UTCI {utci_val} should be > air temp 35.0")

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason='PyTorch required for UTCI')
    def test_utci_cold_conditions(self):
        """Test UTCI calculation for cold conditions."""
        from solweig_gpu.calculate_utci import utci_calculator
        import torch
        
        # Cold conditions: low temperature and high wind (wind chill)
        ta = torch.tensor([5.0], dtype=torch.float32)
        rh = torch.tensor([70.0], dtype=torch.float32)
        tmrt = torch.tensor([5.0], dtype=torch.float32)
        wind = torch.tensor([5.0], dtype=torch.float32)
        
        utci = utci_calculator(ta, rh, tmrt, wind)
        
        # UTCI should be lower than air temperature due to wind chill
        utci_val = utci.item() if hasattr(utci, 'item') else float(utci[0])
        self.assertTrue(utci_val < 5.0, f"UTCI {utci_val} should be < air temp 5.0 due to wind")

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason='PyTorch required for UTCI')
    def test_utci_extreme_heat(self):
        """Test UTCI calculation for extreme heat conditions."""
        from solweig_gpu.calculate_utci import utci_calculator
        import torch
        
        # Extreme heat
        ta = torch.tensor([40.0], dtype=torch.float32)
        rh = torch.tensor([30.0], dtype=torch.float32)
        tmrt = torch.tensor([60.0], dtype=torch.float32)
        wind = torch.tensor([0.3], dtype=torch.float32)
        
        utci = utci_calculator(ta, rh, tmrt, wind)
        
        # Should return a valid number (not NaN or inf)
        utci_val = utci.item() if hasattr(utci, 'item') else float(utci[0])
        self.assertFalse(np.isnan(utci_val))
        self.assertFalse(np.isinf(utci_val))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason='PyTorch required for UTCI')
    def test_utci_extreme_cold(self):
        """Test UTCI calculation for extreme cold conditions."""
        from solweig_gpu.calculate_utci import utci_calculator
        import torch
        
        # Extreme cold
        ta = torch.tensor([-10.0], dtype=torch.float32)
        rh = torch.tensor([80.0], dtype=torch.float32)
        tmrt = torch.tensor([-10.0], dtype=torch.float32)
        wind = torch.tensor([10.0], dtype=torch.float32)
        
        utci = utci_calculator(ta, rh, tmrt, wind)
        
        # Should return a valid number (not NaN or inf)
        utci_val = utci.item() if hasattr(utci, 'item') else float(utci[0])
        self.assertFalse(np.isnan(utci_val))
        self.assertFalse(np.isinf(utci_val))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason='PyTorch required for UTCI')
    def test_utci_input_validation(self):
        """Test that invalid inputs are handled correctly."""
        from solweig_gpu.calculate_utci import utci_calculator
        import torch
        
        # Test with valid but edge case inputs
        # Signature: utci_calculator(Ta, RH, Tmrt, va10m)
        
        # Very low wind (should still work)
        utci = utci_calculator(
            torch.tensor([20.0], dtype=torch.float32),
            torch.tensor([50.0], dtype=torch.float32),
            torch.tensor([20.0], dtype=torch.float32),
            torch.tensor([0.1], dtype=torch.float32)
        )
        self.assertTrue(torch.all(torch.isfinite(utci)))
        
        # Very high RH (100%)
        utci = utci_calculator(
            torch.tensor([25.0], dtype=torch.float32),
            torch.tensor([100.0], dtype=torch.float32),
            torch.tensor([25.0], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32)
        )
        self.assertTrue(torch.all(torch.isfinite(utci)))
        
        # Low RH
        utci = utci_calculator(
            torch.tensor([25.0], dtype=torch.float32),
            torch.tensor([10.0], dtype=torch.float32),
            torch.tensor([25.0], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32)
        )
        self.assertTrue(torch.all(torch.isfinite(utci)))


class TestUTCIArrayProcessing(unittest.TestCase):
    """Test UTCI calculation with array inputs."""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason='PyTorch required for UTCI')
    def test_utci_2d_array(self):
        """Test UTCI calculation with 2D arrays."""
        from solweig_gpu.calculate_utci import utci_calculator
        import torch
        
        # Create 2D arrays (e.g., 5x5 spatial grid)
        # Signature: utci_calculator(Ta, RH, Tmrt, va10m)
        shape = (5, 5)
        ta = torch.full(shape, 25.0, dtype=torch.float32)
        rh = torch.full(shape, 50.0, dtype=torch.float32)
        tmrt = torch.full(shape, 30.0, dtype=torch.float32)
        wind = torch.full(shape, 1.5, dtype=torch.float32)
        
        utci = utci_calculator(ta, rh, tmrt, wind)
        
        # Should return array of same shape
        self.assertEqual(utci.shape, shape)
        # All values should be valid (or -999 for invalid)
        valid_utci = utci[utci > -999]
        self.assertTrue(torch.all(torch.isfinite(valid_utci)))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason='PyTorch required for UTCI')
    def test_utci_3d_array(self):
        """Test UTCI calculation with 3D arrays (spatial + temporal)."""
        from solweig_gpu.calculate_utci import utci_calculator
        import torch
        
        # Create 3D arrays (e.g., 3 time steps x 4x4 spatial grid)
        shape = (3, 4, 4)
        ta = torch.from_numpy(np.random.uniform(20, 30, shape).astype(np.float32))
        rh = torch.from_numpy(np.random.uniform(30, 70, shape).astype(np.float32))
        tmrt = torch.from_numpy(np.random.uniform(25, 35, shape).astype(np.float32))
        wind = torch.from_numpy(np.random.uniform(0.5, 3.0, shape).astype(np.float32))
        
        utci = utci_calculator(ta, rh, tmrt, wind)
        
        # Should return array of same shape
        self.assertEqual(utci.shape, shape)
        # All valid values should be finite
        valid_utci = utci[utci > -999]
        self.assertTrue(torch.all(torch.isfinite(valid_utci)))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason='PyTorch required for UTCI')
    def test_utci_nan_handling(self):
        """Test that invalid values are handled correctly."""
        from solweig_gpu.calculate_utci import utci_calculator
        import torch
        
        # Create arrays with invalid (-999) values
        ta = torch.tensor([25.0, -999.0, 30.0], dtype=torch.float32)
        rh = torch.tensor([50.0, 50.0, -999.0], dtype=torch.float32)
        tmrt = torch.tensor([25.0, 30.0, 30.0], dtype=torch.float32)
        wind = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        
        utci = utci_calculator(ta, rh, tmrt, wind)
        
        # Function should return -999 for invalid inputs
        self.assertEqual(utci.shape[0], 3)
        # First value should be valid
        self.assertTrue(utci[0] > -999)
        # Second value should be -999 (invalid Ta)
        self.assertEqual(utci[1].item(), -999.0)


if __name__ == '__main__':
    unittest.main()

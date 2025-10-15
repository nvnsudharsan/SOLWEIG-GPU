"""
Test suite for the calculate_utci module.

This module tests the UTCI (Universal Thermal Climate Index) calculation.
"""

import unittest
import numpy as np


class TestUTCICalculation(unittest.TestCase):
    """Test UTCI calculation functionality."""

    def test_utci_comfortable_conditions(self):
        """Test UTCI calculation for comfortable conditions."""
        from solweig_gpu.calculate_utci import utci_calculator
        
        # Comfortable conditions: Ta = 25°C, Tmrt = 25°C, wind = 1 m/s, RH = 50%
        # Expected UTCI should be close to 25°C
        ta = 25.0  # Air temperature
        tmrt = 25.0  # Mean radiant temperature
        wind = 1.0  # Wind speed
        rh = 50.0  # Relative humidity
        
        utci = utci_calculator(ta, tmrt, wind, rh)
        
        # UTCI should be close to air temp in comfortable conditions
        self.assertIsInstance(utci, (int, float, np.ndarray))
        self.assertTrue(23.0 < utci < 27.0, f"UTCI {utci} out of expected range")

    def test_utci_hot_conditions(self):
        """Test UTCI calculation for hot conditions."""
        from solweig_gpu.calculate_utci import utci_calculator
        
        # Hot conditions: high temperature and high radiation
        ta = 35.0
        tmrt = 50.0  # High radiant temperature
        wind = 0.5
        rh = 40.0
        
        utci = utci_calculator(ta, tmrt, wind, rh)
        
        # UTCI should be higher than air temperature due to radiation
        self.assertIsInstance(utci, (int, float, np.ndarray))
        self.assertTrue(utci > ta, f"UTCI {utci} should be > air temp {ta}")

    def test_utci_cold_conditions(self):
        """Test UTCI calculation for cold conditions."""
        from solweig_gpu.calculate_utci import utci_calculator
        
        # Cold conditions: low temperature and high wind (wind chill)
        ta = 5.0
        tmrt = 5.0
        wind = 5.0  # High wind
        rh = 70.0
        
        utci = utci_calculator(ta, tmrt, wind, rh)
        
        # UTCI should be lower than air temperature due to wind chill
        self.assertIsInstance(utci, (int, float, np.ndarray))
        self.assertTrue(utci < ta, f"UTCI {utci} should be < air temp {ta} due to wind")

    def test_utci_extreme_heat(self):
        """Test UTCI calculation for extreme heat conditions."""
        from solweig_gpu.calculate_utci import utci_calculator
        
        # Extreme heat
        ta = 40.0
        tmrt = 60.0
        wind = 0.3
        rh = 30.0
        
        utci = utci_calculator(ta, tmrt, wind, rh)
        
        # Should return a valid number (not NaN or inf)
        self.assertIsInstance(utci, (int, float, np.ndarray))
        self.assertFalse(np.isnan(utci))
        self.assertFalse(np.isinf(utci))

    def test_utci_extreme_cold(self):
        """Test UTCI calculation for extreme cold conditions."""
        from solweig_gpu.calculate_utci import utci_calculator
        
        # Extreme cold
        ta = -10.0
        tmrt = -10.0
        wind = 10.0
        rh = 80.0
        
        utci = utci_calculator(ta, tmrt, wind, rh)
        
        # Should return a valid number (not NaN or inf)
        self.assertIsInstance(utci, (int, float, np.ndarray))
        self.assertFalse(np.isnan(utci))
        self.assertFalse(np.isinf(utci))

    def test_utci_input_validation(self):
        """Test that invalid inputs are handled correctly."""
        from solweig_gpu.calculate_utci import utci_calculator
        
        # Test with valid but edge case inputs
        try:
            # Very low wind (should still work)
            utci = utci_calculator(20.0, 20.0, 0.1, 50.0)
            self.assertIsInstance(utci, (int, float, np.ndarray))
            
            # Very high RH (100%)
            utci = utci_calculator(25.0, 25.0, 1.0, 100.0)
            self.assertIsInstance(utci, (int, float, np.ndarray))
            
            # Low RH
            utci = utci_calculator(25.0, 25.0, 1.0, 10.0)
            self.assertIsInstance(utci, (int, float, np.ndarray))
        except Exception as e:
            self.fail(f"UTCI calculation failed with valid inputs: {e}")


class TestUTCIArrayProcessing(unittest.TestCase):
    """Test UTCI calculation with array inputs."""

    def test_utci_2d_array(self):
        """Test UTCI calculation with 2D arrays."""
        from solweig_gpu.calculate_utci import utci_calculator
        
        # Create 2D arrays (e.g., 5x5 spatial grid)
        shape = (5, 5)
        ta = np.full(shape, 25.0, dtype=np.float32)
        tmrt = np.full(shape, 30.0, dtype=np.float32)
        wind = np.full(shape, 1.5, dtype=np.float32)
        rh = np.full(shape, 50.0, dtype=np.float32)
        
        utci = utci_calculator(ta, tmrt, wind, rh)
        
        # Should return array of same shape
        self.assertEqual(utci.shape, shape)
        # All values should be valid
        self.assertFalse(np.any(np.isnan(utci)))
        self.assertFalse(np.any(np.isinf(utci)))

    def test_utci_3d_array(self):
        """Test UTCI calculation with 3D arrays (spatial + temporal)."""
        from solweig_gpu.calculate_utci import utci_calculator
        
        # Create 3D arrays (e.g., 3 time steps x 4x4 spatial grid)
        shape = (3, 4, 4)
        ta = np.random.uniform(20, 30, shape).astype(np.float32)
        tmrt = np.random.uniform(25, 35, shape).astype(np.float32)
        wind = np.random.uniform(0.5, 3.0, shape).astype(np.float32)
        rh = np.random.uniform(30, 70, shape).astype(np.float32)
        
        utci = utci_calculator(ta, tmrt, wind, rh)
        
        # Should return array of same shape
        self.assertEqual(utci.shape, shape)
        # All values should be valid
        self.assertFalse(np.any(np.isnan(utci)))
        self.assertFalse(np.any(np.isinf(utci)))

    def test_utci_nan_handling(self):
        """Test that NaN values are handled correctly."""
        from solweig_gpu.calculate_utci import utci_calculator
        
        # Create array with some NaN values
        ta = np.array([25.0, np.nan, 30.0], dtype=np.float32)
        tmrt = np.array([25.0, 30.0, np.nan], dtype=np.float32)
        wind = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        rh = np.array([50.0, 50.0, 50.0], dtype=np.float32)
        
        try:
            utci = utci_calculator(ta, tmrt, wind, rh)
            # Function should handle NaNs gracefully (either propagate or handle)
            self.assertEqual(len(utci), 3)
        except Exception:
            # If function doesn't handle NaNs, it should at least not crash silently
            pass


if __name__ == '__main__':
    unittest.main()

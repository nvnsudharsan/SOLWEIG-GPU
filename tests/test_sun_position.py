"""
Test suite for the sun_position module.

This module tests the solar position calculations to ensure accuracy.
"""

import unittest
import numpy as np
import datetime


class TestSolarPosition(unittest.TestCase):
    """Test solar position calculations."""

    def test_solar_position_noon(self):
        """Test solar position calculation at solar noon."""
        from solweig_gpu.sun_position import sun_position
        
        # Test at solar noon in summer
        # Austin, TX: 30.27°N, 97.74°W on summer day
        time_dict = {
            'year': 2020,
            'month': 7,
            'day': 18,
            'hour': 12,
            'minute': 0,
            'second': 0,
            'timezone': -5
        }
        location_dict = {
            'latitude': 30.27,
            'longitude': -97.74,
            'altitude': 0
        }
        
        sun = sun_position(time_dict, location_dict)
        
        # Should return a dict with zenith and azimuth
        self.assertIsInstance(sun, dict)
        self.assertIn('zenith', sun)
        self.assertIn('azimuth', sun)
        # At noon in summer, zenith should be low (high altitude)
        self.assertLess(sun['zenith'], 30.0, "Solar zenith should be low at summer noon")

    def test_solar_position_sunrise(self):
        """Test solar position calculation at sunrise."""
        from solweig_gpu.sun_position import sun_position
        
        # Austin, TX at sunrise (approximately 6:30 AM)
        time_dict = {
            'year': 2020,
            'month': 7,
            'day': 18,
            'hour': 6,
            'minute': 30,
            'second': 0,
            'timezone': -5
        }
        location_dict = {
            'latitude': 30.27,
            'longitude': -97.74,
            'altitude': 0
        }
        
        sun = sun_position(time_dict, location_dict)
        
        # At sunrise, zenith should be close to 90 (altitude close to 0)
        self.assertIsInstance(sun, dict)
        self.assertTrue(75 <= sun['zenith'] <= 95, f"Zenith {sun['zenith']} should be near 90 at sunrise")

    def test_solar_position_sunset(self):
        """Test solar position calculation at sunset."""
        from solweig_gpu.sun_position import sun_position
        
        # Austin, TX at sunset (approximately 8:00 PM)
        time_dict = {
            'year': 2020,
            'month': 7,
            'day': 18,
            'hour': 20,
            'minute': 0,
            'second': 0,
            'timezone': -5
        }
        location_dict = {
            'latitude': 30.27,
            'longitude': -97.74,
            'altitude': 0
        }
        
        sun = sun_position(time_dict, location_dict)
        
        # At sunset, zenith should be close to 90 (altitude close to 0)
        self.assertIsInstance(sun, dict)
        self.assertTrue(75 <= sun['zenith'] <= 95, f"Zenith {sun['zenith']} should be near 90 at sunset")

    def test_solar_azimuth_range(self):
        """Test that solar azimuth is within valid range [0, 360]."""
        from solweig_gpu.sun_position import sun_position
        
        # Test multiple times of day
        location_dict = {
            'latitude': 30.27,
            'longitude': -97.74,
            'altitude': 0
        }
        
        for hour in [6, 9, 12, 15, 18]:
            time_dict = {
                'year': 2020,
                'month': 7,
                'day': 18,
                'hour': hour,
                'minute': 0,
                'second': 0,
                'timezone': -5
            }
            sun = sun_position(time_dict, location_dict)
            # Azimuth should be in valid range
            self.assertTrue(0 <= sun['azimuth'] <= 360, 
                          f"Azimuth {sun['azimuth']} out of range [0, 360] at hour {hour}")

    def test_solar_altitude_range(self):
        """Test that solar altitude is within valid range [-90, 90]."""
        from solweig_gpu.sun_position import sun_position
        
        # Test multiple times including night
        location_dict = {
            'latitude': 30.27,
            'longitude': -97.74,
            'altitude': 0
        }
        
        for hour in [0, 6, 12, 18, 23]:
            time_dict = {
                'year': 2020,
                'month': 7,
                'day': 18,
                'hour': hour,
                'minute': 0,
                'second': 0,
                'timezone': -5
            }
            sun = sun_position(time_dict, location_dict)
            # Zenith should be in valid range [0, 180], which means altitude in [-90, 90]
            altitude = 90 - sun['zenith']
            self.assertTrue(-90 <= altitude <= 90, 
                          f"Altitude {altitude} out of range [-90, 90] at hour {hour}")


class TestDayLength(unittest.TestCase):
    """Test day length calculations."""

    def test_summer_solstice(self):
        """Test day length calculation at summer solstice."""
        from solweig_gpu.solweig import daylen
        import torch
        
        # Summer solstice (approximately day 172)
        DOY = torch.tensor(172.0)
        # Austin, TX latitude
        XLAT = torch.tensor(30.27)
        
        DAYL, DEC, SNDN, SNUP = daylen(DOY, XLAT)
        
        # Day length should be greater than 12 hours in northern hemisphere summer
        self.assertGreater(DAYL.item(), 12.0)
        # Declination should be positive (northern hemisphere summer)
        self.assertGreater(DEC.item(), 0.0)

    def test_winter_solstice(self):
        """Test day length calculation at winter solstice."""
        from solweig_gpu.solweig import daylen
        import torch
        
        # Winter solstice (approximately day 355)
        DOY = torch.tensor(355.0)
        # Austin, TX latitude
        XLAT = torch.tensor(30.27)
        
        DAYL, DEC, SNDN, SNUP = daylen(DOY, XLAT)
        
        # Day length should be less than 12 hours in northern hemisphere winter
        self.assertLess(DAYL.item(), 12.0)
        # Declination should be negative (northern hemisphere winter)
        self.assertLess(DEC.item(), 0.0)

    def test_equinox(self):
        """Test day length calculation at equinox."""
        from solweig_gpu.solweig import daylen
        import torch
        
        # Spring equinox (approximately day 80)
        DOY = torch.tensor(80.0)
        # Austin, TX latitude
        XLAT = torch.tensor(30.27)
        
        DAYL, DEC, SNDN, SNUP = daylen(DOY, XLAT)
        
        # Day length should be approximately 12 hours at equinox
        self.assertAlmostEqual(DAYL.item(), 12.0, delta=0.5)
        # Declination should be close to zero
        self.assertAlmostEqual(DEC.item(), 0.0, delta=5.0)


if __name__ == '__main__':
    unittest.main()

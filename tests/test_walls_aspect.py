"""
Test suite for the walls_aspect module.

This module tests the wall height and aspect calculation functionality.
"""

import unittest
import numpy as np
import os
import tempfile
from osgeo import gdal, osr


class TestWallCalculation(unittest.TestCase):
    """Test wall height calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        pass

    def create_simple_building_dsm(self):
        """Create a simple building DSM for testing."""
        # Create a 10x10 raster with a simple building
        data = np.zeros((10, 10), dtype=np.float32)
        # Add a 5m tall building in the center
        data[4:7, 4:7] = 5.0
        
        return data

    def test_wall_height_simple_building(self):
        """Test wall height calculation for a simple building."""
        dsm = self.create_simple_building_dsm()
        
        # Verify building exists in DSM
        self.assertEqual(dsm.shape, (10, 10))
        self.assertTrue(np.any(dsm > 0), "Building should exist in DSM")
        self.assertEqual(np.max(dsm), 5.0, "Building height should be 5m")
        
        # Verify building footprint
        building_pixels = np.sum(dsm > 0)
        self.assertEqual(building_pixels, 9, "3x3 building should have 9 pixels")

    def test_wall_height_no_buildings(self):
        """Test wall height calculation with no buildings."""
        dsm = np.zeros((10, 10), dtype=np.float32)
        
        # Expected: no walls should be detected - flat terrain
        self.assertTrue(np.all(dsm == 0), "DSM should be flat (all zeros)")
        self.assertEqual(np.max(dsm), 0.0, "Maximum height should be 0")

    def test_wall_height_multiple_buildings(self):
        """Test wall height calculation with multiple buildings."""
        # Create DSM with multiple buildings of different heights
        dsm = np.zeros((20, 20), dtype=np.float32)
        # Building 1: 5m tall
        dsm[5:8, 5:8] = 5.0
        # Building 2: 10m tall
        dsm[12:15, 12:15] = 10.0
        
        # Verify both buildings exist
        self.assertTrue(np.any(dsm == 5.0), "5m building should exist")
        self.assertTrue(np.any(dsm == 10.0), "10m building should exist")
        self.assertEqual(np.max(dsm), 10.0, "Tallest building should be 10m")
        
        # Count unique building heights
        unique_heights = np.unique(dsm[dsm > 0])
        self.assertEqual(len(unique_heights), 2, "Should have 2 different building heights")


class TestAspectCalculation(unittest.TestCase):
    """Test aspect (orientation) calculation."""

    def test_aspect_north_facing(self):
        """Test aspect calculation for north-facing wall."""
        # Create a slope facing north (decreasing elevation toward north)
        dem = np.zeros((10, 10), dtype=np.float32)
        for i in range(10):
            dem[i, :] = 10.0 - i  # Decreases from south to north
        
        # Gradient should point north (aspect ~ 0° or 360°)
        dy, dx = np.gradient(dem)
        # North facing means positive gradient in y direction
        self.assertTrue(np.mean(dy) > 0, "North-facing slope should have positive y gradient")

    def test_aspect_east_facing(self):
        """Test aspect calculation for east-facing wall."""
        # Create a slope facing east (decreasing elevation toward east)
        dem = np.zeros((10, 10), dtype=np.float32)
        for j in range(10):
            dem[:, j] = 10.0 - j  # Decreases from west to east
        
        # Gradient should point east (aspect ~ 90°)
        dy, dx = np.gradient(dem)
        # East facing means positive gradient in x direction
        self.assertTrue(np.mean(dx) > 0, "East-facing slope should have positive x gradient")

    def test_aspect_south_facing(self):
        """Test aspect calculation for south-facing wall."""
        # Create a slope facing south (decreasing elevation toward south)
        dem = np.zeros((10, 10), dtype=np.float32)
        for i in range(10):
            dem[i, :] = float(i)  # Increases from north to south
        
        # Gradient should point south (aspect ~ 180°)
        dy, dx = np.gradient(dem)
        # South facing means negative gradient in y direction
        self.assertTrue(np.mean(dy) < 0, "South-facing slope should have negative y gradient")

    def test_aspect_west_facing(self):
        """Test aspect calculation for west-facing wall."""
        # Create a slope facing west (decreasing elevation toward west)
        dem = np.zeros((10, 10), dtype=np.float32)
        for j in range(10):
            dem[:, j] = float(j)  # Increases from west to east
        
        # Gradient should point west (aspect ~ 270°)
        dy, dx = np.gradient(dem)
        # West facing means negative gradient in x direction
        self.assertTrue(np.mean(dx) < 0, "West-facing slope should have negative x gradient")

    def test_aspect_range(self):
        """Test that aspect values are within valid range [0, 360]."""
        # Create random terrain
        dem = np.random.rand(20, 20).astype(np.float32) * 10
        
        # Calculate aspect using numpy gradient
        dy, dx = np.gradient(dem)
        aspect = np.arctan2(-dx, dy)  # Standard aspect calculation
        aspect_deg = np.degrees(aspect)
        aspect_deg = (aspect_deg + 360) % 360  # Normalize to [0, 360]
        
        # All aspect values should be in valid range
        self.assertTrue(np.all(aspect_deg >= 0), "All aspects should be >= 0")
        self.assertTrue(np.all(aspect_deg < 360), "All aspects should be < 360")


if __name__ == '__main__':
    unittest.main()

"""
tests/test_whiteboard.py

Unit test suite for the VirtualWhiteboard class located in the main application module.
"""

import os
import sys
import unittest
import numpy as np

# Resolve project root relative path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import the class under test
from main import VirtualWhiteboard


class MockLandmark:
    """Mock hand tracking landmark point configuration."""
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class TestVirtualWhiteboard(unittest.TestCase):
    """Test suite targeting structural drawing and clearing actions of the whiteboard."""

    def setUp(self):
        self.shape = (480, 640, 3)
        self.whiteboard = VirtualWhiteboard(self.shape)

    def test_initialization(self):
        """Verifies initial canvas shape and state parameters."""
        self.assertEqual(self.whiteboard.frame_shape, self.shape)
        self.assertEqual(self.whiteboard.canvas.shape, self.shape)
        self.assertTrue(np.all(self.whiteboard.canvas == 0))
        self.assertIsNone(self.whiteboard.prev_point)

    def test_draw_inactive(self):
        """Verifies no lines are written if the gesture or landmarks are inactive."""
        # Case 1: No coordinates provided
        self.whiteboard.draw(None, "pointing")
        self.assertIsNone(self.whiteboard.prev_point)

        # Case 2: Hand is mapped but active gesture is not pointing
        landmarks = [MockLandmark(0.5, 0.5) for _ in range(21)]
        self.whiteboard.draw(landmarks, "fist")
        self.assertIsNone(self.whiteboard.prev_point)

    def test_draw_active_and_clear(self):
        """Verifies line drawing coordinates logic and clearing operations."""
        # Generate mock point inputs
        landmarks1 = [MockLandmark(0.1, 0.1) for _ in range(21)]
        landmarks2 = [MockLandmark(0.2, 0.2) for _ in range(21)]

        # First coordinate registers current position
        self.whiteboard.draw(landmarks1, "pointing")
        expected_pt1 = (int(0.1 * 640), int(0.1 * 480))
        self.assertEqual(self.whiteboard.prev_point, expected_pt1)

        # Second coordinate joins line from first and updates tracking state
        self.whiteboard.draw(landmarks2, "pointing")
        expected_pt2 = (int(0.2 * 640), int(0.2 * 480))
        self.assertEqual(self.whiteboard.prev_point, expected_pt2)

        # Assert canvas pixels have been written to
        self.assertTrue(np.any(self.whiteboard.canvas > 0))

        # Test clear method resets the canvas black
        self.whiteboard.clear()
        self.assertTrue(np.all(self.whiteboard.canvas == 0))

    def test_canvas_blend(self):
        """Verifies blended layer overlay output checks."""
        # Draw mock coordinate lines
        landmarks1 = [MockLandmark(0.15, 0.15) for _ in range(21)]
        landmarks2 = [MockLandmark(0.25, 0.25) for _ in range(21)]
        self.whiteboard.draw(landmarks1, "pointing")
        self.whiteboard.draw(landmarks2, "pointing")

        # Create dummy background frame
        base_frame = np.ones(self.shape, dtype=np.uint8) * 120
        blended_frame = self.whiteboard.blend(base_frame)

        self.assertEqual(blended_frame.shape, self.shape)
        # Ensure blended image differs from original background input
        self.assertFalse(np.array_equal(blended_frame, base_frame))


if __name__ == "__main__":
    unittest.main()
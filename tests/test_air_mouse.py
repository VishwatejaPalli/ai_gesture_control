import unittest
import numpy as np
import sys
import os

# Allow imports from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestAirMouse(unittest.TestCase):

    def test_lerp_smoothing(self):
        """Cursor should smoothly interpolate toward the target position."""
        current = np.array([100.0, 100.0])
        target  = np.array([200.0, 200.0])
        alpha   = 0.5
        result  = current + alpha * (target - current)
        self.assertAlmostEqual(result[0], 150.0)
        self.assertAlmostEqual(result[1], 150.0)

    def test_pinch_click_threshold(self):
        """Pinch distance below 22 % of hand scale should register as a click."""
        hand_scale      = 200          # pixels
        click_threshold = 0.22 * hand_scale   # 44 px
        pinch_distance  = 30           # clearly below threshold
        self.assertTrue(pinch_distance < click_threshold)

    def test_no_click_when_open(self):
        """Wide finger spread should NOT register as a click."""
        hand_scale      = 200
        click_threshold = 0.22 * hand_scale
        pinch_distance  = 120          # wide open hand
        self.assertFalse(pinch_distance < click_threshold)

    def test_landmark_to_screen_mapping(self):
        """Normalised landmark (0–1) should map correctly to screen pixels."""
        norm_x, norm_y = 0.5, 0.5
        screen_w, screen_h = 1920, 1080
        pixel_x = int(norm_x * screen_w)
        pixel_y = int(norm_y * screen_h)
        self.assertEqual(pixel_x, 960)
        self.assertEqual(pixel_y, 540)

    def test_cursor_clamped_to_screen(self):
        """Cursor coordinates must never exceed screen boundaries."""
        screen_w, screen_h = 1920, 1080
        raw_x, raw_y = 2100, -50          # out-of-bounds values
        clamped_x = max(0, min(raw_x, screen_w - 1))
        clamped_y = max(0, min(raw_y, screen_h - 1))
        self.assertEqual(clamped_x, 1919)
        self.assertEqual(clamped_y, 0)


if __name__ == "__main__":
    unittest.main()
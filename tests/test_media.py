"""
tests/test_media.py

Unit test suite for the MediaController and VolumeController components.
Mocks UI and system-level API drivers to allow testing on any hardware architecture.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Mock platform-specific system dependencies before importing
sys.modules['pyautogui'] = MagicMock()
sys.modules['keyboard'] = MagicMock()
sys.modules['comtypes'] = MagicMock()
sys.modules['pycaw'] = MagicMock()
sys.modules['pycaw.pycaw'] = MagicMock()

import numpy as np
from media_control.media_controller import MediaController
from media_control.volume_controller import VolumeController


class MockLandmark:
    """Mock coordinate landmark element."""
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class TestMediaController(unittest.TestCase):
    """Verifies interface commands and routing inside the MediaController."""

    def setUp(self):
        self.controller = MediaController()

    @patch('media_control.media_controller.pyautogui')
    def test_discrete_commands(self, mock_pyautogui):
        """Verifies specific keys are pressed via PyAutoGUI."""
        self.controller.play_pause()
        mock_pyautogui.press.assert_called_with("playpause")

        self.controller.next_track()
        mock_pyautogui.press.assert_called_with("nexttrack")

        self.controller.previous_track()
        mock_pyautogui.press.assert_called_with("prevtrack")

        self.controller.mute()
        mock_pyautogui.press.assert_called_with("volumemute")

        self.controller.fullscreen()
        mock_pyautogui.press.assert_called_with("f11")

    def test_execute_dispatcher(self):
        """Verifies string-to-command dispatcher routing."""
        with patch.object(self.controller, 'play_pause', return_value=True) as mock_play:
            status = self.controller.execute("play_pause")
            self.assertTrue(status)
            mock_play.assert_called_once()

        with patch.object(self.controller, 'mute', return_value=True) as mock_mute:
            status = self.controller.execute("mute")
            self.assertTrue(status)
            mock_mute.assert_called_once()


class TestVolumeController(unittest.TestCase):
    """Verifies mathematical interpolation and smoothing in the VolumeController."""

    def setUp(self):
        # Instantiate with a small smoothing factor for instant response testing
        self.vol_controller = VolumeController(
            min_distance=0.05, 
            max_distance=0.25, 
            smoothing_factor=1.0
        )

    def test_finger_distance_math(self):
        """Tests the 2D Euclidean distance calculation helper."""
        landmarks = [MockLandmark(0.0, 0.0) for _ in range(21)]
        landmarks[4] = MockLandmark(0.1, 0.2)
        landmarks[8] = MockLandmark(0.4, 0.6)

        expected_distance = np.sqrt((0.4 - 0.1)**2 + (0.6 - 0.2)**2)
        calculated_distance = self.vol_controller.finger_distance(landmarks)
        self.assertAlmostEqual(calculated_distance, expected_distance)

    def test_distance_to_volume_mapping(self):
        """Tests linear interpolation mapping boundaries."""
        # Lower bound constraint
        vol_min = self.vol_controller.distance_to_volume(0.02)
        self.assertEqual(vol_min, 0.0)

        # Upper bound constraint
        vol_max = self.vol_controller.distance_to_volume(0.30)
        self.assertEqual(vol_max, 100.0)

        # Midpoint mapping check
        vol_mid = self.vol_controller.distance_to_volume(0.15)
        self.assertEqual(vol_mid, 49.99999999999999)


if __name__ == "__main__":
    unittest.main()
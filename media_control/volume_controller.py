"""
media_control/volume_controller.py

Implements hand gesture volume controls based on coordinate landmarks (typically 
retrieved from hand tracking engines like MediaPipe). Maps physical distance between 
the thumb and index finger to system volume level percentage.

Requirements:
    pip install pycaw numpy comtypes
"""

import sys
import time
import logging
from typing import List, Union, Dict, Any, Optional
import numpy as np

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Check if comtypes can be safely initialized (only relevant on Windows)
IS_WINDOWS = sys.platform.startswith("win")


class VolumeController:
    """
    VolumeController tracks distance between hand landmarks and translates 
    the measurements into system audio volume levels on Windows platforms.
    """

    def __init__(
        self, 
        min_distance: float = 0.03, 
        max_distance: float = 0.22, 
        smoothing_factor: float = 0.25
    ):
        """
        Initializes the VolumeController with distance constraints and filters.

        Args:
            min_distance (float): Distance indicating 0% volume (typically relative coordinates).
            max_distance (float): Distance indicating 100% volume (typically relative coordinates).
            smoothing_factor (float): Exponential moving average weight (0.0 to 1.0) to dampen fluctuations.
        """
        self.min_distance: float = min_distance
        self.max_distance: float = max_distance
        self.smoothing_factor: float = max(0.01, min(1.0, smoothing_factor))
        self.current_volume: float = 0.0
        
        # Reference to PyCaw endpoint volume control interface
        self.volume_interface: Any = None
        self._init_audio_interface()

    def _init_audio_interface(self) -> None:
        """
        Attempts to initialize the Windows endpoint audio interface using PyCaw.
        Silently bypasses initialization on non-Windows platforms or environments without audio devices.
        """
        if not IS_WINDOWS:
            logging.debug("Non-Windows OS detected. Native audio interfaces bypassed.")
            return

        try:
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume_interface = interface.QueryInterface(IAudioEndpointVolume)
            
            # Read initial master system volume (mapped from 0.0-1.0 scalar to 0-100 percentage)
            initial_scalar = self.volume_interface.GetMasterVolumeLevelScalar()
            self.current_volume = initial_scalar * 100.0
        except Exception as e:
            logging.debug(f"Audio interface connection failed: {e}")

    def finger_distance(self, landmarks: List[Any]) -> float:
        """
        Calculates the 2D Euclidean distance between the thumb tip (landmark 4)
        and index finger tip (landmark 8).

        Args:
            landmarks: A list of landmarks. Supports objects containing .x and .y attributes,
                       nested dictionaries, or list coordinates.

        Returns:
            float: 2D distance. Defaults to 0.0 if index structure is malformed.
        """
        try:
            pt4 = landmarks[4]
            pt8 = landmarks[8]
            
            # Extract thumb tip coordinates
            if hasattr(pt4, "x") and hasattr(pt4, "y"):
                x1, y1 = pt4.x, pt4.y
            elif isinstance(pt4, dict):
                x1, y1 = pt4["x"], pt4["y"]
            else:
                x1, y1 = pt4[0], pt4[1]

            # Extract index tip coordinates
            if hasattr(pt8, "x") and hasattr(pt8, "y"):
                x2, y2 = pt8.x, pt8.y
            elif isinstance(pt8, dict):
                x2, y2 = pt8["x"], pt8["y"]
            else:
                x2, y2 = pt8[0], pt8[1]

            # Compute Euclidean distance
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            return float(distance)
        except Exception as e:
            logging.debug(f"Could not calculate finger distance: {e}")
            return 0.0

    def distance_to_volume(self, distance: float) -> float:
        """
        Converts a raw distance measurement to a target volume percentage.

        Args:
            distance (float): Calculated raw distance.

        Returns:
            float: Mapped system volume percentage (0.0 to 100.0).
        """
        # Interpolate raw distance between min and max boundaries to the volume range [0.0, 100.0]
        mapped_volume = np.interp(distance, [self.min_distance, self.max_distance], [0.0, 100.0])
        return float(mapped_volume)

    def set_volume(self, value: float) -> None:
        """
        Applies a smoothed target volume value to the active audio device.

        Args:
            value (float): Unfiltered target volume percentage (0.0 to 100.0).
        """
        constrained_val = max(0.0, min(100.0, value))
        
        # Apply exponential moving average filter for smoothing
        self.current_volume = (self.smoothing_factor * constrained_val) + \
                              ((1.0 - self.smoothing_factor) * self.current_volume)
        
        if self.volume_interface:
            try:
                # Convert percentage [0.0 - 100.0] to scalar value [0.0 - 1.0]
                scalar_val = self.current_volume / 100.0
                self.volume_interface.SetMasterVolumeLevelScalar(scalar_val, None)
            except Exception as e:
                logging.debug(f"Failed to apply master volume change: {e}")

    def process(self, landmarks: List[Any]) -> float:
        """
        Integrates tracking landmarks, converts coordinates, applies filters,
        updates the master volume, and returns current status.

        Args:
            landmarks: Tracked hand coordinate landmarks array.

        Returns:
            float: Updated system volume percentage level.
        """
        distance = self.finger_distance(landmarks)
        target_vol = self.distance_to_volume(distance)
        self.set_volume(target_vol)
        return self.current_volume

    def get_visual_feedback(self) -> str:
        """
        Generates a character-based progress indicator representing the system volume state.

        Returns:
            str: Visual feedback indicator string.
        """
        bar_length = 20
        filled_length = int(round(bar_length * self.current_volume / 100.0))
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        return f"Volume: [{bar}] {self.current_volume:.1f}%"


# Simulated Landmark class to run test pipeline on any system
class MockLandmark:
    """
    Mock coordinate class to simulate standard hand coordinate tracking.
    """
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


if __name__ == "__main__":
    print("=== Hand Gesture Volume Controller Simulation ===")
    print("This simulation runs mock tracking points representing your hand movement.")
    print("Fingers pinch to mute (0%), and pull apart to maximize volume (100%).\n")

    # Initialize Controller (using standard relative coordinate parameters)
    vc = VolumeController(min_distance=0.03, max_distance=0.20, smoothing_factor=0.3)

    # Initialize placeholder coordinates array representing standard landmarks (0 through 20)
    mock_landmarks = [MockLandmark(x=0.0, y=0.0) for _ in range(21)]

    # Mock sequence: fingers closing together (decrease volume), then separating apart (increase volume)
    thumb_fixed_x = 0.1
    index_coordinate_steps = [0.32, 0.28, 0.22, 0.16, 0.12, 0.14, 0.20, 0.26, 0.32]

    for index, index_x in enumerate(index_coordinate_steps):
        # Configure landmarks (index 4 is thumb, index 8 is index finger)
        mock_landmarks[4] = MockLandmark(x=thumb_fixed_x, y=0.5)
        mock_landmarks[8] = MockLandmark(x=index_x, y=0.5)

        # Process the mocked frame landmarks
        volume_output = vc.process(mock_landmarks)
        raw_distance = vc.finger_distance(mock_landmarks)
        feedback_bar = vc.get_visual_feedback()

        print(f"Frame {index + 1:02d} | Raw Gap: {raw_distance:.4f} | {feedback_bar}")
        time.sleep(0.15)

    print("\nSimulation process finished.")
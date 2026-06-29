import cv2
import numpy as np
import time
from typing import List, Tuple, Any, Optional


class AirWriter:
    """
    An Air Writing system that records hand movements as vector-like strokes.
    
    This class maintains a history of coordinate paths, allowing for smooth, 
    multi-stroke writing and drawing without the 'teleportation' artifacts
    common in simple drawing scripts.
    """

    def __init__(self):
        """Initialize the AirWriter state."""
        # A list of lists, where each sub-list is a sequence of (x, y) points
        self.strokes: List[List[Tuple[int, int]]] = []
        
        # The stroke currently being recorded
        self.current_stroke: List[Tuple[int, int]] = []
        
        # Rendering settings
        self.color = (0, 255, 0)  # Neon Green (BGR)
        self.thickness = 4
        
        # Save state
        self.last_save_time = 0.0

    def add_point(self, x: int, y: int):
        """Add a point to the active stroke."""
        self.current_stroke.append((x, y))

    def _finalize_stroke(self):
        """Move the current stroke to history and reset the active buffer."""
        if len(self.current_stroke) > 1:
            self.strokes.append(list(self.current_stroke))
        self.current_stroke = []

    def clear(self):
        """Wipe all writing history."""
        self.strokes = []
        self.current_stroke = []

    def save(self, frame_shape: Tuple[int, int]):
        """
        Save the writing to a PNG file on a clean white background.

        Args:
            frame_shape (Tuple[int, int]): The (height, width) of the frame 
                                           the drawing was made on.
        """
        current_time = time.time()
        if current_time - self.last_save_time < 2.0:
            return # Cooldown
            
        # Create a white background
        paper = np.full((frame_shape[0], frame_shape[1], 3), 255, dtype=np.uint8)
        
        # Draw all strokes on the 'paper'
        self.draw_strokes(paper)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(f"air_writing_{timestamp}.png", paper)
        self.last_save_time = current_time
        print(f"Air writing saved as air_writing_{timestamp}.png")

    def draw_strokes(self, frame: np.ndarray):
        """
        Render all recorded strokes and the active stroke onto a frame.
        
        Args:
            frame (np.ndarray): The image to draw on.
        """
        # Combine history and current for rendering
        all_strokes = self.strokes + ([self.current_stroke] if self.current_stroke else [])
        
        for stroke in all_strokes:
            if len(stroke) < 2:
                continue
            
            # Convert list of points to a polyline-compatible format
            points = np.array(stroke, np.int32)
            points = points.reshape((-1, 1, 2))
            
            # Draw smooth anti-aliased lines
            cv2.polylines(
                frame, 
                [points], 
                isClosed=False, 
                color=self.color, 
                thickness=self.thickness, 
                lineType=cv2.LINE_AA
            )

    def process(self, gesture: str, landmarks: List[Any], frame: np.ndarray) -> np.ndarray:
        """
        Interpret gestures to control the writing process.

        Args:
            gesture (str): The current stable gesture.
            landmarks: Hand landmarks from the tracker.
            frame: The current video frame.
        """
        h, w, _ = frame.shape
        
        # 1. Extract Index Finger Tip (Landmark 8)
        x, y = 0, 0
        if landmarks and len(landmarks) > 8:
            x = int(landmarks[8].x * w)
            y = int(landmarks[8].y * h)

        # 2. State Machine Dispatcher
        if gesture == "pointing":
            # Continue writing
            self.add_point(x, y)
        else:
            # If any other gesture is detected, finalize the current stroke
            # This allows the user to 'lift the pen'
            self._finalize_stroke()

        if gesture == "fist":
            self.clear()
        
        elif gesture == "thumbs_up":
            self.save(frame.shape[:2])

        # 3. Render
        output = frame.copy()
        self.draw_strokes(output)
        
        # HUD Information
        cv2.putText(output, "AIR WRITER ACTIVE", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw a cursor at the finger tip
        if gesture == "pointing":
            cv2.circle(output, (x, y), 8, (0, 0, 255), -1)

        return output


if __name__ == "__main__":
    # Test Section: Demonstration
    writer = AirWriter()
    
    # Create a dummy 720p frame
    blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    class MockLandmark:
        def __init__(self, x, y):
            self.x, self.y = x, y

    print("Running AirWriter Demo...")

    # Simulate writing the letter 'V'
    # Stroke 1: Downward
    for i in range(10):
        lm = [MockLandmark(0.4 + (i*0.01), 0.4 + (i*0.02)) for _ in range(21)]
        blank_frame = writer.process("pointing", lm, blank_frame)
    
    # Lift pen (Palm)
    blank_frame = writer.process("palm", None, blank_frame)

    # Stroke 2: Upward
    for i in range(10):
        lm = [MockLandmark(0.5 + (i*0.01), 0.6 - (i*0.02)) for _ in range(21)]
        blank_frame = writer.process("pointing", lm, blank_frame)

    # Save simulation
    writer.process("thumbs_up", None, blank_frame.copy()) # Pass a copy to simulate real usage
    
    print("Demo complete. Check folder for 'air_writing_...png'")
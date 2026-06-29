import cv2
import time
import numpy as np
from typing import List, Any, Optional
from virtual_draw.canvas import DrawingCanvas


class VirtualWhiteboard:
    """
    A gesture-controlled virtual whiteboard that integrates hand tracking
    with a drawing canvas.
    """

    def __init__(self):
        """Initialize the whiteboard with a canvas and default settings."""
        # Initialize the underlying canvas (1280x720 default)
        self.canvas = DrawingCanvas(width=1280, height=720)
        
        # State settings
        self.mode = "Neutral"
        self.current_color = (255, 0, 255)  # Magenta (BGR)
        self.brush_size = 5
        
        # Save debouncing
        self.last_save_time = 0.0
        self.save_cooldown = 2.0  # seconds

    def clear_canvas(self):
        """Clear all drawings from the canvas."""
        self.mode = "Clearing"
        self.canvas.clear()

    def save_canvas(self):
        """Trigger the canvas save method with a timestamped filename."""
        current_time = time.time()
        if current_time - self.last_save_time > self.save_cooldown:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"whiteboard_capture_{timestamp}.png"
            self.canvas.save(filename)
            self.last_save_time = current_time
            self.mode = "Saved!"

    def process(self, gesture: Optional[str], landmarks: List[Any], frame: np.ndarray) -> np.ndarray:
        """
        The main loop for the whiteboard.
        
        1. Translates gestures into actions.
        2. Updates the canvas.
        3. Renders the UI/HUD.
        """
        h, w, _ = frame.shape
        
        # Ensure canvas matches frame dimensions
        if self.canvas.width != w or self.canvas.height != h:
            self.canvas = DrawingCanvas(width=w, height=h)

        # 1. Coordinate Extraction (Landmark 8: Index Finger Tip)
        x, y = 0, 0
        if landmarks and len(landmarks) > 8:
            x = int(landmarks[8].x * w)
            y = int(landmarks[8].y * h)

        # 2. Gesture Dispatcher
        if gesture == "pointing":
            self.mode = "Drawing"
            self.canvas.set_color(self.current_color)
            self.canvas.set_brush_size(self.brush_size)
            self.canvas.draw(x, y)

        elif gesture == "peace":
            self.mode = "Erasing"
            self.canvas.erase(x, y)

        elif gesture == "palm":
            self.mode = "Neutral"
            self.canvas.reset_previous()

        elif gesture == "fist":
            self.clear_canvas()

        elif gesture == "thumbs_up":
            self.save_canvas()
        
        else:
            # For any unrecognized gesture, ensure we don't keep drawing lines
            self.canvas.reset_previous()

        # 3. Blend Canvas with Camera Frame
        output_frame = self.canvas.overlay(frame)

        # 4. Draw HUD (Heads-Up Display)
        self._draw_hud(output_frame)
        
        # If we are in "Drawing" or "Erasing", show a cursor preview
        if gesture in ["pointing", "peace"]:
            cv2.circle(output_frame, (x, y), 10, (255, 255, 255), 2)

        return output_frame

    def _draw_hud(self, frame: np.ndarray):
        """Render the status information on the frame."""
        # Background for the HUD text
        cv2.rectangle(frame, (10, 10), (350, 120), (30, 30, 30), -1)
        
        # Display Mode
        cv2.putText(frame, f"MODE: {self.mode}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display Brush Size
        cv2.putText(frame, f"BRUSH SIZE: {self.brush_size}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display Current Color Indicator
        cv2.putText(frame, "COLOR:", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.rectangle(frame, (110, 85), (140, 105), self.current_color, -1)
        cv2.rectangle(frame, (110, 85), (140, 105), (255, 255, 255), 1)


if __name__ == "__main__":
    # Test Section: Simulation
    whiteboard = VirtualWhiteboard()
    
    # Create a dummy blank frame
    test_img = np.zeros((720, 1280, 3), dtype=np.uint8)

    class MockLandmark:
        def __init__(self, x, y):
            self.x, self.y = x, y

    # Simulate index finger at center
    mock_landmarks = [MockLandmark(0.5, 0.5) for _ in range(21)]

    print("Simulating gestures...")
    
    # 1. Test Drawing
    res = whiteboard.process("pointing", mock_landmarks, test_img)
    print("Processed 'pointing' gesture.")

    # 2. Test Eraser
    res = whiteboard.process("peace", mock_landmarks, res)
    print("Processed 'peace' gesture.")

    # 3. Test Clear
    res = whiteboard.process("fist", mock_landmarks, res)
    print("Processed 'fist' gesture (Canvas Cleared).")

    # 4. Test Save
    whiteboard.process("thumbs_up", mock_landmarks, res)
    print("Processed 'thumbs_up' gesture (Saved).")
    
    print("Test complete.")
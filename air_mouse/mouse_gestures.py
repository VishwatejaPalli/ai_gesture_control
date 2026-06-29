import time
from typing import List, Any, Optional


class MouseGestures:
    """
    Maps recognized hand gestures and landmarks to mouse actions.
    
    Logic:
    - 'pointing': Moves the cursor using Index Finger Tip.
    - 'pinch': Left Click (with debounce).
    - 'peace': Right Click (with debounce).
    - 'fist': Drag (MouseDown).
    - 'palm' or any other: Release Drag (MouseUp).
    """

    def __init__(self, mouse_controller, click_cooldown: float = 0.5):
        """
        Initialize the Mouse Gestures processor.

        Args:
            mouse_controller: An instance of the MouseController class.
            click_cooldown (float): Minimum time in seconds between clicks.
        """
        self.mc = mouse_controller
        self.click_cooldown = click_cooldown
        
        # State tracking
        self.dragging = False
        self.last_click_time = 0.0

    def process(self, gesture: str, landmarks: List[Any], frame_width: int, frame_height: int):
        """
        Dispatch the appropriate mouse action based on the gesture.

        Args:
            gesture (str): The stable gesture name from the state machine.
            landmarks: List of landmarks (each having .x, .y, .z).
            frame_width (int): Width of the camera frame.
            frame_height (int): Height of the camera frame.
        """
        # 1. Handle Cursor Movement (Always move if pointing or pinching/fisting)
        # Note: We allow movement during 'fist' to enable dragging.
        if gesture in ["pointing", "fist", "pinch"]:
            self.handle_cursor(landmarks, frame_width, frame_height)

        # 2. Handle Clicks and Drags
        if gesture == "pinch":
            self.handle_left_click()
        
        elif gesture == "peace":
            self.handle_right_click()
        
        elif gesture == "fist":
            self.start_drag()
        
        elif gesture == "palm":
            # Explicit stop for palm
            self.stop_drag()

        # 3. Transition Logic: If the gesture is no longer 'fist', stop dragging
        if gesture != "fist" and self.dragging:
            self.stop_drag()

    def handle_cursor(self, landmarks: List[Any], frame_width: int, frame_height: int):
        """
        Extract Index Finger Tip (Landmark 8) and move the mouse.
        """
        if not landmarks or len(landmarks) < 9:
            return

        # Landmark 8 is the Index Finger Tip
        idx_tip = landmarks[8]
        
        # MediaPipe landmarks are normalized (0.0 to 1.0). 
        # Convert to pixel coordinates based on frame size.
        pixel_x = idx_tip.x * frame_width
        pixel_y = idx_tip.y * frame_height

        self.mc.move(pixel_x, pixel_y, frame_width, frame_height)

    def handle_left_click(self):
        """Perform a debounced left click."""
        current_time = time.time()
        if current_time - self.last_click_time > self.click_cooldown:
            self.mc.click(button='left')
            self.last_click_time = current_time
            print("[Action] Left Click")

    def handle_right_click(self):
        """Perform a debounced right click."""
        current_time = time.time()
        if current_time - self.last_click_time > self.click_cooldown:
            self.mc.click(button='right')
            self.last_click_time = current_time
            print("[Action] Right Click")

    def start_drag(self):
        """Start the drag (MouseDown) operation."""
        if not self.dragging:
            self.mc.drag_start()
            self.dragging = True
            print("[Action] Drag Started")

    def stop_drag(self):
        """Stop the drag (MouseUp) operation."""
        if self.dragging:
            self.mc.drag_stop()
            self.dragging = False
            print("[Action] Drag Stopped")


if __name__ == "__main__":
    # Mock Objects for Testing
    class MockLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class MockController:
        def move(self, x, y, fw, fh): print(f"Mouse moved to {x}, {y}")
        def click(self, button): print(f"Mouse clicked: {button}")
        def drag_start(self): print("Hardware Mouse Down")
        def drag_stop(self): print("Hardware Mouse Up")

    # Initialize
    mock_mc = MockController()
    gestures = MouseGestures(mock_mc)
    
    # Mock frame data
    W, H = 640, 480
    # Generate 21 dummy landmarks
    dummy_landmarks = [MockLandmark(0.5, 0.5) for _ in range(21)]

    print("--- Test 1: Cursor Movement ---")
    gestures.process("pointing", dummy_landmarks, W, H)

    print("\n--- Test 2: Left Click (Pinch) ---")
    gestures.process("pinch", dummy_landmarks, W, H)

    print("\n--- Test 3: Right Click (Peace) ---")
    gestures.process("peace", dummy_landmarks, W, H)

    print("\n--- Test 4: Drag Start (Fist) ---")
    gestures.process("fist", dummy_landmarks, W, H)

    print("\n--- Test 5: Drag Stop (Transition to Palm) ---")
    gestures.process("palm", dummy_landmarks, W, H)
import pyautogui
from typing import Tuple

# Disable PyAutoGUI's fail-safe to prevent the program from crashing 
# if the cursor hits a corner (common in air-mouse applications)
pyautogui.FAILSAFE = False
# Set PAUSE to 0 for near-instant mouse movement response
pyautogui.PAUSE = 0

class MouseController:
    """
    Handles hardware-level mouse control using PyAutoGUI.
    Includes logic for mapping coordinate systems and handling screen boundaries.
    """

    def __init__(self, screen_padding: int = 100):
        """
        Initialize the Mouse Controller.

        Args:
            screen_padding (int): Pixels to subtract from the camera frame 
                                  boundaries to create an 'active zone'. 
                                  Helps in reaching screen edges easily.
        """
        self.screen_width, self.screen_height = pyautogui.size()
        self.padding = screen_padding
        
        # Track state to prevent redundant clicks
        self.is_pressed = False

    def get_screen_size(self) -> Tuple[int, int]:
        """
        Get the resolution of the primary monitor.
        
        Returns:
            Tuple[int, int]: (width, height) in pixels.
        """
        return self.screen_width, self.screen_height

    def move(self, x: float, y: float, frame_w: int, frame_h: int):
        """
        Map coordinates from the camera frame to the screen and move the cursor.

        Args:
            x (float): Raw X coordinate from the tracker.
            y (float): Raw Y coordinate from the tracker.
            frame_w (int): Width of the input camera frame.
            frame_h (int): Height of the input camera frame.
        """
        # 1. Define the active region inside the camera frame using padding
        # This makes it so the user doesn't have to move their hand to the very
        # edge of the camera view to hit the edge of the monitor.
        active_w = frame_w - (2 * self.padding)
        active_h = frame_h - (2 * self.padding)

        # 2. Normalize coordinates within that active region
        # Subtract padding and then clamp between 0 and 1
        norm_x = (x - self.padding) / active_w
        norm_y = (y - self.padding) / active_h

        # Clamp values to [0.0, 1.0]
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        # 3. Scale to screen resolution
        screen_x = int(norm_x * self.screen_width)
        screen_y = int(norm_y * self.screen_height)

        # 4. Perform the move
        pyautogui.moveTo(screen_x, screen_y)

    def click(self, button: str = 'left'):
        """Perform a single mouse click."""
        pyautogui.click(button=button)

    def double_click(self):
        """Perform a double mouse click."""
        pyautogui.doubleClick()

    def scroll(self, direction: str):
        """
        Perform a scroll action.
        
        Args:
            direction (str): 'up' or 'down'
        """
        amount = 300 if direction == 'up' else -300
        pyautogui.scroll(amount)

    def drag_start(self):
        """Initiate a mouse-down state for dragging."""
        if not self.is_pressed:
            pyautogui.mouseDown()
            self.is_pressed = True

    def drag_stop(self):
        """Release the mouse-down state."""
        if self.is_pressed:
            pyautogui.mouseUp()
            self.is_pressed = False


if __name__ == "__main__":
    # Test Section: Demonstration of screen mapping
    controller = MouseController(screen_padding=50)
    w, h = controller.get_screen_size()
    
    print(f"Detected Screen Size: {w}x{h}")
    print("Simulating movement across a 640x480 camera frame...")

    # Test center movement
    print("Moving to center...")
    controller.move(320, 240, 640, 480)
    
    # Test top-left (should hit 0,0 because of padding)
    print("Moving to top-left (with padding)...")
    controller.move(50, 50, 640, 480)
    
    # Test clicking
    print("Performing test click...")
    controller.click()
    
    print("Test complete.")
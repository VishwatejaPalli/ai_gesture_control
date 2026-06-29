import cv2
import numpy as np
from typing import Tuple, Optional


class DrawingCanvas:
    """
    A production-quality digital canvas for virtual drawing.
    
    Handles drawing logic, erasing, and sophisticated blending (overlaying)
    the drawing onto a live camera feed.
    """

    def __init__(self, width: int = 1280, height: int = 720):
        """
        Initialize the Drawing Canvas.

        Args:
            width (int): Width of the canvas.
            height (int): Height of the canvas.
        """
        self.width = width
        self.height = height
        
        # Initialize canvas as a black image (3 channels)
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Default settings
        self.color = (255, 0, 255)  # Default: Magenta (BGR)
        self.brush_size = 5
        self.eraser_size = 50
        
        # State tracking for continuous lines
        self.prev_point: Optional[Tuple[int, int]] = None

    def clear(self):
        """Clear the canvas completely."""
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.reset_previous()

    def set_color(self, color: Tuple[int, int, int]):
        """
        Set the brush color.
        Args:
            color (Tuple[int, int, int]): Color in BGR format.
        """
        self.color = color

    def set_brush_size(self, size: int):
        """Set the thickness of the drawing line."""
        self.brush_size = size

    def reset_previous(self):
        """Reset the previous point. Call this when the hand is lifted or lost."""
        self.prev_point = None

    def draw(self, x: int, y: int):
        """
        Draw a line from the last known point to the current (x, y).
        
        Args:
            x (int): Current X coordinate.
            y (int): Current Y coordinate.
        """
        if self.prev_point is None:
            self.prev_point = (x, y)

        # Draw an anti-aliased line
        cv2.line(
            self.canvas, 
            self.prev_point, 
            (x, y), 
            self.color, 
            self.brush_size, 
            cv2.LINE_AA
        )
        
        # Update state
        self.prev_point = (x, y)

    def erase(self, x: int, y: int):
        """
        Erase a circular area around the given coordinates.
        """
        cv2.circle(self.canvas, (x, y), self.eraser_size, (0, 0, 0), -1)
        # We reset previous to prevent lines "jumping" through the erased area
        self.reset_previous()

    def overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Blend the drawing canvas with the provided camera frame.
        Uses masking to ensure the drawing is opaque and vibrant.

        Args:
            frame (np.ndarray): The camera frame (BGR).

        Returns:
            np.ndarray: The combined output image.
        """
        # Ensure frame size matches canvas size
        if frame.shape[:2] != self.canvas.shape[:2]:
            frame = cv2.resize(frame, (self.width, self.height))

        # 1. Convert canvas to grayscale
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)

        # 2. Create a binary mask of the drawing (where it's not black)
        _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)

        # 3. Invert the mask
        mask_inv = cv2.bitwise_not(mask)

        # 4. Use the inverse mask to punch a 'hole' in the original frame
        # where the drawing will go
        img_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # 5. Use the mask to extract only the drawing from the canvas
        img_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)

        # 6. Add the two images
        result = cv2.add(img_bg, img_fg)
        
        return result

    def save(self, filename: str = "artwork.png"):
        """Save the current canvas to disk."""
        cv2.imwrite(filename, self.canvas)
        print(f"Canvas saved to {filename}")


if __name__ == "__main__":
    # Test Section
    # Create a dummy camera frame (dark blue background)
    test_frame = np.full((720, 1280, 3), (50, 0, 0), dtype=np.uint8)
    
    cv_canvas = DrawingCanvas(width=1280, height=720)
    
    # 1. Simulate drawing a triangle
    print("Drawing test shapes...")
    cv_canvas.set_color((0, 255, 0)) # Green
    cv_canvas.draw(100, 100)
    cv_canvas.draw(200, 100)
    cv_canvas.draw(150, 200)
    cv_canvas.draw(100, 100)
    cv_canvas.reset_previous()
    
    # 2. Simulate drawing a thick blue line
    cv_canvas.set_color((255, 0, 0)) # Blue
    cv_canvas.set_brush_size(20)
    cv_canvas.draw(400, 400)
    cv_canvas.draw(800, 400)
    
    # 3. Simulate an erase action
    print("Simulating eraser...")
    cv_canvas.erase(150, 150)
    
    # 4. Perform Overlay
    output = cv_canvas.overlay(test_frame)
    
    # Show results if running in a GUI environment
    # cv2.imshow("Test Output", output)
    # cv2.waitKey(0)
    
    print("Overlay test completed. Dimensions:", output.shape)
    
    # 5. Save test
    cv_canvas.save("test_canvas.png")
import collections
from typing import Tuple, Optional


class CursorFilter:
    """
    A dual-stage cursor smoothing filter combining Moving Average (MA) 
    and Exponential Moving Average (EMA) to provide stable cursor movement 
    for gesture-controlled systems.
    """

    def __init__(self, alpha: float = 0.25, history_size: int = 5):
        """
        Initialize the Cursor Filter.

        Args:
            alpha (float): Smoothing factor for Exponential Smoothing (0.0 to 1.0).
                           Lower values are smoother but add more latency.
            history_size (int): Number of frames for the Moving Average window.
        """
        self.alpha = alpha
        self.history_size = history_size
        
        # Buffer for Moving Average
        self.history = collections.deque(maxlen=history_size)
        
        # State for Exponential Smoothing
        self.previous_x: Optional[float] = None
        self.previous_y: Optional[float] = None

    def moving_average(self, x: float, y: float) -> Tuple[float, float]:
        """
        Stage 1: Calculate the mean of the last N coordinates.
        This helps reduce jitter from sensor noise.
        """
        self.history.append((x, y))
        
        # Sum all x and y in history
        sum_x = sum(point[0] for point in self.history)
        sum_y = sum(point[1] for point in self.history)
        count = len(self.history)
        
        return sum_x / count, sum_y / count

    def exponential_smoothing(self, x: float, y: float) -> Tuple[float, float]:
        """
        Stage 2: Apply Exponential Moving Average (EMA).
        Formula: new = alpha * current + (1 - alpha) * previous
        """
        # If this is the first point, initialize previous values
        if self.previous_x is None or self.previous_y is None:
            self.previous_x, self.previous_y = x, y
            return x, y

        # Apply the EMA formula
        smoothed_x = (self.alpha * x) + (1 - self.alpha) * self.previous_x
        smoothed_y = (self.alpha * y) + (1 - self.alpha) * self.previous_y

        # Update state for next frame
        self.previous_x, self.previous_y = smoothed_x, smoothed_y

        return smoothed_x, smoothed_y

    def filter(self, x: float, y: float) -> Tuple[float, float]:
        """
        Apply the full filtering pipeline:
        1. Moving Average
        2. Exponential Smoothing
        """
        # Step 1: Reduce local jitter via Moving Average
        ma_x, ma_y = self.moving_average(x, y)
        
        # Step 2: Smooth the transition via Exponential Smoothing
        final_x, final_y = self.exponential_smoothing(ma_x, ma_y)
        
        return final_x, final_y

    def reset(self):
        """
        Clears the filter history and state.
        Call this when a hand is lost or a new tracking session begins.
        """
        self.history.clear()
        self.previous_x = None
        self.previous_y = None


if __name__ == "__main__":
    # Test Section: Demonstrate smoothing with noisy data
    # Scenario: Hand is moving toward (100, 100) but has sensor jitter
    cursor_filter = CursorFilter(alpha=0.3, history_size=5)

    noisy_inputs = [
        (10, 10),
        (12, 8),    # Jitter
        (15, 15),
        (14, 16),   # Jitter
        (20, 20),
        (22, 18),   # Jitter
        (100, 100), # Sudden movement
        (105, 95),  # Jitter at destination
        (100, 100),
        (100, 100)
    ]

    print(f"{'Input (X, Y)':<20} | {'Filtered (X, Y)':<20}")
    print("-" * 45)

    for raw_x, raw_y in noisy_inputs:
        smooth_x, smooth_y = cursor_filter.filter(raw_x, raw_y)
        print(f"({raw_x:>3}, {raw_y:>3})           ->   ({smooth_x:>6.2f}, {smooth_y:>6.2f})")

    print("\nResetting filter...")
    cursor_filter.reset()
import time
from typing import Optional

class GestureStateMachine:
    """
    A production-ready state machine for hand gesture recognition.
    
    This class handles:
    1. Stability (Hysteresis): Requires a gesture to persist for N frames.
    2. Debouncing (Cooldown): Enforces a minimum time between event triggers.
    3. One-Shot Triggering: Ensures a continuous gesture triggers an event only once.
    4. Jitter/Noise Filtering: Ignores transient or noisy gesture detections.
    """

    def __init__(self, hold_frames: int = 5, cooldown: float = 0.5):
        """
        Initialize the Gesture State Machine.

        Args:
            hold_frames (int): Number of consecutive frames a gesture must 
                               persist to be accepted.
            cooldown (float): Minimum time in seconds required between 
                              accepted gesture triggers.
        """
        # Configuration
        self.hold_frames = hold_frames
        self.cooldown = cooldown

        # State tracking
        self.candidate_gesture: Optional[str] = None
        self.frame_count: int = 0
        self.last_trigger_time: float = 0.0
        # Flag to ensure a continuous gesture only triggers once
        self.triggered_for_candidate: bool = False

    def update(self, gesture: Optional[str]) -> Optional[str]:
        """
        Process a gesture from a new frame.

        Args:
            gesture (str, optional): The gesture detected in the current frame.

        Returns:
            str or None: The confirmed gesture name if it meets all criteria, 
                         otherwise None.
        """
        # 1. Update Candidate and Frame Count (Hysteresis Logic)
        if gesture == self.candidate_gesture:
            self.frame_count += 1
        else:
            # New gesture detected, or input is None. Reset the candidate.
            self.candidate_gesture = gesture
            self.frame_count = 1
            self.triggered_for_candidate = False # Allow triggering for this new candidate

        # 2. Check for a valid, stable, and ready-to-trigger gesture
        # - Must have a non-None candidate gesture
        # - Must have been held for the required number of frames
        # - Must not have been triggered for this continuous gesture instance yet
        if (self.candidate_gesture is not None and
            self.frame_count >= self.hold_frames and
            not self.triggered_for_candidate):
            
            current_time = time.time()
            elapsed = current_time - self.last_trigger_time

            # 3. Enforce Cooldown
            if elapsed >= self.cooldown:
                # All conditions met. Trigger the event.
                self.last_trigger_time = current_time
                self.triggered_for_candidate = True # Prevent re-triggering until gesture changes
                return self.candidate_gesture

        # No event triggered in this frame
        return None

    def reset(self):
        """
        Reset all internal state tracking variables to their initial values.
        Useful when the hand leaves the frame or the mode is changed.
        """
        self.candidate_gesture = None
        self.frame_count = 0
        self.last_trigger_time = 0.0
        self.triggered_for_candidate = False


if __name__ == "__main__":
    # Test demonstrations
    # Using a slightly lower cooldown for faster test execution
    sm = GestureStateMachine(hold_frames=5, cooldown=0.2)

    def run_test(name, sequence, expected_triggers):
        print(f"--- Running Test: {name} ---")
        sm.reset()
        triggers = []
        for i, g in enumerate(sequence):
            result = sm.update(g)
            print(f"Frame {i+1:<2}: Input='{str(g):<7}', Output='{str(result):<7}'")
            if result:
                triggers.append(result)
        print(f"Result: Got {triggers}, Expected {expected_triggers}")
        assert triggers == expected_triggers
        print("--- Test PASSED ---\n")

    # 1. Basic stability test
    test_1_seq = ["fist"] * 4 + ["fist"] + ["fist"] * 5
    run_test("1. Basic Stability", test_1_seq, ["fist"])

    # 2. Jitter/Noise filtering test
    test_2_seq = ["palm", "palm", "fist", "palm", "palm", "palm", "palm"]
    run_test("2. Jitter Filtering", test_2_seq, ["palm"])

    # 3. One-shot trigger test (should not re-trigger)
    test_3_seq = ["thumb"] * 10
    run_test("3. One-Shot Trigger", test_3_seq, ["thumb"])

    # 4. Cooldown test (should not trigger peace)
    test_4_seq = ["fist"] * 5 + ["peace"] * 5
    run_test("4. Cooldown", test_4_seq, ["fist"])

    # 5. Cooldown + Recovery test
    print("Waiting for cooldown...")
    time.sleep(0.25)
    test_5_seq = ["fist"] * 5 + ["peace"] * 5
    run_test("5. Cooldown Recovery", test_5_seq, ["fist", "peace"])

    # 6. Re-triggering after gesture change
    test_6_seq = ["fist"] * 5 + ["palm"] * 5 + ["fist"] * 5
    run_test("6. Re-triggering", test_6_seq, ["fist", "palm", "fist"])

    # 7. Filtering None inputs
    test_7_seq = ["point", "point", None, "point", "point", "point"]
    run_test("7. None Input Filtering", test_7_seq, ["point"])
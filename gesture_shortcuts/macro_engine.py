"""
gesture_shortcuts/macro_engine.py

A gesture macro engine designed to map user gestures to actions. Features
include custom mapping registration, run-time modification, debounce
(cooldown) timers, execution history tracking, and JSON persistence.
"""

import json
import time
from typing import Dict, List, Any, Optional


class MacroEngine:
    """
    A class that maps gestures to actions, handles cooldown timers to prevent
    duplicate activations, manages macro execution, and supports loading/saving
    configurations via JSON.
    """

    def __init__(self, default_cooldown: float = 1.0):
        """
        Initializes the MacroEngine with default mappings and cooldown.

        Args:
            default_cooldown (float): Minimum time in seconds between consecutive
                                      executions of the same gesture.
        """
        self.default_cooldown: float = default_cooldown
        
        # Default mappings as requested
        self.mappings: Dict[str, str] = {
            "thumbs_up": "open_browser",
            "peace": "copy",
            "three": "paste",
            "four": "save",
            "fist": "undo",
            "palm": "stop"
        }
        
        # Track the last execution timestamp for each gesture
        self.last_execution_times: Dict[str, float] = {}
        
        # Track execution history
        self.history: List[Dict[str, Any]] = []

    def register_macro(self, gesture: str, action: str) -> None:
        """
        Registers a new gesture mapping or updates an existing one at runtime.

        Args:
            gesture (str): The name of the gesture (e.g., 'thumbs_up').
            action (str): The action to associate with the gesture (e.g., 'open_browser').
        """
        self.mappings[gesture] = action

    def remove_macro(self, gesture: str) -> bool:
        """
        Removes a registered gesture mapping.

        Args:
            gesture (str): The name of the gesture to remove.

        Returns:
            bool: True if the gesture was mapped and successfully removed, False otherwise.
        """
        if gesture in self.mappings:
            del self.mappings[gesture]
            # Clear historical cooldown for this gesture to free memory
            self.last_execution_times.pop(gesture, None)
            return True
        return False

    def execute(self, gesture: str) -> Optional[str]:
        """
        Executes the action associated with a gesture if it is registered
        and has passed the cooldown period (debounce check).

        Args:
            gesture (str): The name of the gesture to trigger.

        Returns:
            Optional[str]: The corresponding action string if successfully executed,
                           None if the gesture is unknown or blocked by the cooldown.
        """
        if gesture not in self.mappings:
            return None

        action = self.mappings[gesture]
        current_time = time.time()
        last_time = self.last_execution_times.get(gesture, 0.0)

        # Debounce (cooldown) check
        if current_time - last_time < self.default_cooldown:
            self.history.append({
                "timestamp": current_time,
                "gesture": gesture,
                "action": action,
                "status": "ignored_cooldown"
            })
            return None

        # Record execution time and log history
        self.last_execution_times[gesture] = current_time
        self.history.append({
            "timestamp": current_time,
            "gesture": gesture,
            "action": action,
            "status": "executed"
        })
        return action

    def load_config(self, filename: str) -> None:
        """
        Loads gesture-to-action mappings and cooldown values from a JSON configuration file.

        Args:
            filename (str): Path to the JSON configuration file.
        """
        try:
            with open(filename, 'r') as file:
                config_data = json.load(file)
                if isinstance(config_data, dict):
                    # Check if standard config layout or simplified dict
                    if "mappings" in config_data and isinstance(config_data["mappings"], dict):
                        self.mappings = config_data["mappings"]
                        self.default_cooldown = config_data.get("cooldown", self.default_cooldown)
                    else:
                        self.mappings = config_data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise IOError(f"Error loading configuration from {filename}: {e}")

    def save_config(self, filename: str) -> None:
        """
        Saves the current mappings and cooldown configurations to a JSON file.

        Args:
            filename (str): Path to the destination JSON file.
        """
        output_data = {
            "cooldown": self.default_cooldown,
            "mappings": self.mappings
        }
        try:
            with open(filename, 'w') as file:
                json.dump(output_data, file, indent=4)
        except IOError as e:
            raise IOError(f"Error saving configuration to {filename}: {e}")


if __name__ == "__main__":
    import os

    print("=== Gesture Shortcuts Macro Engine Demo ===")
    
    # 1. Initialize engine
    engine = MacroEngine(default_cooldown=1.0)
    print("\nInitial mappings configuration:")
    for gesture, action in engine.mappings.items():
        print(f"  {gesture} -> {action}")

    # 2. Test Execution and Debouncing
    print("\n--- Testing execution and debounce functionality ---")
    
    # First execution (should succeed)
    print("Executing 'thumbs_up'...")
    res1 = engine.execute("thumbs_up")
    print(f"Result: {res1} (Expected: open_browser)")

    # Immediate second execution (should be blocked by cooldown)
    print("Executing 'thumbs_up' again immediately...")
    res2 = engine.execute("thumbs_up")
    print(f"Result: {res2} (Expected: None due to cooldown)")

    # Sleep to clear the cooldown
    sleep_time = 1.1
    print(f"Sleeping for {sleep_time} seconds...")
    time.sleep(sleep_time)

    # Executing after cooldown (should succeed)
    print("Executing 'thumbs_up' after cooldown...")
    res3 = engine.execute("thumbs_up")
    print(f"Result: {res3} (Expected: open_browser)")

    # 3. Runtime modifications
    print("\n--- Testing runtime modifications ---")
    
    # Registering a new gesture
    print("Registering 'ok' -> 'confirm_selection'")
    engine.register_macro("ok", "confirm_selection")
    print(f"Executing 'ok': {engine.execute('ok')}")

    # Removing a gesture
    print("Removing 'palm' gesture")
    removed = engine.remove_macro("palm")
    print(f"Removal success status: {removed}")
    print(f"Executing 'palm' (should be None): {engine.execute('palm')}")

    # 4. Save and Load Configurations
    config_filepath = "gesture_config_test.json"
    print(f"\n--- Testing JSON configuration persistence ({config_filepath}) ---")
    
    try:
        engine.save_config(config_filepath)
        print(f"Config saved to {config_filepath}.")

        # Create a blank engine and load configuration
        new_engine = MacroEngine()
        new_engine.load_config(config_filepath)
        print("Loaded mappings into a new engine instance:")
        for gesture, action in new_engine.mappings.items():
            print(f"  {gesture} -> {action}")
    finally:
        # Cleanup test config file
        if os.path.exists(config_filepath):
            os.remove(config_filepath)
            print(f"\nCleaned up temporary config file: {config_filepath}")

    # 5. Reviewing Execution History
    print("\n--- Reviewing Execution History Log ---")
    for idx, entry in enumerate(engine.history, 1):
        print(f"{idx}. Time: {entry['timestamp']:.4f} | Gesture: {entry['gesture']} | Action: {entry['action']} | Status: {entry['status']}")
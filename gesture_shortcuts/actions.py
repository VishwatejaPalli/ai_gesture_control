import os
import subprocess
import webbrowser
import pyautogui
import keyboard
import time
from datetime import datetime
from typing import Callable, Dict

class ActionExecutor:
    """
    Executes desktop automation tasks and system shortcuts.
    
    This class maps gesture-triggered action names to physical 
    keyboard combinations or system process calls.
    """

    def __init__(self):
        # Mapping action strings to internal methods
        self.action_map: Dict[str, Callable[[], bool]] = {
            "open_browser": self.open_browser,
            "open_calculator": self.open_calculator,
            "open_notepad": self.open_notepad,
            "take_screenshot": self.take_screenshot,
            "copy": self.copy,
            "paste": self.paste,
            "undo": self.undo,
            "redo": self.redo,
            "save": self.save,
            "lock_screen": self.lock_screen,
            "open_explorer": self.open_explorer,
            "volume_up": self.volume_up,
            "volume_down": self.volume_down
        }

    def execute(self, action_name: str) -> bool:
        """
        Dispatches and executes the requested action.
        
        Args:
            action_name (str): The key representing the action to perform.
            
        Returns:
            bool: True if execution started successfully, False otherwise.
        """
        if action_name in self.action_map:
            print(f"[ActionExecutor] Executing: {action_name}")
            return self.action_map[action_name]()
        
        print(f"[ActionExecutor] Action '{action_name}' not recognized.")
        return False

    # --- System Applications ---

    def open_browser(self) -> bool:
        """Opens the default system web browser."""
        try:
            webbrowser.open("https://www.google.com")
            return True
        except Exception as e:
            print(f"Error opening browser: {e}")
            return False

    def open_calculator(self) -> bool:
        """Launches the Windows Calculator."""
        try:
            subprocess.Popen('calc.exe')
            return True
        except Exception as e:
            print(f"Error opening calculator: {e}")
            return False

    def open_notepad(self) -> bool:
        """Launches Notepad."""
        try:
            subprocess.Popen('notepad.exe')
            return True
        except Exception as e:
            print(f"Error opening notepad: {e}")
            return False

    def open_explorer(self) -> bool:
        """Opens Windows File Explorer."""
        try:
            os.startfile("explorer.exe")
            return True
        except Exception as e:
            print(f"Error opening explorer: {e}")
            return False

    # --- System Controls ---

    def take_screenshot(self) -> bool:
        """Captures the screen and saves it to the current directory."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            pyautogui.screenshot(filename)
            print(f"Screenshot saved as {filename}")
            return True
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            return False

    def lock_screen(self) -> bool:
        """Locks the Windows workstation."""
        try:
            # Standard Windows Lock command
            os.system("rundll32.exe user32.dll,LockWorkStation")
            return True
        except Exception as e:
            print(f"Error locking screen: {e}")
            return False

    def volume_up(self) -> bool:
        """Increases system volume."""
        try:
            keyboard.press_and_release("volume up")
            return True
        except Exception as e:
            print(f"Error changing volume: {e}")
            return False

    def volume_down(self) -> bool:
        """Decreases system volume."""
        try:
            keyboard.press_and_release("volume down")
            return True
        except Exception as e:
            print(f"Error changing volume: {e}")
            return False

    # --- Keyboard Shortcuts (Editing) ---

    def copy(self) -> bool:
        """Simulates Ctrl+C."""
        try:
            keyboard.press_and_release('ctrl+c')
            return True
        except Exception as e:
            print(f"Error performing copy: {e}")
            return False

    def paste(self) -> bool:
        """Simulates Ctrl+V."""
        try:
            keyboard.press_and_release('ctrl+v')
            return True
        except Exception as e:
            print(f"Error performing paste: {e}")
            return False

    def undo(self) -> bool:
        """Simulates Ctrl+Z."""
        try:
            keyboard.press_and_release('ctrl+z')
            return True
        except Exception as e:
            print(f"Error performing undo: {e}")
            return False

    def redo(self) -> bool:
        """Simulates Ctrl+Y."""
        try:
            keyboard.press_and_release('ctrl+y')
            return True
        except Exception as e:
            print(f"Error performing redo: {e}")
            return False

    def save(self) -> bool:
        """Simulates Ctrl+S."""
        try:
            keyboard.press_and_release('ctrl+s')
            return True
        except Exception as e:
            print(f"Error performing save: {e}")
            return False


if __name__ == "__main__":
    # Test Section
    executor = ActionExecutor()
    
    print("--- ActionExecutor Test Suite ---")
    
    # Test 1: App Launching
    print("Testing Calculator launch...")
    executor.execute("open_calculator")
    time.sleep(1)
    
    # Test 2: Volume Control
    print("Testing Volume Up...")
    executor.execute("volume_up")
    
    # Test 3: Screenshot
    print("Testing Screenshot...")
    executor.execute("take_screenshot")
    
    # Test 4: Invalid Action
    print("Testing Invalid Action...")
    executor.execute("shutdown_everything_now")
    
    print("\nTests completed successfully.")
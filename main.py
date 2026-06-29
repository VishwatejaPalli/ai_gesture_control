"""
main.py

Main application entry point for the gesture-controlled AI platform.
Coordinates real-time web camera capture, processing pipelines, system controls,
whiteboard painting, cursor navigation, volume mixing, and AI model routing.
"""

import os
import sys
import time
import logging
from collections import deque
import cv2
import numpy as np

# Ensure standard import structures are satisfied
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Safe import of optional cursor mapping libraries
try:
    import pyautogui
    # Configure safety limits for automated cursor actions
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
except ImportError:
    pyautogui = None
    logging.warning("pyautogui is not installed. System cursor controls will be deactivated.")

# Pipeline modular imports
from core.gesture_engine import GestureEngine
from media_control.media_controller import MediaController
from media_control.volume_controller import VolumeController
from gesture_shortcuts.macro_engine import MacroEngine


# =====================================================================
# Inline Auxiliary Application Controllers
# =====================================================================

class MouseController:
    """
    Handles cursor displacement mapping and clicking using PyAutoGUI.
    """
    def __init__(self, smoothing: int = 5):
        self.smoothing = smoothing
        self.prev_x, self.prev_y = 0, 0
        self.screen_w, self.screen_h = (1920, 1080)
        
        if pyautogui:
            try:
                self.screen_w, self.screen_h = pyautogui.size()
            except Exception:
                pass

    def move_to(self, rel_x: float, rel_y: float):
        """
        Moves the system cursor smoothly to normalized coordinate coordinates.
        """
        if not pyautogui:
            return
        try:
            # Map coordinates to monitor resolution bounds
            target_x = int(rel_x * self.screen_w)
            target_y = int(rel_y * self.screen_h)

            # Linear interpolation smoothing to reduce coordinate noise
            curr_x = self.prev_x + (target_x - self.prev_x) / self.smoothing
            curr_y = self.prev_y + (target_y - self.prev_y) / self.smoothing

            pyautogui.moveTo(int(curr_x), int(curr_y))
            self.prev_x, self.prev_y = curr_x, curr_y
        except Exception:
            pass

    def click(self, button: str = "left"):
        """Triggers standard mouse click events."""
        if not pyautogui:
            return
        try:
            pyautogui.click(button=button)
        except Exception:
            pass


class MouseGestures:
    """
    Translates tracked hand states to index finger cursor tracking and click triggers.
    """
    def __init__(self, controller: MouseController):
        self.controller = controller
        self.is_clicking = False

    def update(self, landmarks: list, accepted_gesture: str):
        """Moves cursor with index tip and registers left click on pinches."""
        if not landmarks or len(landmarks) < 9:
            return
            
        # Standard index tip landmark index (8)
        index_tip = landmarks[8]
        self.controller.move_to(index_tip.x, index_tip.y)

        # Trigger clicking if index and thumb are pinched
        if accepted_gesture == "pinch":
            if not self.is_clicking:
                self.controller.click()
                self.is_clicking = True
        else:
            self.is_clicking = False


class VirtualWhiteboard:
    """
    Implements a painting interface tracking continuous lines drawn with the index finger.
    """
    def __init__(self, frame_shape: tuple = (480, 640, 3)):
        self.frame_shape = frame_shape
        self.canvas = np.zeros(frame_shape, dtype=np.uint8)
        self.prev_point = None
        self.color = (0, 0, 255)  # Crimson red default paint brush
        self.thickness = 5

    def draw(self, landmarks: list, accepted_gesture: str):
        """Draws lines when pointing finger is moving."""
        if not landmarks or len(landmarks) < 9 or accepted_gesture != "pointing":
            self.prev_point = None
            return

        h, w, _ = self.frame_shape
        index_tip = landmarks[8]
        curr_point = (int(index_tip.x * w), int(index_tip.y * h))

        if self.prev_point is not None:
            cv2.line(self.canvas, self.prev_point, curr_point, self.color, self.thickness)

        self.prev_point = curr_point

    def clear(self):
        """Resets canvas elements."""
        self.canvas = np.zeros(self.frame_shape, dtype=np.uint8)

    def blend(self, frame: np.ndarray) -> np.ndarray:
        """Blends whiteboard strokes on top of camera frames."""
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        return cv2.add(bg, fg)


class AirWriter:
    """
    Draws short sliding queues of index coordinates to trace trails in the air.
    """
    def __init__(self, frame_shape: tuple = (480, 640, 3), max_points: int = 30):
        self.frame_shape = frame_shape
        self.points = deque(maxlen=max_points)
        self.color = (0, 255, 0)  # Neon green trail
        self.thickness = 6

    def update(self, landmarks: list, accepted_gesture: str):
        """Appends index coordinate positions during active writing states."""
        if not landmarks or len(landmarks) < 9 or accepted_gesture != "pointing":
            return

        h, w, _ = self.frame_shape
        index_tip = landmarks[8]
        curr_point = (int(index_tip.x * w), int(index_tip.y * h))
        self.points.append(curr_point)

    def clear(self):
        """Resets trace history queue."""
        self.points.clear()

    def draw_trail(self, frame: np.ndarray) -> np.ndarray:
        """Applies decaying traces directly onto the target frame matrix."""
        processed_frame = frame.copy()
        if len(self.points) < 2:
            return processed_frame

        for idx in range(1, len(self.points)):
            pt1 = self.points[idx - 1]
            pt2 = self.points[idx]

            # Dynamic opacity decay calculation
            alpha_ratio = idx / len(self.points)
            alpha = int(alpha_ratio * 255)
            color = (0, alpha, 0)
            thickness = int(alpha_ratio * self.thickness) + 1

            cv2.line(processed_frame, pt1, pt2, color, thickness)

        return processed_frame


# =====================================================================
# Main Application Environment
# =====================================================================

class GesturePlatform:
    """
    Central controller initializing frame capture, managing state routing,
    and handling UI presentation configurations.
    """
    def __init__(self):
        # Configure system modules
        self.gesture_engine = GestureEngine()
        self.media_controller = MediaController()
        self.volume_controller = VolumeController()
        self.macro_engine = MacroEngine()

        # Keyboard shortcuts mapping configuration
        self.mouse_controller = MouseController()
        self.mouse_gestures = MouseGestures(self.mouse_controller)
        
        # State modes definitions
        self.modes = {
            "1": "AIR_MOUSE",
            "2": "WHITEBOARD",
            "3": "AIR_WRITER",
            "4": "MEDIA_CONTROL",
            "5": "SHORTCUTS",
            "6": "AI_GESTURES"
        }
        self.current_mode = "AIR_MOUSE"

        # Frame parameters placeholder configuration
        self.frame_shape = (480, 640, 3)
        self.whiteboard = VirtualWhiteboard(self.frame_shape)
        self.air_writer = AirWriter(self.frame_shape)

        # Macro hotkey executor maps
        self.macro_actions = {
            "copy": lambda: pyautogui.hotkey("ctrl", "c") if pyautogui else None,
            "paste": lambda: pyautogui.hotkey("ctrl", "v") if pyautogui else None,
            "save": lambda: pyautogui.hotkey("ctrl", "s") if pyautogui else None,
            "undo": lambda: pyautogui.hotkey("ctrl", "z") if pyautogui else None,
            "open_browser": self._launch_browser,
            "stop": lambda: None
        }

    def _launch_browser(self):
        try:
            import webbrowser
            webbrowser.open("https://google.com")
        except Exception:
            pass

    def run(self):
        """
        Starts frame execution threads and coordinates the dispatch pipeline.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open the webcam.")
            return

        # Fetch frame dim configs
        ret, frame = cap.read()
        if ret:
            self.frame_shape = frame.shape
            self.whiteboard = VirtualWhiteboard(self.frame_shape)
            self.air_writer = AirWriter(self.frame_shape)

        prev_time = time.time()
        print("\n=== Gesture Platform Application Online ===")
        print("Use numerical keyboard keys [1 - 6] to change operating states.")
        print("Press 'Q' inside OpenCV display window to exit gracefully.\n")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Mirror view horizontally
            frame = cv2.flip(frame, 1)

            # 1. Process Frame coordinates using standard pipeline
            output = self.gesture_engine.process_frame(frame)
            processed_frame = output["frame"]
            raw_hands_data = output["hands"]
            raw_gesture = output["gesture"]
            accepted_gesture = output["accepted_gesture"]

            # Compute actual frames-per-second values
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time

            # Retrieve active hand coordinates reference if present
            primary_hand_lms = None
            if self.gesture_engine.tracker.hands and raw_hands_data:
                # MediaPipe internal tracking list reference
                results = self.gesture_engine.tracker.find_hands(frame)
                if results and results.multi_hand_landmarks:
                    primary_hand_lms = results.multi_hand_landmarks[0].landmark

            # 2. Dispatch operations depending on active application states
            self.dispatch_current_mode(primary_hand_lms, raw_gesture, accepted_gesture, processed_frame)

            # 3. Layer whiteboard / tracker paintings onto viewport frame
            if self.current_mode == "WHITEBOARD":
                processed_frame = self.whiteboard.blend(processed_frame)
            elif self.current_mode == "AIR_WRITER":
                processed_frame = self.air_writer.draw_trail(processed_frame)

            # 4. Draw HUD controls overlay on final processed frame output
            confidence = 1.0
            if raw_hands_data:
                confidence = raw_hands_data[0].get("confidence", 1.0)

            self.draw_hud(
                processed_frame, 
                fps, 
                self.current_mode, 
                raw_gesture, 
                accepted_gesture, 
                confidence, 
                len(raw_hands_data)
            )

            # 5. Show image
            cv2.imshow("Gesture AI Platform", processed_frame)

            # 6. Read Keyboard Polling Events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif chr(key) in self.modes:
                target_mode = self.modes[chr(key)]
                self.switch_mode(target_mode)
            elif key == ord('c'):
                # Clear canvas elements
                self.whiteboard.clear()
                self.air_writer.clear()
            elif key == ord('s'):
                # Save processed snapshot
                filepath = f"snapshot_{int(time.time())}.png"
                cv2.imwrite(filepath, processed_frame)
                print(f"Captured screen saved to: {filepath}")

        # Gracefully shutdown resources
        cap.release()
        cv2.destroyAllWindows()
        print("\nPlatform application closed gracefully.")

    def switch_mode(self, mode_name: str):
        """Handles state transition configurations."""
        if self.current_mode == mode_name:
            return

        print(f"Switching state: {self.current_mode} -> {mode_name}")
        self.current_mode = mode_name
        self.gesture_engine.reset()

        # Handle AI loading operations on demand
        if mode_name == "AI_GESTURES":
            model_path = "best_model.pth"
            if os.path.exists(model_path):
                self.gesture_engine.enable_ai_model(model_path)
            else:
                print("Notice: 'best_model.pth' weight file not found. AI operations will fallback to heuristic checks.")
        else:
            self.gesture_engine.disable_ai_model()

    def dispatch_current_mode(self, landmarks: list, raw_gesture: str, accepted_gesture: str, frame: np.ndarray):
        """
        Routes hand coordinates, gestures, and features to mode interfaces.
        """
        if not landmarks:
            return

        if self.current_mode == "AIR_MOUSE":
            self.mouse_gestures.update(landmarks, accepted_gesture)

        elif self.current_mode == "WHITEBOARD":
            self.whiteboard.draw(landmarks, accepted_gesture)

        elif self.current_mode == "AIR_WRITER":
            self.air_writer.update(landmarks, accepted_gesture)

        elif self.current_mode == "MEDIA_CONTROL":
            # Pass absolute landmark elements to volume mixer
            self.volume_controller.process(landmarks)
            
            # Dispatch discrete state signals to media controller
            if accepted_gesture == "palm":
                self.media_controller.execute("play_pause")
            elif accepted_gesture == "fist":
                self.media_controller.execute("mute")
            elif accepted_gesture == "peace":
                self.media_controller.execute("fullscreen")

        elif self.current_mode == "SHORTCUTS":
            # Query mapped macros configurations
            action_key = self.macro_engine.execute(accepted_gesture)
            if action_key and action_key in self.macro_actions:
                print(f"Executing shortcut action: {action_key}")
                try:
                    self.macro_actions[action_key]()
                except Exception as e:
                    logging.debug(f"Shortcut executor failure: {e}")

        elif self.current_mode == "AI_GESTURES":
            # AI outputs run inside process_frame; handled inside HUD drawing routines
            pass

    def draw_hud(self, frame: np.ndarray, fps: float, current_mode: str, gesture: str, accepted_gesture: str, confidence: float, num_hands: int):
        """Renders information overlay bars and HUD trackers on-screen."""
        h, w, _ = frame.shape

        # Draw translucent dashboard background panel
        panel = frame.copy()
        cv2.rectangle(panel, (0, 0), (w, 100), (35, 35, 35), -1)
        cv2.addWeighted(panel, 0.55, frame, 0.45, 0, frame)

        # Print basic status variables
        cv2.putText(frame, f"MODE: {current_mode}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Raw: {gesture}", (260, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Filtered: {accepted_gesture}", (260, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        ai_text = f"AI Conf: {confidence:.2%}" if self.gesture_engine.ai_enabled and num_hands > 0 else "AI Conf: N/A"
        cv2.putText(frame, ai_text, (480, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Hands: {num_hands}", (480, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Print dynamic visual alerts if weights files are not found in AI mode
        if self.current_mode == "AI_GESTURES" and not self.gesture_engine.ai_enabled:
            cv2.rectangle(frame, (10, 110), (w - 10, 140), (0, 0, 120), -1)
            cv2.putText(frame, "NOTICE: best_model.pth missing! Run ai/train_model.py first to enable AI predictions.", 
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Print controls menu inside bottom bar
        cv2.rectangle(frame, (0, h - 35), (w, h), (25, 25, 25), -1)
        cv2.putText(frame, "1:Mouse | 2:Whiteboard | 3:Writer | 4:Media | 5:Macros | 6:AI | C:Clear | S:Save | Q:Quit",
                    (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


if __name__ == "__main__":
    platform = GesturePlatform()
    try:
        platform.run()
    except KeyboardInterrupt:
        print("\nApplication closed via keyboard interrupt.")
    except Exception as error:
        print(f"\nFatal error detected during execution: {error}")
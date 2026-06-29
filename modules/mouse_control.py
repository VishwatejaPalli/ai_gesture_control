import sys
import os
import time
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
import pyautogui

# Adjust system path for local modules
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ai.model import GestureNet

# Optimize PyAutoGUI responsiveness
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True  

class GestureControlHub:
    def __init__(self):
        # Configuration & Constants
        self.modes = ["INFERENCE", "PAINT", "MOUSE CONTROL"]
        self.current_mode_idx = 0
        
        # Gesture Mapping
        self.MODE_SWITCH_GESTURE = "palm"
        self.PAINT_CURSOR_GESTURE = "pointing"
        self.PAINT_CLEAR_GESTURE = "peace"
        self.MOUSE_CURSOR_GESTURE = "pointing"
        self.MOUSE_CLICK_GESTURE = "fist"
        
        # UI & Drawing Settings
        self.action_cooldown = 1.0
        self.brush_color = (0, 255, 255)
        self.brush_thickness = 7
        self.cursor_color = (0, 255, 0)
        self.classes = ["fist", "palm", "peace", "pinch", "pointing", "thumbs_up"]
        
        # Smoothing & Screen Mapping Settings
        self.smoothing = 0.25  
        self.screen_margin = 80  
        
        # Dynamic State Variables
        self.canvas = None
        self.prev_x, self.prev_y = 0, 0
        self.smooth_x, self.smooth_y = 0, 0
        self.last_action_time = 0
        
        # Button UI Geometry [{ "mode_idx": idx, "x1": x1, "y1": y1, "x2": x2, "y2": y2 }]
        self.buttons = []
        self.hover_target = None
        self.hover_start_time = 0
        self.hover_delay = 0.4  # Seconds to hover over a button to trigger it
        
        # Initialization methods
        self._init_hardware()
        self._init_ai_models()

    def _init_hardware(self):
        print("Initializing Hardware Dynamics...")
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            print("Camera index 1 failed. Falling back to default camera (0)...")
            self.cap = cv2.VideoCapture(0)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.screen_w, self.screen_h = pyautogui.size()

    def _init_ai_models(self):
        print("Loading AI Inference Modules...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GestureNet(num_classes=len(self.classes)).to(self.device)
        
        try:
            model_path = "models/best_model.pth"
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Model successfully anchored to target: {self.device}")
        except Exception as e:
            print(f"Critical Error loading model: {e}")
            
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7
        )

    def draw_ui(self, frame, gesture, confidence, idx_x=None, idx_y=None):
        h, w, _ = frame.shape
        current_mode = self.modes[self.current_mode_idx]
        current_time = time.time()
        
        # 1. Header background bar
        cv2.rectangle(frame, (0, 0), (w, 75), (20, 20, 20), -1)
        
        # 2. Build / Draw UI Buttons dynamically based on frame width
        self.buttons = []
        btn_w = 220
        btn_h = 45
        spacing = 20
        start_x = 20
        y1 = 15
        y2 = y1 + btn_h
        
        new_hover_target = None

        for i, mode_name in enumerate(self.modes):
            x1 = start_x + i * (btn_w + spacing)
            x2 = x1 + btn_w
            self.buttons.append({"mode_idx": i, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            
            # Check if finger cursor is hovering over this button
            is_hovered = False
            if idx_x is not None and idx_y is not None:
                if x1 <= idx_x <= x2 and y1 <= idx_y <= y2:
                    is_hovered = True
                    new_hover_target = i
            
            # Determine color theme based on state
            if i == self.current_mode_idx:
                btn_color = (0, 180, 0)  # Active Active Mode (Green)
                text_color = (255, 255, 255)
            elif is_hovered:
                btn_color = (120, 120, 255)  # Hover state (Light Red/Orange)
                text_color = (0, 0, 0)
            else:
                btn_color = (60, 60, 60)  # Inactive Default (Gray)
                text_color = (200, 200, 200)
                
            # Render Button rect and label text
            cv2.rectangle(frame, (x1, y1), (x2, y2), btn_color, -1, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1, cv2.LINE_AA)
            
            # Text layout centering adjustment
            cv2.putText(frame, mode_name, (x1 + 15, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA)

        # 3. Handle Hover Timers for Mode Switching Action
        if new_hover_target is not None and new_hover_target != self.current_mode_idx:
            if self.hover_target != new_hover_target:
                self.hover_target = new_hover_target
                self.hover_start_time = current_time
            elif current_time - self.hover_start_time >= self.hover_delay:
                # Trigger Mode Change
                self.current_mode_idx = new_hover_target
                self.prev_x, self.prev_y = 0, 0
                self.hover_target = None
                print(f"Switched operation via Button to: {self.modes[self.current_mode_idx]}")
            else:
                # Draw a tiny visual progress bar/circle overlay for feedback
                progress = (current_time - self.hover_start_time) / self.hover_delay
                cv2.circle(frame, (idx_x, idx_y + 25), int(5 + progress * 10), (0, 255, 255), 2)
        else:
            if idx_x is not None:
                self.hover_target = None

        # 4. Right side text inference stats
        if gesture:
            cv2.putText(frame, f"AI: {gesture} ({confidence:.2f})", (w - 320, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 5. Dynamic Context Footer Instructions
        if current_mode == "INFERENCE":
            info = f"Show '{self.MODE_SWITCH_GESTURE}' or Hover on Top Buttons to cycle operational modes."
        elif current_mode == "PAINT":
            info = f"'{self.PAINT_CURSOR_GESTURE}'=Hover | Pinch Finger tips=Draw | '{self.PAINT_CLEAR_GESTURE}'=Clear | Buttons=Switch Mode"
        elif current_mode == "MOUSE CONTROL":
            info = f"'{self.MOUSE_CURSOR_GESTURE}'=Smooth Move | '{self.MOUSE_CLICK_GESTURE}'=Trigger Click"
            
        cv2.rectangle(frame, (0, h - 45), (w, h), (15, 15, 15), -1)
        cv2.putText(frame, info, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def process_mouse_movement(self, idx_x, idx_y, w, h):
        x_mapped = np.interp(idx_x, (self.screen_margin, w - self.screen_margin), (0, self.screen_w))
        y_mapped = np.interp(idx_y, (self.screen_margin, h - self.screen_margin), (0, self.screen_h))
        
        self.smooth_x = self.smooth_x + (x_mapped - self.smooth_x) * self.smoothing
        self.smooth_y = self.smooth_y + (y_mapped - self.smooth_y) * self.smoothing
        
        target_x = max(0, min(self.screen_w - 1, int(self.smooth_x)))
        target_y = max(0, min(self.screen_h - 1, int(self.smooth_y)))
        
        pyautogui.moveTo(target_x, target_y)

    def run(self):
        print("\nControl Hub Live. Buttons Active. Press 'q' inside window to exit.")
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape

            if self.canvas is None:
                self.canvas = np.zeros((h, w, c), dtype=np.uint8)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            current_gesture, confidence = None, 0.0
            current_time = time.time()
            idx_x, idx_y = None, None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Package features for GestureNet
                    flat_data = []
                    for lm in hand_landmarks.landmark:
                        flat_data.extend([lm.x, lm.y, lm.z])

                    x_tensor = torch.tensor([flat_data], dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        output = self.model(x_tensor)
                        probs = F.softmax(output, dim=1)
                        confidence = probs.max().item()
                        current_gesture = self.classes[output.argmax(dim=1).item()]

                    # Extract Finger tracking index coordinates
                    index_tip = hand_landmarks.landmark[8]
                    idx_x, idx_y = int(index_tip.x * w), int(index_tip.y * h)
                    
                    # Gesture Mode Rotator Logic (Kept as fallback variant)
                    if current_gesture == self.MODE_SWITCH_GESTURE and (current_time - self.last_action_time > self.action_cooldown):
                        self.current_mode_idx = (self.current_mode_idx + 1) % len(self.modes)
                        self.last_action_time = current_time
                        self.prev_x, self.prev_y = 0, 0
                        print(f"Switched operation to: {self.modes[self.current_mode_idx]}")

                    mode = self.modes[self.current_mode_idx]
                    
                    if mode == "PAINT":
                        thumb_tip = hand_landmarks.landmark[4]
                        thmb_x, thmb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                        draw_distance = np.hypot(idx_x - thmb_x, idx_y - thmb_y)

                        # Draw only if we are below the top menu bar boundary space
                        if idx_y > 75 and draw_distance < 35:  
                            cv2.circle(frame, (idx_x, idx_y), 10, self.brush_color, cv2.FILLED)
                            if self.prev_x == 0 and self.prev_y == 0:
                                self.prev_x, self.prev_y = idx_x, idx_y
                            cv2.line(self.canvas, (self.prev_x, self.prev_y), (idx_x, idx_y), self.brush_color, self.brush_thickness)
                            self.prev_x, self.prev_y = idx_x, idx_y
                        elif current_gesture == self.PAINT_CLEAR_GESTURE and (current_time - self.last_action_time > self.action_cooldown):
                            self.canvas = np.zeros((h, w, c), dtype=np.uint8)
                            self.last_action_time = current_time
                        else:
                            cv2.circle(frame, (idx_x, idx_y), 12, self.cursor_color, 2)
                            self.prev_x, self.prev_y = 0, 0

                    elif mode == "MOUSE CONTROL":
                        if current_gesture == self.MOUSE_CURSOR_GESTURE:
                            self.process_mouse_movement(idx_x, idx_y, w, h)
                            cv2.circle(frame, (idx_x, idx_y), 12, self.cursor_color, 2)
                        elif current_gesture == self.MOUSE_CLICK_GESTURE and (current_time - self.last_action_time > self.action_cooldown):
                            pyautogui.click()
                            self.last_action_time = current_time
                            cv2.circle(frame, (idx_x, idx_y), 20, (0, 0, 255), cv2.FILLED)

            # Composite canvas overlay rendering
            img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, img_inv = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)
            img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, img_inv)
            frame = cv2.bitwise_or(frame, self.canvas)

            # Display Runtime UI (passing index finger positions for button collision checks)
            frame = self.draw_ui(frame, current_gesture, confidence, idx_x, idx_y)
            cv2.imshow("Gesture AI Control Hub", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("System shutdown cleanly.")

if __name__ == "__main__":
    hub = GestureControlHub()
    hub.run()
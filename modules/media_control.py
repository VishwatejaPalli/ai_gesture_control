import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
import time
import numpy as np
import os
import sys
import logging

# Setup basic runtime logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 1. Dynamic Root Path Resolution
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 2. Media Control Path
MEDIA_CONTROL_PATH = os.path.join(ROOT, "media_control")
if MEDIA_CONTROL_PATH not in sys.path:
    sys.path.insert(0, MEDIA_CONTROL_PATH)

# 3. Dynamic Import Block
try:
    from ai.model import GestureNet
    print("Resolved: GestureNet from 'ai.model'")
except ImportError:
    try:
        from model import GestureNet
        print("Resolved: GestureNet from 'models.model'")
    except ImportError:
        try:
            from models import GestureNet
            print("Resolved: GestureNet from 'models'")
        except ImportError:
            GestureNet = None
            print("Warning: GestureNet could not be resolved. Fallback mode.")

# 4. Media Controller Import
try:
    from media_controller import MediaController
except ImportError:
    print("Error: media_controller.py not found in sys.path.")
    exit()

# --- Constants and Configuration ---
GESTURE_MAP = {
    "palm": "play_pause",
    "thumbs_up": "play_pause",
    "pointing": "volume",
    "fist": "seek",
    "peace": "brightness"
}

ACTION_COOLDOWN = 0.8  
SEEK_COOLDOWN = 0.6
VOLUME_COOLDOWN = 0.15
BRIGHTNESS_COOLDOWN = 0.2

VERTICAL_MOVE_THRESHOLD = 0.03
HORIZONTAL_MOVE_THRESHOLD = 0.04

# --- Initialization ---
sys_controller = MediaController()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["fist", "palm", "peace", "pinch", "pointing", "thumbs_up"]

if GestureNet is not None:
    model = GestureNet(num_classes=len(classes)).to(device)
    try:
        model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
        model.eval()
        print(f"Model loaded successfully on: {device}")
    except FileNotFoundError:
        print("!!! FATAL: 'models/best_model.pth' not found. !!!")
        exit()
else:
    model = None

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(1)

# State Variables
last_action_time = 0
last_feedback_msg = ""
last_feedback_time = 0
prev_pos = None 

def show_feedback(message):
    global last_feedback_msg, last_feedback_time
    last_feedback_msg = message
    last_feedback_time = time.time()

def draw_ui(frame, gesture):
    h, w, _ = frame.shape
    if gesture:
        cv2.putText(frame, f"Gesture: {gesture}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if time.time() - last_feedback_time < 1.2:
        (text_w, text_h), _ = cv2.getTextSize(last_feedback_msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cv2.putText(frame, last_feedback_msg, (int((w - text_w) / 2), int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
        cv2.putText(frame, last_feedback_msg, (int((w - text_w) / 2), int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return frame

print("\nMedia Controller active. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_gesture = None

    if results.multi_hand_landmarks and model is not None:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmarks
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],dtype=np.float32)

            # translate
            coords = coords - coords[0]

            # scale
            scale = np.linalg.norm(coords[9])

            if scale < 1e-6:
                scale = 1.0

            coords = coords / scale

            flat_data = coords.flatten()

            x = torch.tensor(
                [flat_data],
                dtype=torch.float32
            ).to(device)

            # FIXED: Indentation for inference
            with torch.no_grad():
                output = model(x)
                probs = F.softmax(output, dim=1)
                prediction = output.argmax(dim=1).item()
                confidence = float(probs[0][prediction])
                current_gesture = classes[prediction]

            # Only trigger action if confidence is high (e.g., > 70%)
            if confidence > 0.70:
                action = GESTURE_MAP.get(current_gesture)
                current_time = time.time()

                # --- Action Logic ---
                if action == "play_pause" and current_time - last_action_time > ACTION_COOLDOWN:
                    sys_controller.execute("play_pause")
                    show_feedback("Play/Pause")
                    last_action_time = current_time
                    prev_pos = None

                elif action == "volume":
                    # Index finger tip
                    pos = (hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y)
                    if prev_pos and current_time - last_action_time > VOLUME_COOLDOWN:
                        dy = prev_pos[1] - pos[1] # Up is negative in screen coords
                        if dy > VERTICAL_MOVE_THRESHOLD: 
                            sys_controller.execute("volume_up") # Using standard execute if available
                            show_feedback("Volume Up")
                            last_action_time = current_time
                        elif dy < -VERTICAL_MOVE_THRESHOLD: 
                            sys_controller.execute("volume_down")
                            show_feedback("Volume Down")
                            last_action_time = current_time
                    prev_pos = pos

                elif action == "seek":
                    # Palm base
                    pos = (hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y)
                    if prev_pos and current_time - last_action_time > SEEK_COOLDOWN:
                        dx = pos[0] - prev_pos[0]
                        if dx > HORIZONTAL_MOVE_THRESHOLD: 
                            sys_controller.execute("next_track")
                            show_feedback("Next Track >>")
                            last_action_time = current_time
                        elif dx < -HORIZONTAL_MOVE_THRESHOLD: 
                            sys_controller.execute("previous_track")
                            show_feedback("<< Prev Track")
                            last_action_time = current_time
                    prev_pos = pos

                elif action == "brightness":
                    pos = (hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y)
                    if prev_pos and current_time - last_action_time > BRIGHTNESS_COOLDOWN:
                        dy = prev_pos[1] - pos[1]
                        if dy > VERTICAL_MOVE_THRESHOLD:
                            sys_controller.execute("brightness_up")
                            show_feedback("Brightness Up")
                            last_action_time = current_time
                        elif dy < -VERTICAL_MOVE_THRESHOLD:
                            sys_controller.execute("brightness_down")
                            show_feedback("Brightness Down")
                            last_action_time = current_time
                    prev_pos = pos
            else:
                current_gesture = None # Reset if confidence is too low
    else:
        prev_pos = None

    frame = draw_ui(frame, current_gesture)
    cv2.imshow("Gesture Media Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
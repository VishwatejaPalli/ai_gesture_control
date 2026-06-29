import os
import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Drawing State Variables
canvas = None
prev_x, prev_y = 0, 0
brush_color = (0, 255, 255) # Bright neon yellow/cyan
brush_thickness = 5

print("AR Painting Tool active.")
print("Index Finger UP = Move Pointer")
print("Pinch Index + Thumb = DRAW")
print("Open Hand (5 fingers) = CLEAR CANVAS")
print("Index + Middle Together = TAKE AR SCREENSHOT")
print("Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Initialize canvas overlay on the first frame run
    if canvas is None:
        canvas = np.zeros((h, w, c), dtype=np.uint8)

    # Convert to RGB for AI model processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            # 1. Get Coordinates of Key Fingertips
            thumb = landmarks[4]
            index = landmarks[8]
            middle = landmarks[12]
            ring = landmarks[16]
            pinky = landmarks[20]

            # Convert normalized landmarks to pixel dimensions
            idx_x, idx_y = int(index.x * w), int(index.y * h)
            thmb_x, thmb_y = int(thumb.x * w), int(thumb.y * h)
            mid_x, mid_y = int(middle.x * w), int(middle.y * h)

            # Calculate distances for gesture logic
            draw_distance = np.hypot(idx_x - thmb_x, idx_y - thmb_y)
            snip_distance = np.hypot(idx_x - mid_x, idx_y - mid_y)

            # 2. GESTURE DETECTION

            # --- A. open hand check to CLEAR canvas (if fingertips are higher than joints)
            if index.y < landmarks[6].y and middle.y < landmarks[10].y and ring.y < landmarks[14].y and pinky.y < landmarks[18].y and draw_distance > 60:
                canvas = np.zeros((h, w, c), dtype=np.uint8)
                cv2.putText(frame, "Canvas Cleared", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                prev_x, prev_y = 0, 0

            # --- B. DRAW ACTION (Pinch Index & Thumb)
            elif draw_distance < 35:
                cv2.circle(frame, (idx_x, idx_y), 10, brush_color, cv2.FILLED)
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = idx_x, idx_y

                # Draw a persistent line on our dedicated canvas layer
                cv2.line(canvas, (prev_x, prev_y), (idx_x, idx_y), brush_color, brush_thickness)
                prev_x, prev_y = idx_x, idx_y

            # --- C. AR SNIP/SCREENSHOT ACTION (Pinch Index & Middle)
            elif snip_distance < 30:
                # Merge current camera view with our drawn canvas lines
                img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
                img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
                final_blend = cv2.bitwise_and(frame, img_inv)
                final_blend = cv2.bitwise_or(final_blend, canvas)

                # Save the screen/canvas combination
                os.makedirs("result/paint", exist_ok=True)
                ts = int(time.time())
                filename = f"result/paint/ar_snip_{ts}.png"
                cv2.imwrite(filename, final_blend)
                print(f"Saved AR Snip: {filename}")
                
                # Visual Flash Feedback
                cv2.rectangle(frame, (0,0), (w,h), (255,255,255), cv2.FILLED)
                cv2.imshow("AR Snip Paint", frame)
                cv2.waitKey(100)
                time.sleep(0.4)

            # --- D. HOVERING (Moving finger around without drawing)
            else:
                # Draw a temporary tracking cursor on the live video frame (not saved to canvas)
                cv2.circle(frame, (idx_x, idx_y), 8, (0, 255, 0), cv2.FILLED)
                prev_x, prev_y = 0, 0 # Reset drawing path continuity

    # 3. MERGE LIVE CAMERA WITH DIGITAL CANVAS
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, img_inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("AR Snip Paint", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Screen size
screen_w, screen_h = pyautogui.size()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Change camera index if needed
cap = cv2.VideoCapture(0)

# Disable PyAutoGUI failsafe
pyautogui.FAILSAFE = False

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        # Index fingertip
        index_tip = hand.landmark[8]

        # Thumb tip
        thumb_tip = hand.landmark[4]

        # Middle fingertip
        middle_tip = hand.landmark[12]

        # Convert to pixel coordinates
        ix = int(index_tip.x * w)
        iy = int(index_tip.y * h)

        tx = int(thumb_tip.x * w)
        ty = int(thumb_tip.y * h)

        mx = int(middle_tip.x * w)
        my = int(middle_tip.y * h)

        # Map camera coordinates to screen
        mouse_x = np.interp(ix, [0, w], [0, screen_w])
        mouse_y = np.interp(iy, [0, h], [0, screen_h])

        # Move mouse
        pyautogui.moveTo(mouse_x, mouse_y)

        # Distance between thumb and index
        click_dist = ((ix - tx) ** 2 + (iy - ty) ** 2) ** 0.5

        # Distance between index and middle
        right_dist = ((ix - mx) ** 2 + (iy - my) ** 2) ** 0.5

        # Left click
        if click_dist < 40:
            pyautogui.click()
            cv2.putText(
                frame,
                "LEFT CLICK",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # Right click
        if right_dist < 30:
            pyautogui.rightClick()
            cv2.putText(
                frame,
                "RIGHT CLICK",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

    cv2.imshow("Gesture Mouse", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
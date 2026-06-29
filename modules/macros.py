import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Laptop webcam
cap = cv2.VideoCapture(0)

canvas = None
prev_point = None
drawing = False

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        h, w, _ = frame.shape
        canvas = np.zeros((h, w), dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        index_tip = hand.landmark[8]
        thumb_tip = hand.landmark[4]

        ix = int(index_tip.x * w)
        iy = int(index_tip.y * h)

        tx = int(thumb_tip.x * w)
        ty = int(thumb_tip.y * h)

        # pinch distance
        dist = ((ix - tx)**2 + (iy - ty)**2)**0.5

        # pinch -> draw
        if dist < 40:

            drawing = True

            if prev_point is not None:
                cv2.line(
                    canvas,
                    prev_point,
                    (ix, iy),
                    255,
                    8
                )

            prev_point = (ix, iy)

        else:

            drawing = False
            prev_point = None

        cv2.circle(
            frame,
            (ix, iy),
            10,
            (0, 255, 0),
            -1
        )

    overlay = cv2.cvtColor(
        canvas,
        cv2.COLOR_GRAY2BGR
    )

    output = cv2.addWeighted(
        frame,
        0.7,
        overlay,
        0.3,
        0
    )

    cv2.putText(
        output,
        "Pinch thumb+index to draw",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Vision Style Air Writing",
        output
    )

    key = cv2.waitKey(1)

    # ESC
    if key == 27:
        break

    # clear
    elif key == ord('c'):
        canvas[:] = 0

    # save
    elif key == ord('s'):
        filename = f"char_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        print("Saved:", filename)

cap.release()
cv2.destroyAllWindows()
import cv2
import sys
import os

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.hand_tracker import HandTracker


tracker = HandTracker()

cap = cv2.VideoCapture(1)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    frame, hands = tracker.process(frame)

    frame = tracker.draw_fps(frame)

    for hand in hands:
        print(
            hand["hand"],
            len(hand["landmarks"])
        )

    cv2.imshow("Hand Tracker", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
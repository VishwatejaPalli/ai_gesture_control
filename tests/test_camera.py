import cv2
from core.hand_tracker import HandTracker

tracker = HandTracker()

cap = cv2.VideoCapture(0)

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
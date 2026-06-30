import sys
import os

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.hand_tracker import HandTracker
import cv2

from core.landmark_processor import LandmarkProcessor


def main():

    tracker = HandTracker()
    processor = LandmarkProcessor()

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Mirror effect
        frame = cv2.flip(frame, 1)

        # Detect hands
        frame, hands = tracker.process(frame)

        # Draw FPS
        frame = tracker.draw_fps(frame)

        # Process every detected hand
        for hand in hands:

            landmarks = hand["landmarks"]

            features = processor.extract_features(
                landmarks
            )

            # Print information to terminal
            print(
                f'{hand["hand"]}:',
                features["fingers"],
                'pinch:',
                features["pinch"]
            )

            # Draw landmark IDs
            for i, lm in enumerate(landmarks):

                cv2.putText(
                    frame,
                    str(i),
                    (lm["x"], lm["y"]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1
                )

            # Display finger states
            finger_text = (
                f'T:{int(features["fingers"]["thumb"])} '
                f'I:{int(features["fingers"]["index"])} '
                f'M:{int(features["fingers"]["middle"])} '
                f'R:{int(features["fingers"]["ring"])} '
                f'P:{int(features["fingers"]["pinky"])}'
            )

            cv2.putText(
                frame,
                finger_text,
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            # Display pinch status
            cv2.putText(
                frame,
                f'Pinch: {features["pinch"]}',
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        cv2.imshow("Gesture AI - Hand Tracker", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
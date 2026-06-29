import cv2

from core.hand_tracker import HandTracker
from core.landmark_processor import LandmarkProcessor
from core.state_machine import GestureStateMachine
from gesture_classifier.classifier import GestureClassifier


def main():

    # Initialize modules
    tracker = HandTracker()
    processor = LandmarkProcessor()
    classifier = GestureClassifier()
    state_machine = GestureStateMachine()

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Press 'q' to quit")

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

        # Process each detected hand
        for hand in hands:

            landmarks = hand["landmarks"]

            # Extract features
            features = processor.extract_features(
                landmarks
            )

            # Classify gesture
            gesture = classifier.classify(
                features["fingers"],
                features["pinch"]
            )

            # State machine filtering
            accepted = state_machine.update(
                gesture
            )

            #################################################
            # Terminal output
            #################################################

            print(
                f'Hand: {hand["hand"]} | '
                f'Gesture: {gesture} | '
                f'Pinch: {features["pinch"]}'
            )

            if accepted:
                print(
                    f'>>> ACCEPTED: {accepted}'
                )

            #################################################
            # Draw landmark IDs
            #################################################

            for idx, lm in enumerate(landmarks):

                cv2.putText(
                    frame,
                    str(idx),
                    (lm["x"], lm["y"]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 0),
                    1
                )

            #################################################
            # Display finger states
            #################################################

            fingers = features["fingers"]

            finger_text = (
                f"T:{int(fingers['thumb'])} "
                f"I:{int(fingers['index'])} "
                f"M:{int(fingers['middle'])} "
                f"R:{int(fingers['ring'])} "
                f"P:{int(fingers['pinky'])}"
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

            #################################################
            # Gesture display
            #################################################

            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            #################################################
            # Pinch display
            #################################################

            cv2.putText(
                frame,
                f"Pinch: {features['pinch']}",
                (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            #################################################
            # Accepted gesture display
            #################################################

            if accepted:

                cv2.putText(
                    frame,
                    f"ACCEPTED: {accepted}",
                    (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3
                )

        # Display frame
        cv2.imshow(
            "Gesture AI System",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
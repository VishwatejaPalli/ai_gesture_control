# core/hand_tracker.py

import cv2
import mediapipe as mp
import time


class HandTracker:
    def __init__(
        self,
        max_hands=2,
        detection_confidence=0.7,
        tracking_confidence=0.7,
    ):

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        self.prev_time = 0

    def process(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        hand_data = []

        if results.multi_hand_landmarks:

            for idx, hand_landmarks in enumerate(
                results.multi_hand_landmarks
            ):

                h, w, _ = frame.shape
                landmarks = []

                for lm in hand_landmarks.landmark:

                    landmarks.append({
                        "x": int(lm.x * w),
                        "y": int(lm.y * h),
                        "z": lm.z
                    })

                hand_type = "Unknown"

                if results.multi_handedness:
                    hand_type = (
                        results
                        .multi_handedness[idx]
                        .classification[0]
                        .label
                    )

                hand_data.append({
                    "hand": hand_type,
                    "landmarks": landmarks
                })

                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

        return frame, hand_data

    def draw_fps(self, frame):

        current = time.time()
        fps = 1 / (current - self.prev_time + 1e-6)
        self.prev_time = current

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        return frame
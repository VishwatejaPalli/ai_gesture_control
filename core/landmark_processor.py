# core/landmark_processor.py

import math
import numpy as np


class LandmarkProcessor:

    def __init__(self):
        pass

    def distance(self, p1, p2):
        return math.sqrt(
            (p1["x"] - p2["x"])**2 +
            (p1["y"] - p2["y"])**2
        )

    def normalize(self, landmarks):

        wrist = landmarks[0]

        normalized = []

        for lm in landmarks:
            normalized.append([
                lm["x"] - wrist["x"],
                lm["y"] - wrist["y"],
                lm["z"]
            ])

        scale = max(
            np.linalg.norm(normalized[9][:2]),
            1
        )

        normalized = [
            [
                x/scale,
                y/scale,
                z
            ]
            for x, y, z in normalized
        ]

        return normalized

    def get_finger_states(self, landmarks):

        fingers = {}

        fingers["thumb"] = (
            landmarks[4]["x"] >
            landmarks[3]["x"]
        )

        fingers["index"] = (
            landmarks[8]["y"] <
            landmarks[6]["y"]
        )

        fingers["middle"] = (
            landmarks[12]["y"] <
            landmarks[10]["y"]
        )

        fingers["ring"] = (
            landmarks[16]["y"] <
            landmarks[14]["y"]
        )

        fingers["pinky"] = (
            landmarks[20]["y"] <
            landmarks[18]["y"]
        )

        return fingers

    def is_pinch(self,
                 landmarks,
                 threshold=40):

        d = self.distance(
            landmarks[4],
            landmarks[8]
        )

        return d < threshold

    def extract_features(self, landmarks):

        features = {}

        features["normalized"] = \
            self.normalize(landmarks)

        features["fingers"] = \
            self.get_finger_states(landmarks)

        features["pinch"] = \
            self.is_pinch(landmarks)

        features["thumb_index"] = \
            self.distance(
                landmarks[4],
                landmarks[8]
            )

        features["index_middle"] = \
            self.distance(
                landmarks[8],
                landmarks[12]
            )

        return features
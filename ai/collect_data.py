"""
ai/collect_data.py

Implements a real-time gesture dataset collection tool. It captures hand landmarks 
via a webcam using MediaPipe and OpenCV, applies translation and scale normalization, 
and persists the samples into a structured CSV file for model training.
"""

import os
import csv
import time
import numpy as np
import cv2
import mediapipe as mp


class GestureDataCollector:
    """
    A class to orchestrate hand gesture data collection from web camera streams.
    Saves scale and translation-invariant 3D coordinates.
    """

    def __init__(self, dataset_path="dataset", samples_per_class=500):
        """
        Initializes the collector settings, mappings, and progress states.

        Args:
            dataset_path (str): Directory where the output CSV will be saved.
            samples_per_class (int): Target number of samples to collect for each class.
        """
        self.dataset_path = dataset_path
        self.samples_per_class = samples_per_class
        self.csv_file = os.path.join(self.dataset_path, "gesture_data.csv")

        # Map keyboard hotkeys to respective gesture categories
        self.gesture_keys = {
            ord('f'): "fist",
            ord('p'): "palm",
            ord('e'): "peace",
            ord('i'): "pointing",
            ord('t'): "thumbs_up",
            ord('n'): "pinch"
        }

        # Track collection counts per category
        self.counts = {name: 0 for name in self.gesture_keys.values()}

        self.create_dataset_structure()
        self._load_existing_counts()

    def create_dataset_structure(self):
        """
        Creates the output folder and initializes the dataset CSV file with 
        headers if not already present.
        """
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                # Header format: label, x0, y0, z0, ..., x20, y20, z20
                headers = ["label"]
                for i in range(21):
                    headers.extend([f"x{i}", f"y{i}", f"z{i}"])
                writer.writerow(headers)

    def _load_existing_counts(self):
        """
        Scans the existing CSV file to recover historical counts 
        to enable resuming collections between sessions.
        """
        if os.path.exists(self.csv_file):
            try:
                with open(self.csv_file, mode='r') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip CSV header
                    for row in reader:
                        if row:
                            label = row[0]
                            if label in self.counts:
                                self.counts[label] += 1
                print("Dataset scanning complete. Current counts loaded:")
                for name, count in self.counts.items():
                    print(f"  {name}: {count} samples")
            except Exception as e:
                print(f"Could not load existing record counts: {e}")

    def normalize_landmarks(self, landmarks):
        """
        Transforms coordinates using the wrist landmark (index 0) as the origin 
        and scales using the distance from the wrist to the middle finger MCP (index 9).

        Args:
            landmarks: MediaPipe Hand landmarks object representing 21 points.

        Returns:
            np.ndarray: A 21x3 normalized numpy array of float coordinates.
        """
        # Convert landmarks into a 21x3 float coordinate array
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])

        # 1. Translate relative to wrist (index 0)
        wrist = coords[0]
        translated = coords - wrist

        # 2. Scale relative to hand size
        # Euclidean distance from wrist (0) to middle finger base (9)
        hand_size = np.linalg.norm(coords[9] - coords[0])
        if hand_size == 0.0:
            hand_size = 1.0  # Safe guard to prevent division-by-zero

        normalized = translated / hand_size
        return normalized

    def save_sample(self, gesture_name, landmarks):
        """
        Applies normalization, flattens coordinates, and writes the 
        recorded sample as a new line inside the CSV file.

        Args:
            gesture_name (str): Label class name of the current gesture.
            landmarks: Raw MediaPipe hand landmarks object.
        """
        normalized = self.normalize_landmarks(landmarks)
        flattened = normalized.flatten()  # Converts 21x3 array into 63-element 1D list

        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([gesture_name] + list(flattened))

        # Update tracking count
        if gesture_name in self.counts:
            self.counts[gesture_name] += 1

    def _draw_overlay(self, frame, fps, last_captured_gesture, last_capture_time, current_time):
        """
        Renders HUD progress stats, instructions, and FPS markers on the video frame.
        """
        # Render translucent dashboard background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 275), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Show frames per second
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Show collection targets
        y_offset = 65
        cv2.putText(frame, "Collection Progress:", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        y_offset += 20
        for key, name in self.gesture_keys.items():
            count = self.counts.get(name, 0)
            # Switch color to green when completion targets are met
            color = (0, 255, 0) if count >= self.samples_per_class else (255, 255, 255)
            text = f"[{chr(key).upper()}] {name}: {count}/{self.samples_per_class}"
            cv2.putText(frame, text, (30, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y_offset += 18

        y_offset += 10
        cv2.putText(frame, "Press 'Q' to Exit", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        # Draw a momentary confirmation flash on-screen when keys are pressed
        if last_captured_gesture and (current_time - last_capture_time < 0.8):
            color = (0, 0, 255) if last_captured_gesture == "No Hand Detected!" else (0, 255, 0)
            text = f"Captured: {last_captured_gesture}" if last_captured_gesture != "No Hand Detected!" else last_captured_gesture
            cv2.putText(frame, text, (350, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    def collect(self):
        """
        Starts the visual display stream thread, processes hands via MediaPipe,
        binds hotkeys, and logs coordinate frames upon user request.
        """
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not locate or initialize active webcam.")
            return

        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        prev_time = time.time()
        last_captured_gesture = None
        last_capture_time = 0.0

        print("\n--- Starting Gesture Dataset Collection ---")
        print("Point the camera towards your hand and press the target keys to capture samples:")
        for key, name in self.gesture_keys.items():
            print(f"  '{chr(key)}' -> {name}")
        print("Press 'q' in the window to quit.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to acquire frame pipeline.")
                break

            # Mirror view horizontally
            frame = cv2.flip(frame, 1)

            # Convert BGR format to RGB for processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            hand_landmarks = None
            if results.multi_hand_landmarks:
                # Use the primary hand detected
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

            # Compute actual frames per second
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
            prev_time = current_time

            # Keyboard polling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in self.gesture_keys:
                gesture_name = self.gesture_keys[key]
                if hand_landmarks:
                    self.save_sample(gesture_name, hand_landmarks)
                    last_captured_gesture = gesture_name
                    last_capture_time = current_time
                else:
                    last_captured_gesture = "No Hand Detected!"
                    last_capture_time = current_time

            # Build and render user panels
            self._draw_overlay(frame, fps, last_captured_gesture, last_capture_time, current_time)
            cv2.imshow("Gesture Dataset Collection Tool", frame)

        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    # Setup instances
    collector = GestureDataCollector(dataset_path="dataset", samples_per_class=500)
    
    try:
        collector.collect()
    except Exception as error:
        print(f"Execution failed: {error}")
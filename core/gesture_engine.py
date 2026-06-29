"""
core/gesture_engine.py

Implements the central GestureEngine processing pipeline. Coordinates real-time
camera frame tracking, coordinate landmarker parsing, heuristic rule checks, 
AI-driven model classifications, temporal state filters, and automated fallbacks.
"""

import os
import sys
import time
import logging
from collections import deque
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import cv2

# Ensure parent and sibling directories are in the path to resolve local imports cleanly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    import mediapipe as mp
except ImportError:
    mp = None
    logging.warning("MediaPipe is not installed. Native frame hand tracking will be disabled.")

# Safe dynamic import of local AI inference components
AI_CAPABLE = False
try:
    from ai.infer import GestureInference
    AI_CAPABLE = True
except ImportError:
    GestureInference = None
    logging.info("GestureInference module not imported. AI-driven classification fallback only.")


# =====================================================================
# Helper Pipeline Subcomponents
# =====================================================================

class HandTracker:
    """
    Locates hand landmarks from video frames using MediaPipe.
    """
    def __init__(self, max_hands: int = 2, min_detection_confidence: float = 0.7):
        self.mp_hands = mp.solutions.hands if mp else None
        self.mp_draw = mp.solutions.drawing_utils if mp else None
        self.hands = None
        
        if self.mp_hands:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5
            )

    def find_hands(self, frame: np.ndarray) -> Optional[Any]:
        """Runs hand location on RGB frames."""
        if not self.hands:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)


class LandmarkProcessor:
    """
    Parses and standardizes raw landmarker points.
    """
    @staticmethod
    def extract_landmarks(hand_landmarks: Any) -> List[Any]:
        """Extracts individual point elements from the MediaPipe tracking array."""
        return list(hand_landmarks.landmark)

    @staticmethod
    def get_features(landmarks: List[Any]) -> Dict[str, bool]:
        """
        Extracts structural geometric relationships such as finger extensions.
        """
        features = {}
        try:
            # Finger extension heuristics: compare Tip position with PIP joint position (y-axis goes downwards)
            features["index_extended"] = landmarks[8].y < landmarks[6].y
            features["middle_extended"] = landmarks[12].y < landmarks[10].y
            features["ring_extended"] = landmarks[16].y < landmarks[14].y
            features["pinky_extended"] = landmarks[20].y < landmarks[18].y

            # Thumb extension heuristic: spatial distance from thumb tip to index knuckles
            thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
            knuckle_2 = np.array([landmarks[2].x, landmarks[2].y])
            knuckle_17 = np.array([landmarks[17].x, landmarks[17].y])
            
            thumb_mcp_dist = np.linalg.norm(thumb_tip - knuckle_17)
            hand_width_reference = np.linalg.norm(knuckle_2 - knuckle_17)
            features["thumb_extended"] = thumb_mcp_dist > (hand_width_reference * 0.9)
        except Exception:
            features = {
                "index_extended": False,
                "middle_extended": False,
                "ring_extended": False,
                "pinky_extended": False,
                "thumb_extended": False
            }
        return features


class GestureClassifier:
    """
    Applies deterministic heuristic checks to identify gesture states.
    """
    @staticmethod
    def classify(landmarks: List[Any], features: Dict[str, bool]) -> str:
        """
        Identifies gesture classes based on mechanical finger constraints.
        """
        try:
            idx = features.get("index_extended", False)
            mid = features.get("middle_extended", False)
            ring = features.get("ring_extended", False)
            pnk = features.get("pinky_extended", False)
            thb = features.get("thumb_extended", False)

            # 1. Pinch Detection (spatial proximity check between thumb tip 4 and index tip 8)
            pt4 = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
            pt8 = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
            pinch_dist = np.linalg.norm(pt4 - pt8)
            
            # Distance normalization relative to hand scale (wrist 0 to middle finger MCP 9)
            pt0 = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
            pt9 = np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])
            scale = np.linalg.norm(pt9 - pt0)
            if scale == 0:
                scale = 1.0
                
            if (pinch_dist / scale) < 0.22:
                return "pinch"

            # 2. Fist: All non-thumb fingers folded
            if not idx and not mid and not ring and not pnk:
                return "fist"

            # 3. Palm: All non-thumb fingers extended
            if idx and mid and ring and pnk:
                return "palm"

            # 4. Peace: Index and middle extended, ring and pinky folded
            if idx and mid and not ring and not pnk:
                return "peace"

            # 5. Pointing: Only index finger extended
            if idx and not mid and not ring and not pnk:
                return "pointing"

            # 6. Thumbs up: Only thumb extended
            if thb and not idx and not mid and not ring and not pnk:
                return "thumbs_up"

        except Exception:
            pass
        return "unknown"


class GestureStateMachine:
    """
    Maintains temporal state filters to smooth transition signals.
    """
    def __init__(self, window_size: int = 5, threshold: int = 4):
        self.window_size = window_size
        self.threshold = threshold
        self.history = deque(maxlen=window_size)
        self.current_state = "unknown"

    def update(self, raw_gesture: str) -> str:
        """Applies voting algorithm over a rolling window."""
        self.history.append(raw_gesture)
        counts = {}
        for item in self.history:
            counts[item] = counts.get(item, 0) + 1

        if counts:
            most_frequent, frequency = max(counts.items(), key=lambda x: x[1])
            if frequency >= self.threshold:
                self.current_state = most_frequent
                
        return self.current_state

    def reset(self) -> None:
        """Clears rolling state buffers."""
        self.history.clear()
        self.current_state = "unknown"


# =====================================================================
# Main Pipeline: GestureEngine
# =====================================================================

class GestureEngine:
    """
    Main controller coordinates and routes frames through trackers, 
    heuristics classifiers, AI classifiers, and state filters.
    """
    def __init__(self):
        # Instantiate architectural pipeline blocks
        self.tracker = HandTracker()
        self.processor = LandmarkProcessor()
        self.classifier = GestureClassifier()
        self.state_machine = GestureStateMachine()
        
        self.ai_inference: Optional[Any] = None
        self.ai_enabled: bool = False

    def enable_ai_model(self, model_path: str) -> bool:
        """
        Dynamically loads PyTorch classification model weights.

        Args:
            model_path (str): Filepath to the PyTorch saved state dictionary.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if not AI_CAPABLE or GestureInference is None:
            logging.warning("PyTorch GestureInference module is not imported. AI mode cannot be enabled.")
            self.ai_enabled = False
            return False

        try:
            self.ai_inference = GestureInference(model_path=model_path)
            self.ai_enabled = True
            logging.info("PyTorch GestureInference loaded and enabled.")
            return True
        except Exception as e:
            logging.error(f"Error loading AI model weights: {e}")
            self.ai_enabled = False
            return False

    def disable_ai_model(self) -> None:
        """Disables AI-driven classification and forces heuristic classification."""
        self.ai_enabled = False
        logging.info("AI classification disabled. Relying on heuristic fallbacks.")

    def reset(self) -> None:
        """Resets the state tracking buffers of the engine."""
        self.state_machine.reset()
        logging.info("GestureEngine state filters reset.")

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Processes an incoming video frame through the pipeline.

        Pipeline Path:
        Frame -> Hand Detection -> Landmark Extraction -> Feature Extraction 
              -> Gesture Classification -> State Filtering -> Output

        Args:
            frame (np.ndarray): OpenCV BGR frame matrix.

        Returns:
            dict: Structured output details including hand details, raw features,
                  and smoothed gesture classifications.
        """
        processed_frame = frame.copy()
        hands_data: List[Dict[str, Any]] = []
        raw_gesture = "unknown"
        accepted_gesture = "unknown"
        primary_features: Dict[str, bool] = {}

        try:
            # 1. Locate hands
            results = self.tracker.find_hands(processed_frame)

            if results and results.multi_hand_landmarks:
                for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    # 2. Extract landmarker arrays
                    landmarks = self.processor.extract_landmarks(hand_lms)
                    
                    # 3. Extract features
                    features = self.processor.get_features(landmarks)

                    # 4. Classify gesture
                    confidence = 1.0
                    gesture_candidate = "unknown"

                    # Try AI classification first if enabled, fall back to rules if uncertain
                    if self.ai_enabled and self.ai_inference:
                        try:
                            gesture_candidate, confidence = self.ai_inference.predict(hand_lms)
                            if gesture_candidate == "unknown":
                                # Automatic fallback to heuristic rules
                                gesture_candidate = self.classifier.classify(landmarks, features)
                                confidence = 0.5
                        except Exception as e:
                            logging.debug(f"AI classification error (falling back to rules): {e}")
                            gesture_candidate = self.classifier.classify(landmarks, features)
                            confidence = 0.5
                    else:
                        # Rule-based classification
                        gesture_candidate = self.classifier.classify(landmarks, features)

                    # Update primary hand stats (first hand detected)
                    if idx == 0:
                        raw_gesture = gesture_candidate
                        primary_features = features
                        # 5. Apply temporal state filtering
                        accepted_gesture = self.state_machine.update(raw_gesture)

                    hands_data.append({
                        "hand_index": idx,
                        "raw_gesture": gesture_candidate,
                        "confidence": confidence,
                        "features": features
                    })

                    # Draw standard overlay markers
                    if self.tracker.mp_draw and self.tracker.mp_hands:
                        self.tracker.mp_draw.draw_landmarks(
                            processed_frame,
                            hand_lms,
                            self.tracker.mp_hands.HAND_CONNECTIONS
                        )
            else:
                # No hands in field of view -> filter state machine towards unknown
                accepted_gesture = self.state_machine.update("unknown")

        except Exception as e:
            logging.error(f"Error within main processing pipeline: {e}")

        # Construct and return output map schema
        return {
            "frame": processed_frame,
            "hands": hands_data,
            "features": primary_features,
            "gesture": raw_gesture,
            "accepted_gesture": accepted_gesture
        }


# =====================================================================
# Main Demo Execution
# =====================================================================

if __name__ == "__main__":
    print("=== Gesture Engine System Demo ===")
    
    # Initialize main engine
    engine = GestureEngine()
    print("GestureEngine initialized. Trackers and filters ready.")

    # 1. Generate empty mock frame to verify basic processing pipeline
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(mock_frame, "Mock Frame Input", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    print("\nProcessing empty frame:")
    result = engine.process_frame(mock_frame)
    print(f"  Detected hands: {len(result['hands'])}")
    print(f"  Raw classification: '{result['gesture']}'")
    print(f"  Smoothed classification: '{result['accepted_gesture']}'")

    # 2. Start Live Camera Assessment (if webcam is connected)
    print("\nAttempting to connect to camera 0 for live feedback...")
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        print("Camera connected. Show your hand on screen. Press 'Q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip image for mirroring
            frame = cv2.flip(frame, 1)
            
            # Process frame using pipeline
            out = engine.process_frame(frame)
            processed_frame = out["frame"]
            accepted_gest = out["accepted_gesture"]
            raw_gest = out["gesture"]
            
            # Draw real-time output text
            cv2.putText(processed_frame, f"Raw: {raw_gest}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(processed_frame, f"Filtered: {accepted_gest}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Gesture Engine Live View", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        print("Camera feed released.")
    else:
        print("No video camera located. Live feedback skipped.")
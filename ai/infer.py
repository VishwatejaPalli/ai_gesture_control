"""
ai/infer.py

Implements the GestureInference engine. It loads a trained PyTorch model, 
handles real-time scale and translation normalization of hand coordinates, 
and returns classification outputs along with confidence margins and safety thresholds.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Any, Optional, Union

# Import model architecture
from model import GestureNet


class GestureInference:
    """
    An inference execution engine for real-time hand gesture classifications.
    Translates coordinate structures and provides confidence-thresholded predictions.
    """

    def __init__(
        self, 
        model_path: str, 
        labels: Optional[List[str]] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            model_path (str): Filepath to the saved state dict weights file.
            labels (list, optional): List of gesture class labels matching the training order.
            device (str, optional): Target device ('cpu' or 'cuda'). Auto-detects if None.
        """
        # Default gesture classes matching standard collection configuration
        self.labels = labels if labels is not None else [
            "fist", "palm", "peace", "pointing", "thumbs_up", "pinch"
        ]

        # Device selection
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate network
        self.model = GestureNet(num_classes=len(self.labels))
        
        # Load weights
        self.model.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, landmarks: Any) -> torch.Tensor:
        """
        Translates landmarks to set the wrist at coordinate origin (0, 0, 0)
        and scales by hand length to maintain scale invariance. Returns a PyTorch tensor.

        Args:
            landmarks: Raw landmarks object (MediaPipe output, list of dicts, or lists of coordinates).

        Returns:
            torch.Tensor: Standardized tensor of shape (1, 63).
        """
        # Parse points dynamically depending on input format structure
        if hasattr(landmarks, "landmark"):
            points = [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
        elif isinstance(landmarks, (list, np.ndarray)):
            points = []
            for lm in landmarks:
                if isinstance(lm, dict):
                    points.append([lm['x'], lm['y'], lm['z']])
                elif hasattr(lm, 'x'):
                    points.append([lm.x, lm.y, lm.z])
                else:
                    points.append([lm[0], lm[1], lm[2]])
        else:
            raise ValueError("Input landmarks format structure is not supported.")

        coords = np.array(points, dtype=np.float32)

        if coords.shape != (21, 3):
            raise ValueError(f"Expected coordinate matrix of shape (21, 3), got {coords.shape}")

        # 1. Translate relative to wrist (index 0)
        wrist = coords[0]
        translated = coords - wrist

        # 2. Scale relative to hand size
        # Euclidean distance from wrist (0) to middle finger base (9)
        hand_size = np.linalg.norm(coords[9] - coords[0])
        if hand_size < 1e-6:
            hand_size = 1.0  # Safeguard against division by zero

        normalized = translated / hand_size
        
        # Flatten to shape (63,) and add batch dimension -> (1, 63)
        flat_tensor = torch.tensor(normalized.flatten(), dtype=torch.float32).unsqueeze(0)
        return flat_tensor

    def predict(self, landmarks: Any, threshold: float = 0.70) -> Tuple[str, float]:
        """
        Classifies a single hand pose, returning the predicted class and confidence.
        If prediction confidence is below the threshold, it is rejected and labeled 'unknown'.

        Args:
            landmarks: Raw coordinate points.
            threshold (float): Rejection threshold value (0.0 to 1.0).

        Returns:
            tuple: (gesture_name, confidence)
        """
        try:
            input_tensor = self.preprocess(landmarks).to(self.device)
            
            with torch.no_grad():
                logits = self.model(input_tensor)
                # Compute probabilities using softmax
                probabilities = torch.softmax(logits, dim=1)
                
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            confidence_val = confidence.item()
            predicted_idx_val = predicted_idx.item()

            # Reject classification if confidence falls below safe margin limits
            if confidence_val < threshold:
                return "unknown", confidence_val

            gesture_name = self.labels[predicted_idx_val]
            return gesture_name, confidence_val

        except Exception as error:
            print(f"Error during prediction execution: {error}")
            return "unknown", 0.0

    def predict_batch(
        self, 
        batch_landmarks: List[Any], 
        threshold: float = 0.70
    ) -> List[Tuple[str, float]]:
        """
        Processes and classifies multiple hands in a single forward pass.

        Args:
            batch_landmarks (list): List of landmarks.
            threshold (float): Rejection threshold value (0.0 to 1.0).

        Returns:
            list: List of tuples representing (gesture_name, confidence).
        """
        if not batch_landmarks:
            return []

        try:
            tensors = [self.preprocess(lm) for lm in batch_landmarks]
            # Concatenate list of (1, 63) tensors into a batch tensor (B, 63)
            batch_tensor = torch.cat(tensors, dim=0).to(self.device)

            with torch.no_grad():
                logits = self.model(batch_tensor)
                probabilities = torch.softmax(logits, dim=1)

            confidences, indices = torch.max(probabilities, dim=1)
            results = []

            for i in range(len(batch_landmarks)):
                conf_val = confidences[i].item()
                idx_val = indices[i].item()

                if conf_val < threshold:
                    results.append(("unknown", conf_val))
                else:
                    results.append((self.labels[idx_val], conf_val))

            return results

        except Exception as error:
            print(f"Error during batch prediction execution: {error}")
            return [("unknown", 0.0) for _ in range(len(batch_landmarks))]


# Simulated landmark point class to support unit testing
class MockLandmarkPoint:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


if __name__ == "__main__":
    print("=== PyTorch Gesture Inference Verification ===")

    # Setup filepaths
    temp_model_filepath = "temp_best_model.pth"
    labels_list = ["fist", "palm", "peace", "pointing", "thumbs_up", "pinch"]

    # 1. Create a dummy model file to allow the test to execute directly
    dummy_model = GestureNet(num_classes=len(labels_list))
    dummy_model.save_model(temp_model_filepath)

    try:
        # 2. Instantiate Inference Engine
        engine = GestureInference(model_path=temp_model_filepath, labels=labels_list)
        print(f"Inference engine initialized on target hardware: '{engine.device}'")

        # 3. Assemble dummy landmarks (21 points)
        # Coordinate 0 is the wrist
        single_hand = [MockLandmarkPoint(x=0.0, y=0.0) for _ in range(21)]
        # Add spatial depth configuration for middle finger base
        single_hand[9] = MockLandmarkPoint(x=0.0, y=0.5) 

        # 4. Perform prediction (Normal execution)
        gesture, confidence = engine.predict(single_hand, threshold=0.10)
        print(f"\nSingle prediction test:")
        print(f"  Gesture: {gesture} | Confidence: {confidence:.2%}")

        # 5. Perform prediction with High Threshold (Testing rejection)
        rejected_gesture, rejected_confidence = engine.predict(single_hand, threshold=0.99)
        print(f"\nSingle prediction test (High threshold 0.99):")
        print(f"  Gesture: {rejected_gesture} | Confidence: {rejected_confidence:.2%}")

        # 6. Perform batch prediction test
        batch_list = [single_hand, single_hand]
        batch_results = engine.predict_batch(batch_list, threshold=0.10)
        print(f"\nBatch prediction test (2 samples):")
        for idx, (gest, conf) in enumerate(batch_results):
            print(f"  Sample {idx + 1}: Gesture: {gest} | Confidence: {conf:.2%}")

    finally:
        # Cleanup temporary weight files
        if os.path.exists(temp_model_filepath):
            os.remove(temp_model_filepath)
            print(f"\nCleaned up temporary model weights file: {temp_model_filepath}")
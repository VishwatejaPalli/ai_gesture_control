# ePBL Internship Project Report: Gesture AI Toolkit

**Student Name:** [Your Name]
**Internship Domain:** AI/ML & Computer Vision
**Supervisor:** [Your Supervisor's Name]
**Date:** [Date of Submission, e.g., 30 June 2026]

---

## 1. Abstract

This report details the development of the "Gesture AI Toolkit," a real-time human-computer interaction system. The project addresses the limitations of traditional input devices by enabling computer control through hand gestures. The system utilizes a webcam to capture hand movements, which are processed by Google's MediaPipe framework for landmark extraction. A custom-trained PyTorch neural network classifies these landmarks into distinct gestures such as "fist," "palm," and "pointing." These gestures are then mapped to a variety of applications, including an integrated Control Hub for mode-switching, an AR Painter, a virtual mouse, and a media controller for managing playback and volume. The project successfully demonstrates a modular and responsive framework for gesture-based control, achieving **[Your Final Accuracy, e.g., 98.7%]%** accuracy on the training set.

---

## 2. Introduction

### 2.1. Problem Statement
Traditional human-computer interaction (HCI) relies heavily on physical devices like keyboards and mice. While effective, these methods can be restrictive, non-intuitive for certain tasks, and inaccessible for some users. The goal of this project is to explore a more natural and touchless form of interaction by interpreting hand gestures from a standard webcam feed, thereby creating a more flexible and immersive user experience.

### 2.2. Project Objectives
*   To develop a robust, real-time hand gesture recognition system using computer vision and machine learning.
*   To achieve a high classification accuracy (target >95%) for a defined set of gestures.
*   To implement practical applications based on the recognition system:
    *   An AR painting tool where users can draw on their video feed.
    *   A virtual mouse to control the system cursor and perform clicks.
    *   A media controller to manage video playback (play/pause, volume, seek).
*   To build a unified "Control Hub" that integrates these functionalities into a single, mode-switchable application.

### 2.3. Scope
The project focuses on recognizing single-hand gestures from a live webcam feed. The system is trained on five specific gestures: `fist`, `palm`, `peace`, `thumbs_up`, and `pointing`. The scope does not include multi-hand gestures, full-body tracking, or the ability to dynamically add new gestures without retraining the model.

---

## 3. System Design and Methodology

### 3.1. Technology Stack
*   **Programming Language:** Python 3, chosen for its extensive scientific computing libraries and rapid prototyping capabilities.
*   **Hand Tracking:** Google MediaPipe was selected for its high-performance, real-time hand landmark detection, providing 21 3D coordinates per hand with minimal setup.
*   **Machine Learning Framework:** PyTorch was used for its flexibility, strong GPU support, and intuitive API for building and training the neural network model.
*   **Computer Vision:** OpenCV was essential for camera interfacing, image processing (flipping, color conversion), and rendering UI elements.
*   **System Interaction:** PyAutoGUI was used to programmatically control the mouse and keyboard, enabling the virtual mouse and media controller functionalities.

### 3.2. Data Collection and Preparation
The data collection process was managed by `collect_data.py`. For each of the five target gestures, 500 samples were collected. Each sample consists of the 21 hand landmarks provided by MediaPipe. The (x, y, z) coordinates for each landmark were flattened into a single vector of 63 features. This feature vector was then saved as a row in a `.csv` file, creating a structured dataset ready for training.

### 3.3. Model Architecture
The neural network, defined as `GestureNet` in `model.py`, is a Multi-Layer Perceptron (MLP). The architecture is as follows:
*   Input Layer: 63 neurons (for 21 landmarks * 3 coordinates).
*   Hidden Layer 1: 128 neurons with a ReLU activation function.
*   Hidden Layer 2: 64 neurons with a ReLU activation function.
*   Hidden Layer 3: 32 neurons with a ReLU activation function.
*   Output Layer: 5 neurons, corresponding to the number of gesture classes.
This simple yet effective architecture was chosen because the MediaPipe landmarks already provide a high-level, structured feature representation, eliminating the need for complex convolutional layers.

### 3.4. Model Training
The model was trained using the `train_model.py` script.
*   **Loss Function:** `CrossEntropyLoss` was used, as it is standard for multi-class classification problems.
*   **Optimizer:** The Adam optimizer was chosen with a learning rate of `0.001` for its efficiency and adaptive learning rate capabilities.
*   **Training Process:** The model was trained for 50 epochs with a batch size of 64. The dataset was shuffled at each epoch to prevent the model from learning the order of the data.
*   **Final Accuracy:** The model achieved a final training accuracy of **[Enter Your Final Accuracy Here, e.g., 98.7%]%**.

---

## 4. Implementation and Features

### 4.1. Control Hub (`control_hub.py`)
This is the central application of the toolkit. It operates as a state machine with three modes: INFERENCE, PAINT, and MOUSE CONTROL. The user can cycle through these modes by showing a "palm" gesture.
*   **Paint Mode:** A "pointing" gesture moves a cursor. Pinching the index finger and thumb allows the user to draw on a digital canvas overlaid on the video feed. A "peace" gesture clears the canvas.
*   **Mouse Control Mode:** A "pointing" gesture moves the system's mouse cursor. A "fist" gesture triggers a left-click.

### 4.2. Media Controller (`media_controller.py`)
This script allows for touchless control of media players.
*   **Play/Pause:** "palm" or "thumbs_up".
*   **Volume:** "pointing" gesture, moving the hand up or down.
*   **Seeking:** "fist" gesture, moving the hand left or right.
Cooldown timers are implemented to prevent a single gesture from firing an action multiple times in rapid succession.

### 4.3. AR Painter (`ar_paint.py`)
This standalone script implements the AR painting feature. It uses a transparent overlay (`canvas`) to store the drawings. A pinch gesture between the thumb and index finger is used to draw, while an open hand gesture clears the canvas. It also includes a feature to save a screenshot of the AR drawing.

---

## 5. Results and Discussion

### 5.1. Model Performance
The trained `GestureNet` model performed exceptionally well on the collected data, reaching a final accuracy of **[Enter Your Final Accuracy Here]**. This high accuracy confirms that the MLP architecture is well-suited for classifying the structured landmark data from MediaPipe.
*(Optional: If you created a confusion matrix, you can insert it here and discuss any minor misclassifications, e.g., between 'fist' and 'palm' in certain lighting conditions.)*

### 5.2. Application Functionality
The applications are highly responsive, with minimal latency between performing a gesture and seeing the result. The gesture mappings proved to be intuitive during testing.
Below are screenshots demonstrating the key functionalities.

**Control Hub in Paint Mode**
*This screenshot shows the main `control_hub.py` application in "PAINT" mode, with some drawings on the canvas.*
!Control Hub in Paint Mode

**Media Controller Feedback**
*This screenshot shows the `media_controller.py` script providing visual feedback ("Volume Up") after a gesture was performed.*
!Media Controller Feedback

**AR Paint Demo**
*This screenshot shows the legacy `ar_paint.py` script, demonstrating the core drawing functionality.*
!AR Paint Demo

### 5.3. Challenges and Solutions
*   **Challenge:** Jittery mouse movement due to minor fluctuations in hand tracking.
*   **Solution:** This was addressed in `mouse.py` and `control_hub.py` by using `np.interp` to map the camera's coordinate space to the screen's coordinate space, which provides a smoothing effect.
*   **Challenge:** A single gesture (e.g., "fist" for click) being triggered repeatedly.
*   **Solution:** Cooldown timers (`last_action_time`) were implemented in `control_hub.py` and `media_controller.py`. A new action can only be triggered if a certain amount of time has passed since the last one.

---

## 6. Conclusion and Future Work

### 6.1. Conclusion
The Gesture AI Toolkit project successfully met all its objectives. A reliable gesture recognition model was developed and integrated into several practical applications, demonstrating the viability of touchless HCI using commodity hardware. The final system is robust, responsive, and provides a strong foundation for more advanced gesture-based applications.

### 6.2. Future Work
*   **Expand Gesture Library:** Collect data for and train more complex gestures, such as an "OK" sign or a "call me" sign, to increase the system's vocabulary.
*   **Two-Handed Gestures:** Extend the system to track both hands, enabling more complex interactions like zooming (by pinching with two hands) or rotating objects.
*   **GUI for Training:** Create a graphical user interface that simplifies the data collection and model training workflow for non-technical users.
*   **Dynamic Gestures:** Implement a system (e.g., using an LSTM or Transformer model) that can recognize dynamic gestures that unfold over time, such as a swipe or a wave.

---

## 7. References
*   Google MediaPipe Documentation: https://google.github.io/mediapipe/
*   PyTorch Documentation: https://pytorch.org/docs/stable/index.html
*   OpenCV-Python Tutorials: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
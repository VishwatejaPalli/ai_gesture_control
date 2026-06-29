# ePBL Internship Project Report: Gesture AI Toolkit

**Student Name:** [Your Name]
**Internship Domain:** [Your Domain, e.g., AI/ML, Computer Vision]
**Supervisor:** [Your Supervisor's Name]
**Date:** [Date of Submission, e.g., 30 June 2026]

---

## 1. Abstract

*A brief summary (150-250 words) of the entire project. It should cover the problem, the solution, the key technologies used, and the main results. Write this section last.*

---

## 2. Introduction

### 2.1. Problem Statement
*Describe the problem you are trying to solve. Why is gesture control useful? What are the limitations of current human-computer interaction methods that this project addresses?*

### 2.2. Project Objectives
*List the specific, measurable goals you set out to achieve. For example:*
*   *To develop a real-time hand gesture recognition system with at least 95% accuracy.*
*   *To create an intuitive AR painting application controlled by hand movements.*
*   *To implement a media controller that uses gestures to manage playback, volume, and seeking.*
*   *To build a unified "Control Hub" that integrates multiple functionalities.*

### 2.3. Scope
*Define the boundaries of your project. What features are included? What features are intentionally left out? (e.g., "The project focuses on single-hand gestures and does not support multi-hand or full-body gestures.")*

---

## 3. System Design and Methodology

### 3.1. Technology Stack
*List and justify the key technologies, libraries, and frameworks you used.*
*   **Programming Language:** Python
*   **Hand Tracking:** Google MediaPipe (explain why it was chosen - e.g., performance, ease of use).
*   **Machine Learning Framework:** PyTorch (explain why - e.g., flexibility, dynamic computation graph).
*   **Computer Vision:** OpenCV
*   **GUI/Interaction:** PyAutoGUI

### 3.2. Data Collection and Preparation
*Describe the process you followed using `collect_data.py`.*
*   *What gestures did you collect?*
*   *How many samples per gesture?*
*   *How is the data structured (e.g., 21 landmarks with x, y, z coordinates, resulting in 63 features)?*

### 3.3. Model Architecture
*Explain the neural network defined in `model.py` (`GestureNet`).*
*   *Describe the layers (Linear, ReLU, Dropout).*
*   *Why was this architecture chosen? (e.g., "A simple MLP is sufficient for this classification task as the input data is already a structured feature vector.")*

### 3.4. Model Training
*Detail the training process from `train_model.py`.*
*   *What was the loss function (Cross-Entropy Loss)?*
*   *What was the optimizer (Adam)?*
*   *What were the key hyperparameters (learning rate, batch size, number of epochs)?*
*   *What was the final training accuracy?*

---

## 4. Implementation and Features

*This is the core section where you describe what you built. Dedicate a subsection to each major script.*

### 4.1. Control Hub (`control_hub.py`)
*Explain the state machine logic (switching between INFERENCE, PAINT, MOUSE CONTROL modes). Describe how gestures trigger different actions in each mode.*

### 4.2. Media Controller (`media_controller.py`)
*Describe how gestures are mapped to media keys. Explain the logic for continuous control (volume/seek) vs. discrete actions (play/pause). Mention the use of cooldowns to prevent spamming commands.*

### 4.3. AR Painter (`ar_paint.py`)
*Explain the canvas overlay technique. Describe how you differentiate between moving the cursor and drawing (e.g., pinch gesture). Detail the screenshot functionality.*

---

## 5. Results and Discussion

### 5.1. Model Performance
*Present the final accuracy of your trained model. If possible, include a confusion matrix to show which gestures were sometimes misclassified.*

### 5.2. Application Functionality
*Discuss how well the applications work. Are they responsive? Are the gestures intuitive? Include the screenshots you took here as well.*

### 5.3. Challenges and Solutions
*Describe any significant challenges you faced during the project and how you overcame them. Examples:*
*   *Challenge: Jittery mouse movement.*
*   *Solution: Implemented smoothing or used `np.interp` for stable mapping.*
*   *Challenge: Model misclassifying 'fist' and 'palm'.*
*   *Solution: Collected more varied data or adjusted model hyperparameters.*

---

## 6. Conclusion and Future Work

### 6.1. Conclusion
*Summarize the project's achievements and reiterate how you met the objectives.*

### 6.2. Future Work
*Suggest potential improvements or new features. For example:*
*   *Adding support for more complex gestures or two-handed gestures.*
*   *Creating a more user-friendly GUI for data collection and training.*
*   *Porting the application to a mobile device.*

---

## 7. References

*List any articles, documentation, or tutorials you found helpful. (e.g., MediaPipe documentation, PyTorch tutorials).*
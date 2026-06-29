"""
ai/model.py

Implements GestureNet, a deep feedforward neural network in PyTorch designed
to classify 21 hand landmarks (63 coordinates) into discrete gesture classes.
"""

import os
import torch
import torch.nn as nn


class GestureNet(nn.Module):
    """
    A PyTorch Neural Network module designed to classify hand gesture landmarks.
    Accepts 21 3D coordinates (63 features total) and maps them to gesture classes.
    """

    def __init__(self, num_classes: int, dropout_rate: float = 0.3):
        """
        Initializes the model architecture.

        Args:
            num_classes (int): The number of gesture classes to classify.
            dropout_rate (float): The dropout probability used to combat overfitting.
        """
        super(GestureNet, self).__init__()
        
        # Layer 1: Input size 63 to 128 hidden units
        self.fc1 = nn.Linear(63, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Layer 2: 128 to 64 hidden units
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Layer 3: 64 to 32 hidden units
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        
        # Layer 4: Output layer mapping 32 units to target gesture classes
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the neural network.

        Args:
            x (torch.Tensor): Input batch tensor of shape (batch_size, 63).

        Returns:
            torch.Tensor: Raw output logits of shape (batch_size, num_classes).
        """
        # Pass through the first hidden layer block
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Pass through the second hidden layer block
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Pass through the third hidden layer block
        x = self.fc3(x)
        x = self.relu3(x)
        
        # Linear output layer (no softmax, as CrossEntropyLoss handles logits natively)
        x = self.fc4(x)
        return x

    def save_model(self, path: str) -> None:
        """
        Saves the model's state dictionary parameters to a file.

        Args:
            path (str): Filepath where model weights will be saved.
        """
        # Create directory if it does not exist
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.state_dict(), path)
        print(f"Model successfully saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Loads the model's state dictionary parameters from a file.

        Args:
            path (str): Filepath containing the saved model state dictionary.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)
        self.eval()  # Set the model to evaluation mode after loading weights
        print(f"Model successfully loaded from {path}")


if __name__ == "__main__":
    print("=== PyTorch GestureNet Instantiation and Sanity Test ===")
    
    # 1. Define network config
    num_classes = 6
    batch_size = 4
    temp_model_path = "temp_gesture_net.pth"

    # 2. Instantiate Model
    model = GestureNet(num_classes=num_classes)
    print(f"Model initialized:\n{model}\n")

    # 3. Simulate dummy tensor inputs
    # Shape: (batch_size, 63 features)
    dummy_input = torch.randn(batch_size, 63)
    print(f"Input batch shape: {dummy_input.shape}")

    # 4. Perform forward pass
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Forward pass output shape: {output.shape} (Expected: {batch_size}, {num_classes})")
    print(f"Logits snippet:\n{output}\n")

    # 5. Test saving model weights
    try:
        model.save_model(temp_model_path)
        
        # 6. Test loading model weights into a fresh instance
        fresh_model = GestureNet(num_classes=num_classes)
        fresh_model.load_model(temp_model_path)
        
        # 7. Verify functional equivalence post-load
        with torch.no_grad():
            fresh_output = fresh_model(dummy_input)
            
        is_identical = torch.allclose(output, fresh_output, atol=1e-6)
        print(f"Equivalence validation: {is_identical} (Outputs are consistent)")
        
    finally:
        # Cleanup temporary file
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
            print(f"Cleaned up temporary test file: {temp_model_path}")
import torch
import numpy as np
from model import GestureNet

LABELS = [
    "fist",
    "palm",
    "peace",
    "pointing",
    "thumbs_up",
    "pinch"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")

model = GestureNet(num_classes=6)
model.load_model("models/best_model.pth")
model.to(DEVICE)
model.eval()

print("Model loaded.")

# create random normalized hand
sample = np.random.randn(63).astype(np.float32)

x = torch.tensor(sample).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)

print()
print("Raw logits:")
print(logits)

print()
print("Probabilities:")
print(probs)

conf, idx = torch.max(probs, dim=1)

print()
print("Prediction:")
print(LABELS[idx.item()])
print(f"Confidence: {conf.item()*100:.2f}%")

print()
print("Sum:", probs.sum().item())
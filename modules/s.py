import torch

checkpoint = torch.load(
    "models/best_model.pth",
    map_location="cpu"
)

print(type(checkpoint))

if isinstance(checkpoint, dict):
    print(checkpoint.keys())
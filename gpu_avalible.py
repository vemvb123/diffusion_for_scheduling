"""
gpu_avalible.py contains code for checking if the gpu is avalible
"""

import torch

# Check if CUDA GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available! Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

# Create a simple tensor
x = torch.randn(3, 3)

# Move tensor to the chosen device
x = x.to(device)
print("Tensor device:", x.device)

# Example operation on that device
y = x * 2
print("Result:", y)

# If you have a model, move it too
# model = MyModel()
# model.to(device)
# output = model(x)

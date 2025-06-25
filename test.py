import torch


print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Default device:", "mps" if torch.backends.mps.is_available()
      else "cuda" if torch.cuda.is_available() else "cpu")

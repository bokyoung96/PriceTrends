from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DeviceSelector:
    prefer_cuda: bool = True
    prefer_mps: bool = True

    def resolve(self) -> torch.device:
        if self.prefer_cuda and torch.cuda.is_available():
            return torch.device("cuda")

        if self.prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    def summary(self, label: Optional[str] = None) -> str:
        device = self.resolve()
        prefix = f"[{label}] " if label else ""
        if device.type == "cuda":
            name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CUDA"
            return f"{prefix}Using CUDA GPU ({name})"
        if device.type == "mps":
            return f"{prefix}Using Apple MPS GPU"
        return f"{prefix}Using CPU"

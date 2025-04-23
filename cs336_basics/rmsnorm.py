import torch
import math
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        weights: dict[str, torch.FloatTensor],
        eps: float = 1e-5
    ):
        super().__init__()
        self.d_model = d_model
        self.weights = weights['weight']
        self.eps = eps
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x / (self.rms(x)) * self.weights

    def rms(self, x: torch.FloatTensor):
        return torch.sqrt(
            torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps
        )


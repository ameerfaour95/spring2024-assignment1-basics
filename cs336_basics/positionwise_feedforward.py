import torch
import math
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.FloatTensor):
        return 0.5 * x * (
            1 + torch.erf(x / math.sqrt(2))
        )

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gelu = GELU()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.FloatTensor):
        x = self.w1(x)
        x = self.gelu(x)
        x = self.w2(x)
        return x
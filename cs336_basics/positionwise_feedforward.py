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
    def __init__(self, w1: torch.FloatTensor, w2: torch.FloatTensor):
        super().__init__()
        self.gelu = GELU()
        self.w1 = w1
        self.w2 = w2

    def forward(self, x: torch.FloatTensor):
        x = torch.matmul(x, self.w1.transpose(0, 1))
        x = self.gelu(x)
        return torch.matmul(x, self.w2.transpose(0, 1))
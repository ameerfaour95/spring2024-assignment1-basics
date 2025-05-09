import torch
import math
import torch.nn as nn

GPT2_PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Softmax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.FloatTensor):
        max_ent = torch.max(x, dim=self.dim, keepdim=True).values
        x_exp = torch.exp(x - max_ent)
        x_exp_sum = torch.sum(x_exp, dim=self.dim, keepdim=True)
        return x_exp / x_exp_sum
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.FloatTensor):
        return 0.5 * x * (
            1 + torch.erf(x / math.sqrt(2))
        )
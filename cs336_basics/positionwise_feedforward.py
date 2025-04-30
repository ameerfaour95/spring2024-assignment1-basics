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
        self.ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.ff2 = nn.Linear(d_ff, d_model, bias=False)

    def load_state_dict(self, state_dict, strict = True, assign = False):
        new_state_dict = {}
        for k, v in state_dict.items():
            if "w1" in k:
                new_state_dict["ff1.weight"] = v
            elif "w2" in k:
                new_state_dict["ff2.weight"] = v
        return super().load_state_dict(new_state_dict, strict, assign)

    def forward(self, x: torch.FloatTensor):
        x = self.ff1(x)
        x = self.gelu(x)
        x = self.ff2(x)
        return x
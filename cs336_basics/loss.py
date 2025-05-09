import torch
from torch import nn
from cs336_basics.utils import Softmax

class CrossEntropyLoss(nn.Module):
    def __init__(self, targets: torch.LongTensor, dim: int):
        super().__init__()
        self.softmax = Softmax(dim=dim)
        self.targets = targets
    
    def forward(self, x):
        log_sum_exp = torch.logsumexp(x, dim=-1)
        targets_logits = x[range(x.shape[0]), self.targets]
        loss = log_sum_exp - targets_logits
        return torch.mean(loss)
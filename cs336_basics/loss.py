import torch
from torch import nn
class CrossEntropyLoss(nn.Module):
    def forward(self, logits, targets: torch.LongTensor):
        vocab_size = logits.shape[-1]
    
        flatten_logits = logits.view(-1, vocab_size)
        flatten_targets = targets.view(-1)

        log_sum_exp = torch.logsumexp(flatten_logits, dim=-1)
        targets_logits = flatten_logits[range(flatten_logits.shape[0]), flatten_targets]
        loss = log_sum_exp - targets_logits
        return torch.mean(loss)
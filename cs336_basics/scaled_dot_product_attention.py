import torch
import torch.nn as nn
import math
from typing import Optional
from cs336_basics.softmax import Softmax

def attention_head(
    K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
    pdrop: Optional[float] = None
):
    dk = K.shape[-1]
    softmax = Softmax(dim=-1)
    pre_softmax = (torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk))
    if mask is not None:
        masked_pre_softmax = pre_softmax.masked_fill(mask, -torch.inf)
    else:
        masked_pre_softmax = pre_softmax
    attn_weights = softmax(masked_pre_softmax)
    if pdrop is not None:
        dropout = nn.Dropout(pdrop)
        attn_weights = dropout(attn_weights)

    return torch.matmul(attn_weights, V)
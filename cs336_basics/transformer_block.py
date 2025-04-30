import torch
import torch.nn as nn
from cs336_basics.multihead_self_attention import MultiHeadAttention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.positionwise_feedforward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float | None = None,
        residual_pdrop: float | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff)
        self.attn = MultiHeadAttention(d_model, num_heads, attn_pdrop)
        self.dropout1 = nn.Dropout(residual_pdrop)
        self.dropout2 = nn.Dropout(residual_pdrop)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        residual_skip = x
        x = self.ln1(x)
        x = self.attn(x)
        x = self.dropout1(x)
        x = x + residual_skip

        residual_skip = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = x + residual_skip

        return x
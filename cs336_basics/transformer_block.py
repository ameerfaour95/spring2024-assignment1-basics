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
        self.rms_norm_1 = RMSNorm(d_model=d_model)
        self.rms_norm_2 = RMSNorm(d_model=d_model)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff)
        self.mha = MultiHeadAttention(d_model, num_heads, attn_pdrop)
        self.residual_dropout = nn.Dropout(residual_pdrop)
    
    def load_state_dict(self, state_dict, strict = True, assign = False):
        mha_weights, ffn_weights, ln1_weights, ln2_weights = {}, {}, {}, {}
        for key, value in state_dict.items():
            if "attn" in key:
                mha_weights[key[len("attn."):]] = value
            elif "ffn" in key:
                ffn_weights[key[len("ffn."):]] = value
            elif "ln1." in key:
                ln1_weights[key[len("ln1."):]] = value
            elif "ln2." in key:
                ln2_weights[key[len("ln2."):]] = value
        
        self.rms_norm_1.load_state_dict(ln1_weights, strict, assign)
        self.rms_norm_2.load_state_dict(ln2_weights, strict, assign)
        self.mha.load_state_dict(mha_weights, strict, assign)
        self.ffn.load_state_dict(ffn_weights, strict, assign)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        residual_skip = x
        x = self.rms_norm_1(x)
        x = self.mha(x)
        x = self.residual_dropout(x)
        x = x + residual_skip

        residual_skip = x
        x = self.rms_norm_2(x)
        x = self.ffn(x)
        x = self.residual_dropout(x)
        x = x + residual_skip

        return x
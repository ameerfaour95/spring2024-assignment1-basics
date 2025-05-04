import torch
import math
import torch.nn as nn
from typing import Optional
from cs336_basics.utils import Softmax, GELU

def scaled_dot_product_attention(
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

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5
    ):
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.randn(d_model))
        self.eps = eps
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x / (self.rms(x)) * self.weight

    def rms(self, x: torch.FloatTensor):
        return torch.sqrt(
            torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps
        )
    

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_pdrop: float | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_pdrop = attn_pdrop

    @staticmethod
    def _convert_per_head_weights_to_batched(weights: dict[str, torch.FloatTensor], d_model: int, num_heads: int) -> dict[str, torch.FloatTensor]:
        q_proj_weight = torch.cat([weights[f"q_heads.{i}.weight"] for i in range(num_heads)], dim=0)
        k_proj_weight = torch.cat([weights[f"k_heads.{i}.weight"] for i in range(num_heads)], dim=0)
        v_proj_weight = torch.cat([weights[f"v_heads.{i}.weight"] for i in range(num_heads)], dim=0)

        output_proj_weight = weights["output_proj.weight"]

        return {
            "q_proj.weight": q_proj_weight,
            "k_proj.weight": k_proj_weight,
            "v_proj.weight": v_proj_weight,
            "output_proj.weight": output_proj_weight,
        }

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if any("q_heads" in k for k in state_dict.keys()):
            state_dict = self._convert_per_head_weights_to_batched(
                weights=state_dict,
                d_model=self.d_model,
                num_heads=self.num_heads
            )

        return super(MultiHeadAttention, self).load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, x):
        b, seq_len, _ = x.shape

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = q.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        att = scaled_dot_product_attention(
            Q=q,
            K=k,
            V=v,
            mask=mask,
            pdrop=self.attn_pdrop
        )
        att = att.transpose(1, 2).contiguous()
        out = att.view(b, seq_len, self.d_model)

        return self.output_proj(out)
    

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
    

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float | None = None,
        residual_pdrop: float | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.layers = nn.Sequential(
            *[TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop)
            for _ in range(num_layers)]
        )
        self.residual_dropout = nn.Dropout(residual_pdrop)
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, seq_len = x.shape
        tok_emb = self.token_embeddings(x)
        pos_emb = self.position_embeddings(
            torch.arange(seq_len, device=x.device)
        )
        x = tok_emb + pos_emb
        x = self.residual_dropout(x)

        x = self.layers(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

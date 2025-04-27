import torch
import torch.nn as nn
from cs336_basics.scaled_dot_product_attention import attention_head

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        weights: dict[str, torch.FloatTensor],
        attn_pdrop: float | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.weights = weights
        self.W_q = self._stack_weights(weights, "q_heads")
        self.W_k = self._stack_weights(weights, "k_heads")
        self.W_v = self._stack_weights(weights, "v_heads")
        self.W_o = weights['output_proj.weight']
        self.attn_pdrop = attn_pdrop

    def _stack_weights(
        self,
        weights: dict[str, torch.FloatTensor],
        prefix: str
    ):
        pre_head = [weights[f"{prefix}.{h}.weight"].T
                    for h in range(self.num_heads)
        ]
        return torch.cat(pre_head, dim=1)

    def forward(self, x):
        head_dim = self.d_model // self.num_heads
        b, seq_len, _ = x.shape

        q = torch.matmul(x, self.W_q)
        q = q.view(b, seq_len, self.num_heads, head_dim).transpose(1, 2)

        k = torch.matmul(x, self.W_k)
        k = k.view(b, seq_len, self.num_heads, head_dim).transpose(1, 2)

        v = torch.matmul(x, self.W_v)
        v = v.view(b, seq_len, self.num_heads, head_dim).transpose(1, 2)

        mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        att = attention_head(
            Q=q,
            K=k,
            V=v,
            mask=mask,
            pdrop=self.attn_pdrop
        )
        att = att.transpose(1, 2).contiguous()
        out = att.view(b, seq_len, self.d_model)

        return torch.matmul(out, self.W_o.T)
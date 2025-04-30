import torch
import torch.nn as nn
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention

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
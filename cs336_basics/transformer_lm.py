import torch
import torch.nn as nn
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm

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


import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from cs336_basics.utils.nn import softmax
    
def scaled_dot_product_attention(
    K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
    pdrop: Optional[float] = None,
) -> torch.FloatTensor:
    # Compute the dot product of Q and K
    dot_product = torch.matmul(Q, K.transpose(-1, -2))
    
    # Scale the dot product
    scaled_dot_product = dot_product / np.sqrt(K.size(-1))
    
    # Apply the mask
    if mask is not None:
        scaled_dot_product = scaled_dot_product.masked_fill(mask, float('-inf'))

    # Compute the attention weights
    attention_weights = softmax(scaled_dot_product, dim=-1)

    # Apply dropout
    if pdrop is not None:
        attention_weights = nn.functional.dropout(attention_weights, pdrop)
    
    # Compute the weighted sum of the values
    output = torch.matmul(attention_weights, V)
    
    return output


class GELU(nn.Module):
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return 0.5 * x * (1 + torch.erf(x / np.sqrt(2)))

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        rms_x = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)+self.eps)
        x = (x / rms_x) * self.weight
        return x

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(FFN, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.gelu = GELU()
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.w1(x)
        x = self.gelu(x)
        x = self.w2(x)
        return x

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: float):
        super(MultiheadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.d_k = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        B, T, _ = x.size()
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        x = scaled_dot_product_attention(k, q, v, mask=mask, pdrop=self.attn_pdrop)
        x = x.transpose(1, 2)
        x = x.contiguous().view(B, T, self.d_model)
        x = x.view(B, T, self.d_model)
        x = self.output_proj(x)
        return x

    def load_state_dict_test(self, state_dict):
        """
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `q_heads.{N}.weight`, `q_heads.{N}.weight`:
                Weights for the query projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_key, d_model).
            - `k_heads.{N}.weight`, `k_heads.{N}.weight`:
                Weights for the key projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_key, d_model).
            - `v_heads.{N}.weight`, `v_heads.{N}.weight`:
                Weights for the value projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_value, d_model).
            - `output_proj.weight`:
                Weight of the output projection
                (W^{O} in the original Transformer paper)
                Shape of (d_model, d_value * num_heads).
        """
        weights = state_dict
        for i in range(self.num_heads):
            self.q_proj.weight.data[i*self.d_k:(i+1)*self.d_k] = weights[f"q_heads.{i}.weight"]
            self.k_proj.weight.data[i*self.d_k:(i+1)*self.d_k] = weights[f"k_heads.{i}.weight"]
            self.v_proj.weight.data[i*self.d_k:(i+1)*self.d_k] = weights[f"v_heads.{i}.weight"]
        self.output_proj.weight.data = weights['output_proj.weight']
        
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, attn_pdrop: float, resid_pdrop: float):
        super(TransformerBlock, self).__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, num_heads, attn_pdrop)
        self.drop1 = nn.Dropout(resid_pdrop)
        
        self.ln2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        self.drop2 = nn.Dropout(resid_pdrop)
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x + self.drop1(self.attn(self.ln1(x)))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int,
                 d_model: int, num_heads: int, d_ff: int, attn_pdrop: float,
                 resid_pdrop: float, **kwargs):
        super(TransformerLM, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, resid_pdrop) for _ in range(num_layers)
        ])
        self.drop = nn.Dropout(resid_pdrop)
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        B, T = x.size()
        positions = torch.arange(T, device=x.device).expand(B, T)
        x = self.token_embeddings(x) + self.position_embeddings(positions)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

class TransformerBlockAblation(nn.Module):
    def __init__(self, d_model: int,
                 num_heads: int,
                 d_ff: int,
                 attn_pdrop: float,
                 resid_pdrop: float,
                 no_rmsnorm: bool=False,
                 parallel_layers: bool=False,
                 post_norm: bool=False):
        super(TransformerBlockAblation, self).__init__()
        if not no_rmsnorm:
            self.ln1 = RMSNorm(d_model)
            self.ln2 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, num_heads, attn_pdrop)
        self.drop1 = nn.Dropout(resid_pdrop)
        
        self.ffn = FFN(d_model, d_ff)
        self.drop2 = nn.Dropout(resid_pdrop)

        self.no_rmsnorm = no_rmsnorm
        self.parallel_layers = parallel_layers
        self.post_norm = post_norm
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if self.no_rmsnorm:
            x = x + self.drop1(self.attn(x))
            x = x + self.drop2(self.ffn(x))
        elif self.parallel_layers:
            x1 = x + self.drop1(self.attn(self.ln1(x)))
            x2 = x + self.drop2(self.ffn(self.ln2(x)))
            x = x1 + x2
        elif self.post_norm:
            x = self.ln1(x + self.drop1(self.attn(x)))
            x = self.ln2(x + self.drop2(self.ffn(x)))
        else:
            x = x + self.drop1(self.attn(self.ln1(x)))
            x = x + self.drop2(self.ffn(self.ln2(x)))
        return x

class TransformerLMAblation(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int,
                 d_model: int, num_heads: int, d_ff: int, attn_pdrop: float,
                 resid_pdrop: float,
                 no_rmsnorm: bool=False, parallel_layers: bool=False, post_norm: bool=False,
                 **kwargs):

        super(TransformerLMAblation, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.layers = nn.ModuleList([
            TransformerBlockAblation(d_model, num_heads, d_ff, attn_pdrop, resid_pdrop,
                                  no_rmsnorm, parallel_layers, post_norm
                                 ) for _ in range(num_layers)
        ])
        self.drop = nn.Dropout(resid_pdrop)
        if not no_rmsnorm:
            self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.no_rmsnorm = no_rmsnorm
        self.parallel_layers = parallel_layers
        self.post_norm = post_norm

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        B, T = x.size()
        positions = torch.arange(T, device=x.device).expand(B, T)
        x = self.token_embeddings(x) + self.position_embeddings(positions)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x)
        if not self.no_rmsnorm:
            x = self.ln_final(x)
        x = self.lm_head(x)
        return x
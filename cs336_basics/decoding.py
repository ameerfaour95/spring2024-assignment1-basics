import torch
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

def sample_by_top_p(probs: torch.Tensor, top_p: float):
    probs = probs.squeeze(0)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    cumulative_probs = torch.cumsum(sorted_probs, dim=0)

    cut_off = torch.searchsorted(cumulative_probs, top_p)

    nucleus_mask = torch.zeros_like(sorted_probs, dtype=torch.bool)
    nucleus_mask[:cut_off + 1] = True

    nucleus_probs = torch.where(
    nucleus_mask,
    sorted_probs,
    torch.zeros_like(sorted_probs, dtype=sorted_probs.dtype)  # now correct
)
    nucleus_probs /= nucleus_probs.sum()

    sampled_idx_on_sorted = torch.multinomial(nucleus_probs, num_samples=1)
    sampled_vocab_idx = sorted_indices[sampled_idx_on_sorted]

    return sampled_vocab_idx.unsqueeze(0)

def generate_text(
    prompt: str,
    model: TransformerLM,
    tokenizer: Tokenizer,
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    model.eval()
    device = next(model.parameters()).device
    context_size = model.position_embeddings.num_embeddings
    idx = tokenizer.encode(prompt)
    idx = torch.tensor(
        idx,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    stop_token = torch.tensor(
        [tokenizer.encode(tok)
         for tok in tokenizer.special_tokens
         if tok ==  "<|endoftext|>"]
    ).item()

    for _ in range(max_tokens):
        with torch.no_grad():
            idx_context = idx[:, -context_size:]

            logits = model(idx_context)

            last_token_logit = logits[:, -1, :]

            probs = torch.softmax(
                last_token_logit / (temperature + 1e-6
            ), dim=-1)

            idx_next = sample_by_top_p(probs, top_p)

            if idx_next.item() == stop_token:
                generated_text = tokenizer.decode(idx.squeeze(0).tolist())
                return generated_text
    
            idx = torch.cat((idx, idx_next), dim=1)
            generated_text = tokenizer.decode(idx.squeeze(0).tolist())

    return generated_text
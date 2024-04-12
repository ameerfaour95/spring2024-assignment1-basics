import torch
import numpy as np
from typing import Tuple

def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset `x`, this function will return a batch of size `batch_size` with context length `context_length`.
    The batch will be a tuple of two tensors: the first tensor will be the context, and the second tensor will be the target.
    The target will be the context shifted by one.
    """
    # Get the starting indices for each batch
    starting_indices = torch.randint(0, len(dataset) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy(dataset[start_idx:start_idx + context_length]) for start_idx in starting_indices])
    y = torch.stack([torch.from_numpy(dataset[start_idx + 1:start_idx + context_length + 1]) for start_idx in starting_indices])
    return x.to(device), y.to(device)
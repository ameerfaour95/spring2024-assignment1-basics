import torch
import numpy as np
import numpy.typing as npt
import random


def get_batch(
 dataset: npt.NDArray,
 batch_size: int,
 context_length: int, 
 device: str
):
    input_seq_lst = []
    target_seq_lst = []

    if device == "cuda:0":
        if not torch.cuda.is_available():
            raise ValueError("Cuda is not available")

    elif device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("mps is not available")

    while len(input_seq_lst) < batch_size:
        i = random.choice(range(len(dataset)))
        _current = dataset[i: i + context_length]
        _next = dataset[i + 1: i + context_length + 1]
        if len(_next) < context_length:
            continue
        input_seq_lst.append(_current)
        target_seq_lst.append(_next)
    
    input_array = np.array(input_seq_lst)
    target_array = np.array(target_seq_lst)

    input_tensor  = torch.from_numpy(input_array.astype(np.int64, copy=False)).to(device)
    target_tensor = torch.from_numpy(target_array.astype(np.int64, copy=False)).to(device)

    return input_tensor, target_tensor

import torch
from torch import nn, optim
import typing
import os

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    checkpoint = {}
    checkpoint["iteration"] = iteration
    checkpoint["model_state"] = model.state_dict()
    checkpoint["optimizer_state"] = optimizer.state_dict()
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: optim.Optimizer
):
    checkpoint = torch.load(src)
    iteration = checkpoint.get("iteration", 0)
    model_state = checkpoint.get("model_state", None)
    optimizer_state = checkpoint.get("optimizer_state", None)
    if model_state is None or optimizer is None:
        raise Exception("There is no model state or optimizer founded in the checkpoint")
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    return iteration
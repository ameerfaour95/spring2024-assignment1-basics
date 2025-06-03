import torch
import torch.nn as nn
import math
from typing import Optional, Callable, Iterable


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, 
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                grad = p.grad.data
                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * (grad**2)
                lr_corrected = lr * (math.sqrt(1 - (beta_2 ** t))) / (1 - (beta_1 ** t))
                p.data -= lr_corrected * (m / (torch.sqrt(v) + eps))
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        progress = (it - warmup_iters)/(cosine_cycle_iters - warmup_iters)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        diff_max_min = max_learning_rate - min_learning_rate
        return min_learning_rate + cosine_decay * diff_max_min

    return min_learning_rate


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps = 1e-6
):
    combined_g_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            g_norm = torch.norm(p.grad, 2)
            combined_g_norm += g_norm.item() ** 2
    combined_g_norm = combined_g_norm ** 0.5
    if combined_g_norm >= max_l2_norm:
        scale = max_l2_norm / (combined_g_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale)
import torch

import numpy as np
import contextlib

@contextlib.contextmanager
def random_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

@contextlib.contextmanager
def random_seed_torch(seed, device=0):
    cpu_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        gpu_rng_state = torch.cuda.get_rng_state(0)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(gpu_rng_state, device)

def maybe_torch(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    else:
        return value

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_params(params):
    return sum(p.numel() for p in params if p.requires_grad)

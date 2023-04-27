import random

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_sequence(val):
    if isinstance(val, (list, tuple)) or len(val.shape) > 0:
        return val
    else:
        return [val]


def calculate_constraints(model, model_target, layer_target):
    model_sq_l2 = squared_l2(model)
    layer_sq_l2 = layerwise_squared_l2(model)

    model_violation = model_sq_l2 - model_target
    layer_violation = layer_sq_l2 - layer_target

    return torch.cat([model_violation.unsqueeze(0), layer_violation])


def squared_l2(model):
    l2_norm = 0
    for param in model.parameters():
        l2_norm += torch.sum(param**2)
    return l2_norm


def layerwise_squared_l2(model):
    sq_norms = []
    for layer in model.layers:
        if isinstance(layer, torch.nn.Linear):
            sq_norms.append(torch.sum(layer.weight**2) + torch.sum(layer.bias**2))

    return torch.stack(sq_norms)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_func(output, target):
    return (output.argmax(dim=1) == target).detach().float().mean()

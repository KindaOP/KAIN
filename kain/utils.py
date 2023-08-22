import torch.nn as nn


def join_parameters(*models:nn.Module) -> list:
    result = []
    for m in models:
        result += list(m.parameters())
    return result
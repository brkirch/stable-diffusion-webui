import torch
from packaging import version

from modules import devices, shared
from modules.sd_hijack_utils import GenericHijack


def cat(self, tensors, *args, **kwargs):
    if len(tensors) == 2:
        a, b = tensors
        if a.shape[-2:] != b.shape[-2:]:
            a = torch.nn.functional.interpolate(a, b.shape[-2:], mode="nearest")

        tensors = (a, b)

    return torch.cat(tensors, *args, **kwargs)

# This is torch, but with cat that resizes tensors to appropriate dimensions if they do not match;
# this makes it possible to create pictures with dimensions that are multiples of 8 rather than 64
th = GenericHijack(torch, {'cat': cat})

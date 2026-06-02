from rl4co.envs import FJSPEnv
from tensordict import TensorDict
import torch
import torch.nn.functional as F


from typing import List, Tuple


def expand_matrix(x: torch.Tensor, shape_to_make: Tuple[int, int]) -> torch.Tensor:
    min_val = x.min()
    max_val = x.max()

    # Avoid divide by zero
    if max_val == min_val:
        x_norm = torch.zeros_like(x)   # or ones_like(x), depending on intent
    else:
        x_norm = (x - min_val) / (max_val - min_val)

    x_norm = x_norm.clamp(0, 1)

    # get height/width
    if x_norm.ndim == 2:
        h, w = x_norm.shape
    else:
        _, h, w = x_norm.shape

    # compute padding
    pad_bottom = shape_to_make[0] - h
    pad_right  = shape_to_make[1] - w

    # pad
    x_padded = F.pad(x_norm, (0, pad_right, 0, pad_bottom), value=0.0)

    return x_padded

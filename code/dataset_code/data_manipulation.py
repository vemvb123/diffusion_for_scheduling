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


# kan brukes hvis td ikke inneholder order fra før av
def make_ma_assignment_with_order(
        actions: List, td_unscheduled: TensorDict, env: FJSPEnv, order: bool
    )-> TensorDict:

    n_actions = len(actions)

    td_to_actions = td_unscheduled.copy()
    td_to_actions = td_to_actions.unsqueeze(0)
    prev_adj = td_to_actions["ma_assignment"].clone()

    # this stores the sequence / order matrix
    assignment_adj = torch.zeros_like(prev_adj, dtype=torch.float)

    per_row_counts = torch.zeros(prev_adj.size(0), dtype=torch.int)
    for i, action in enumerate(actions):
        td_to_actions["action"] = torch.tensor([action])
        td_to_actions = env.step(td_to_actions)["next"]

        new_adj = td_to_actions["ma_assignment"]

        diff = (new_adj == 1) & (prev_adj == 0)

        if diff.any():
            normalized_order = i / float(n_actions)
            assignment_adj[diff] = normalized_order
        prev_adj = new_adj.clone()


    td_to_actions["ma_assignment"] = assignment_adj
    return td_to_actions
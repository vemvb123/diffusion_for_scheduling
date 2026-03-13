from typing import List
import torch

from code.inference.inferenced_to_schedule import round_to_values, show_order_clear
import code.inference.inferenced_to_schedule
import code.inference.inferenced_to_schedule as utils
from torch import Tensor


# takes a batch of solutions over time, specifically from a specific generated solution
# shape: B, C, H, W
# B is the amount of timesteps that has been recorded for the solution 
# take a specific solution of a batch of generated solutions, pair it with generated solutions for that specific solution over time
import numpy as np

import torch
import numpy as np


import torch



def insert_column_gaps(matrix, job_lengths, gap_size=1):
    """
    Inserts NaN columns between job groups.
    """
    H, W = matrix.shape
    new_cols = []
    col_start = 0

    for i, length in enumerate(job_lengths):
        cols = matrix[:, col_start:col_start + length]
        new_cols.append(cols)

        col_start += length

        # insert gap except after last job
        if i < len(job_lengths) - 1:
            gap = np.full((H, gap_size), np.nan)
            new_cols.append(gap)

    return np.concatenate(new_cols, axis=1)



"""
# valid tensor: 1 for valide op-ma relasjoner, 0 for ikke.
# får datastructur fra "ops_ma_adj"
def round_to_values(x: Tensor, n_values: int, valid_slots: Tensor) -> torch.Tensor:
    orig_shape = x.shape

    # assume shape [1,1,4,16] or similar, so flatten leading dims
    flat = x.view(-1, x.shape[-2], x.shape[-1])  # [B, 4, 16]

    # find max in each column along row dim (dim=1)
    max_vals, _ = flat.max(dim=1, keepdim=True)  # [B, 1, 16]

    # compare with max and binarize
    mask = torch.isclose(flat, max_vals)  # True where value == max

    # convert to float (1.0/0.0)
    result = mask.float()

    # restore original leading dims
    result = result.view(orig_shape)

    return result
    # flatten and get top k indices
    topk_vals, topk_idx = torch.topk(x.flatten(), n_values)

    # start with all zeros
    y = torch.zeros_like(x).flatten()

    # set top entries to 1
    y[topk_idx] = 1.0

    # reshape back to original shape
    return y.view(x.shape)

    """

# første skedulerte har minst verdi, sist skedulerte har størst verdi
def check_when_inference_makes_final_schedule(assignments_over_time: List[Tensor], final_assignment: Tensor, order: bool, valid_slots: Tensor, valid_h, valid_w):
    for i, assignment_at_time in enumerate(assignments_over_time):
        assignment_at_time = assignment_at_time[:, :, :valid_h, :valid_w]
        if order:
            assignment_at_time = code.inference.inferenced_to_schedule.round_to_values(assignment_at_time, valid_w, valid_slots)
        else:
            assignment_at_time = utils.show_order_clear(assignment_at_time, valid_w, valid_slots)

        if torch.equal(assignment_at_time, final_assignment):
            print(f"assignments are exactly the same at point {i}")
            print(assignment_at_time)
            print(final_assignment)
            break



# x: b, c, h, w
# errors: [total errors in batch 0, total errors in batch 1, etc...]

def replace_batches_with_fittest(x, errors):
    """
    Replace all batch instances with the best ones (least errors).
    
    Args:
        x: tensor of shape [B, C, H, W]
        errors: list or 1D tensor of length B, number of errors per batch
    Returns:
        x_new: tensor of shape [B, C, H, W] with “bad” batches replaced
    """
    device = x.device
    B = x.shape[0]

    # Convert errors to tensor if needed
    errors = torch.tensor(errors, device=device)

    # Step 1: find the minimum error
    min_err = torch.min(errors)

    # Step 2: indices of all batches with minimal error
    best_indices = (errors == min_err).nonzero(as_tuple=False).squeeze(-1)

    # Step 3: pick one of the best batches randomly
    chosen_idx = best_indices[torch.randint(len(best_indices), (1,), device=device)].item()

    # Step 4: replace all bad instances with the chosen best instance
    # mask for bad instances
    bad_mask = (errors != min_err)

    # create output
    x_new = x.clone()
    x_new[bad_mask] = x[chosen_idx]

    return x_new


def count_duplicate_instances(x: torch.Tensor) -> int:
    """
    x: Tensor of shape (B, C, H, W)
    Returns: number of batch instances that are exactly identical
             to at least one other instance.
    """
    B = x.shape[0]

    # Flatten each instance to (B, N)
    flat = x.view(B, -1)

    # Compare all pairs: (B, B, N)
    equal_matrix = flat.unsqueeze(1) == flat.unsqueeze(0)

    # Reduce over N → (B, B): True if entire instance matches
    same_instance = equal_matrix.all(dim=2)

    # Ignore self-comparisons (diagonal)
    same_instance.fill_diagonal_(False)

    # An instance is duplicate if it matches ANY other instance
    is_duplicate = same_instance.any(dim=1)

    # Count how many instances are duplicates
    return int(is_duplicate.sum().item())

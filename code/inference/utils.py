from typing import List
import torch

import code.inference.utils as utils
from torch import Tensor


def round_to_values(
    x: torch.Tensor,
    n_values: int,
    valid_slots: torch.Tensor,
) -> torch.Tensor:
    """
    For each column:
    - If valid_slots has at least one valid entry, pick max among valid slots
    - If valid_slots is all zeros, ignore it and pick max among all values
    """

    x = x.to("cuda")
    valid_slots = valid_slots.to("cuda")

    orig_shape = x.shape

    # Flatten leading dims -> [B, H, W]
    flat_x = x.view(-1, x.shape[-2], x.shape[-1])
    flat_vs = valid_slots.view(-1, x.shape[-2], x.shape[-1])

    # Check if there is ANY valid slot per (batch, column)
    # shape: [B, 1, W]
    has_any_valid = flat_vs.any(dim=1, keepdim=True)

    # Mask invalid slots with -inf
    neg_inf = torch.finfo(flat_x.dtype).min
    masked_x = torch.where(flat_vs.bool(), flat_x, neg_inf)

    # Max among valid slots
    max_valid, _ = masked_x.max(dim=1, keepdim=True)

    # Max among all slots (fallback)
    max_all, _ = flat_x.max(dim=1, keepdim=True)

    # Choose which max to use:
    # - use valid max if any valid exists
    # - otherwise use full max
    max_vals = torch.where(has_any_valid, max_valid, max_all)

    # Build result mask
    mask = torch.isclose(flat_x, max_vals)

    # If valid slots exist, enforce them
    mask = torch.where(has_any_valid, mask & flat_vs.bool(), mask)

    result = mask.float().view(orig_shape)
 
    return result


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

def show_order_clear(x, n_values, valid_slots):
    """
    Rank the top-n_values entries.
    - If valid_slots has any 1s: only consider those positions
    - If valid_slots is all zeros: ignore valid_slots
    """

    device = x.device
    x = x.to(device)
    valid_slots = valid_slots.to(device)
    print("flats")
    print(x.shape)
    print(valid_slots.shape)


    # Flatten
    flat_x = x.flatten()
    flat_vs = valid_slots.flatten()
    # Check if there is any valid slot at all
    has_any_valid = flat_vs.any()

    if has_any_valid:
        # Mask invalid slots by setting them to -inf
        neg_inf = torch.finfo(flat_x.dtype).min
        masked_x = torch.where(flat_vs.bool(), flat_x, neg_inf)
    else:
        # Ignore valid_slots completely
        masked_x = flat_x

    # Get top-k values and indices
    topk_vals, topk_idx = torch.topk(masked_x, n_values)

    # Sort by value so smallest rank = 1, largest = n_values
    sorted_vals, sorted_order = torch.sort(topk_vals, descending=False)
    topk_idx_sorted = topk_idx[sorted_order]

    # Create output tensor
    out = torch.zeros_like(flat_x)

    # Assign ranks
    for rank, idx in enumerate(topk_idx_sorted, start=1):
        out[idx] = rank

    # Reshape back
    return out.view_as(x)




# første skedulerte har minst verdi, sist skedulerte har størst verdi
def check_when_inference_makes_final_schedule(assignments_over_time: List[Tensor], final_assignment: Tensor, order: bool, valid_slots: Tensor):
    for i, assignment_at_time in enumerate(assignments_over_time):
        assignment_at_time = assignment_at_time[:, :, :4, :16]
        if order:
            assignment_at_time = utils.round_to_values(assignment_at_time, 16, valid_slots)
        else:
            assignment_at_time = utils.show_order_clear(assignment_at_time, 16, valid_slots)

        if torch.equal(assignment_at_time, final_assignment):
            print(f"assignments are exactly the same at point {i}")
            print(assignment_at_time)
            print(final_assignment)
            break
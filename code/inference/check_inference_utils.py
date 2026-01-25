from typing import List
import torch

import code.inference.check_inference_utils as utils
from torch import Tensor


def round_to_values(x: torch.Tensor, n_values: int) -> torch.Tensor:
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
    """
    # flatten and get top k indices
    topk_vals, topk_idx = torch.topk(x.flatten(), n_values)

    # start with all zeros
    y = torch.zeros_like(x).flatten()

    # set top entries to 1
    y[topk_idx] = 1.0

    # reshape back to original shape
    return y.view(x.shape)

    """


def show_order_clear(x, n_values):

    # flatten all values
    flat = x.flatten()

    # find the top 16 values and their indices
    topk_vals, topk_idx = torch.topk(flat, n_values)

    # sort those top 16 in descending order so largest -> rank 1
    sorted_vals, sorted_order = torch.sort(topk_vals, descending=False)
    top16_idx_sorted = topk_idx[sorted_order]

    # create an output tensor of zeros
    out = torch.zeros_like(flat)

    # assign ranks 1..16 to those positions
    for rank, idx in enumerate(top16_idx_sorted, start=1):
        out[idx] = rank

    # reshape back to original
    return out.view_as(x)


# første skedulerte har minst verdi, sist skedulerte har størst verdi
def check_when_inference_makes_final_schedule(assignments_over_time: List[Tensor], final_assignment: Tensor, order: bool):
    for i, assignment_at_time in enumerate(assignments_over_time):
        assignment_at_time = assignment_at_time[:, :, :4, :16]
        if order:
            assignment_at_time = utils.round_to_values(assignment_at_time, 16)
        else:
            assignment_at_time = utils.show_order_clear(assignment_at_time, 16)

        if torch.equal(assignment_at_time, final_assignment):
            print(f"assignments are exactly the same at point {i}")
            print(assignment_at_time)
            print(final_assignment)
            break
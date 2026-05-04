


# Model inference gives a data representation of a schedule
# this module decodes that representation into a schedule, as well as various operations needed along the way to achieve that

import torch


def round_to_values(
    x: torch.Tensor,
    n_values: int,

    valid_slots: torch.Tensor,
) -> torch.Tensor:
    """

    For each column:
    - If valid_slots has at least one valid entry, pick max among valid slots
    - If valid_slots is all zeros, ignore it and pick max am
ong all values
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


def show_order_clear(x, n_values, valid_slots, r_global=True):
    print("in show order clear...")
    print(x.shape)
    print(n_values)
    print(valid_slots.shape)
    print(r_global)
    print("...")


    """
    Rank the top-n_values entries.
    - If valid_slots has any 1s: only consider those positions
    - If valid_slots is all zeros: ignore valid_slots
    """
    if r_global:

        device = x.device
        x = x.to(device)
        valid_slots = valid_slots.to(device)
        valid_slots = valid_slots.repeat(x.shape[0], 1, 1, 1)

        B = x.shape[0]

        # Flatten per batch: (B, N)
        flat_x = x.reshape(B, -1)
        flat_vs = valid_slots.reshape(B, -1)


        # Check if each batch item has any valid slots
        has_any_valid = flat_vs.any(dim=1, keepdim=True)  # (B, 1)

        flat_x = flat_x.float()
        neg_inf = torch.finfo(flat_x.dtype).min
        #neg_inf = torch.finfo(flat_x.dtype).min

        # Mask invalid slots ONLY if that batch item has any valid slots
        masked_x = torch.where(
            has_any_valid,
            torch.where(flat_vs.bool(), flat_x, neg_inf),
            flat_x
        )

        # Top-k per batch
        topk_vals, topk_idx = torch.topk(masked_x, n_values, dim=1)

        # Sort top-k values so smallest rank = 1
        _, sorted_order = torch.sort(topk_vals, dim=1, descending=False)
        topk_idx_sorted = torch.gather(topk_idx, 1, sorted_order)

        # Output tensor
        out = torch.zeros(
            flat_x.shape,
            device=flat_x.device,
            dtype=torch.int64
        )
        ranks = torch.arange(
            1, n_values + 1,
            device=flat_x.device,
            dtype=torch.int64
        ).view(1, -1).expand(B, -1)

        out.scatter_(1, topk_idx_sorted, ranks)

        return out.reshape_as(x)

    else:
        device = x.device

        x = x.float()  # convert to float for -inf masking

        B, C, H, W = x.shape

        # Setup valid_slots
        if valid_slots is None:
            valid_slots = torch.ones_like(x, dtype=torch.bool, device=device)
        else:
            valid_slots = valid_slots.to(device).bool()
            if valid_slots.shape[0] != B:
                valid_slots = valid_slots.repeat(B, 1, 1, 1)

        # Step 1: column-wise max
        neg_inf = torch.finfo(x.dtype).min
        # If any valid in column, mask invalid with -inf; else keep original
        has_valid = valid_slots.any(dim=2, keepdim=True)  # (B,C,1,W)
        masked_x = torch.where(has_valid, torch.where(valid_slots, x, neg_inf), x)

        # Argmax along height (H) per column
        max_idx = torch.argmax(masked_x, dim=2, keepdim=True)  # (B,C,1,W)

        # Create mask of column maxima
        col_max_mask = torch.zeros_like(x, dtype=torch.bool)
        col_max_mask.scatter_(2, max_idx, True)

        # Keep only maxima
        filtered_x = torch.where(col_max_mask, x, torch.zeros_like(x))

        # Step 2: global ranking
        flat_vals = filtered_x.flatten()
        mask = col_max_mask.flatten()  # only surviving values

        out_flat = torch.zeros_like(flat_vals, dtype=torch.int64)

        if mask.any():
            vals = flat_vals[mask]
            _, order = torch.sort(vals, descending=False)  # smallest = rank 1
            indices = mask.nonzero(as_tuple=True)[0]

            for rank, idx in enumerate(order, start=1):
                out_flat[indices[idx]] = rank

    return out_flat.view_as(x)



import torch

def fix_x(x, valid_slots):
    """
    x: tensor of shape (B, 1, H, W), float, on any device (CPU/GPU)
    valid_slots: tensor of shape (1, 1, H, W), binary (0/1), possibly on CPU

    Returns:
        fixed_x: same shape as x, only max valid per column kept
    """
    B, C, H, W = x.shape

    # Move valid_slots to the same device as x
    valid = valid_slots.to(x.device).expand(B, C, H, W)

    # Mask invalid positions
    masked = x.clone()
    masked[valid == 0] = float('-inf')

    # Get max per column (over height)
    col_max, _ = masked.max(dim=2, keepdim=True)  # shape (B,1,1,W)

    # Keep only max values, zero others
    fixed_x = torch.where(x == col_max, x, torch.zeros_like(x))

    # Ensure invalid positions are zero
    fixed_x = torch.where(valid == 1, fixed_x, torch.zeros_like(fixed_x))

    return fixed_x
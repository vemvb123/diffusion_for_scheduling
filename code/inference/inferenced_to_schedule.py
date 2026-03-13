


# Model inference gives a data representation of a schedule
# this module decodes that representation into a schedule, as well as various operations needed along the way to achieve that

import torch


def topk_binary_matrix(tensor: torch.Tensor, k: int, valid_slots: torch.Tensor):
    """
    tensor: shape (1, H, W)
    valid_slots: shape (1, 1, H, W) with 1s where allowed
    k: number of top values to select
    """
    # Make valid_slots the same shape as tensor
    mask = valid_slots[0, 0].bool()  # shape (H, W)

    # Flatten tensor and mask
    flat = tensor.reshape(-1)
    mask_flat = mask.reshape(-1)

    # Only consider valid positions
    valid_values = flat.clone()
    valid_values[~mask_flat] = float('-inf')  # ignore invalid slots

    # Take top-k among valid positions
    topk_indices = torch.topk(valid_values, k).indices

    # Create binary result
    result = torch.zeros_like(flat)
    result[topk_indices] = 1

    return result.reshape_as(tensor)


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

        neg_inf = torch.finfo(flat_x.dtype).min

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
        x = x.to(device)

        if valid_slots is not None:
            valid_slots = valid_slots.to(device)
        else:
            valid_slots = torch.ones_like(x)

        B, C, H, W = x.shape
        out = torch.zeros_like(x)

        # Step 1: column-wise max (keeping only one value per column)
        col_max_mask = torch.zeros_like(x, dtype=torch.bool)
        for b in range(B):
            for c in range(C):
                for w in range(W):
                    col_vals = x[b, c, :, w]
                    col_vs = valid_slots[b, c, :, w]

                    # Mask if needed
                    if col_vs.any():
                        masked = torch.where(col_vs.bool(), col_vals, torch.finfo(x.dtype).min)
                    else:
                        masked = col_vals

                    # Index of maximum in this column
                    max_idx = torch.argmax(masked)
                    col_max_mask[b, c, max_idx, w] = True

        # Keep only column maxima
        filtered_x = torch.where(col_max_mask, x, torch.zeros_like(x))

        # Step 2: global ranking of surviving values
        flat_vals = filtered_x.flatten()
        nonzero_mask = flat_vals != 0

        out_flat = torch.zeros_like(flat_vals)
        if nonzero_mask.any():
            vals_to_rank = flat_vals[nonzero_mask]
            sorted_vals, order = torch.sort(vals_to_rank, descending=False)  # smallest -> 1
            rank_indices = nonzero_mask.nonzero(as_tuple=True)[0]

            # Assign ranks 1..n_ops (or less if fewer maxima)
            for rank, idx in enumerate(order, start=1):
                # Cap rank at n_ops
                out_flat[rank_indices[idx]] = min(rank, n_values)

        return out_flat.view_as(x)
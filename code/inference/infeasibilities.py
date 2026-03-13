import torch

import code.inference.inferenced_to_schedule as utils
from code.inference.data_inform import get_job_lengths


def fix_scheduled_multiple_times(x: torch.Tensor, valid_slots: torch.Tensor) -> torch.Tensor:
    print(x.shape)
    print(valid_slots.shape)
    B, _, R, C = x.shape

    # 1) Get all non-zero values per sample
    mask_nonzero = x != 0
    vals_per_batch = []
    for i in range(B):
        vals = x[i][mask_nonzero[i]]                   # select only non-zero
        vals_per_batch.append(vals)

    # 2) Build empty output same shape
    out = torch.zeros_like(x)
    for i in range(B):
        valid_pos = valid_slots[0].nonzero(as_tuple=True)

        vals = vals_per_batch[i]
        n_vals = min(len(vals), valid_pos[0].shape[0])

        if n_vals > 0:
            r_inds = valid_pos[1][:n_vals]
            c_inds = valid_pos[2][:n_vals]
            out[i, 0, r_inds, c_inds] = vals[:n_vals]


    return out


def when_empty_move_largest(
    x: torch.Tensor,
    valid_slots: torch.Tensor,
    largest_values: torch.Tensor,
) -> torch.Tensor:

    B, _, H, W = x.shape

    # modify x
    out = x.clone().float()

    for b in range(B):
        for c in range(W):

            # if column has no largest value
            if (largest_values[b, 0, :, c] == 0).all():

                col_vals = x[b, 0, :, c].float()
                max_val = torch.max(col_vals)

                valid_rows = torch.nonzero(valid_slots[0, 0, :, c]).squeeze(1)

                if valid_rows.numel() > 0:
                    r = valid_rows[0]

                    # place the max value into x
                    out[b, 0, r, c] = max_val

    return out

# when a column has no largest value, move the largest value in the column to a valid position
# largest values has non-zero for the n largest values
# valid_slots is binary, 1 for valid slots. largest_values only has values inside valid slots
def keep_largest_when_scheduled_twice(x, valid_slots, largest_values):

    device = x.device
    valid_slots = valid_slots.to(device)

    B, _, H, W = x.shape
    out = x.clone()

    for b in range(B):
        for c in range(W):

            # rows that are valid and marked as largest
            mask = (valid_slots[0, 0, :, c] == 1) & (largest_values[b, 0, :, c] > 0)
            rows = torch.nonzero(mask, as_tuple=True)[0]

            if rows.numel() > 1:

                # use x values to decide which one to keep
                vals = x[b, 0, rows, c]
                keep_idx = torch.argmax(vals)
                keep_row = rows[keep_idx]

                # zero the others
                remove_rows = rows[rows != keep_row]
                out[b, 0, remove_rows, c] = 0

    return out


def when_break_pred_switch_order(x, valid_slots, largest_values, job_lengths):
    """
    Incrementally fix predecessor violations in each job group.
    For each column in the job:
      - get the cell(s) where largest_values > 0
      - if that value > the next column's largest-value cell, swap the two
    """
    device = x.device
    valid_slots = valid_slots.to(device)
    out = x.clone().float()

    B, _, H, W = x.shape
    start_col = 0

    for group_len in job_lengths:
        end_col = start_col + group_len

        for b in range(B):
            for c in range(start_col, end_col - 1):
                # find row of largest in this column
                mask_current = largest_values[b, 0, :, c] > 0
                if mask_current.any():
                    row_current = torch.nonzero(mask_current, as_tuple=True)[0][0]
                    val_current = out[b, 0, row_current, c]

                    # find row of largest in next column
                    mask_next = largest_values[b, 0, :, c + 1] > 0
                    if mask_next.any():
                        row_next = torch.nonzero(mask_next, as_tuple=True)[0][0]
                        val_next = out[b, 0, row_next, c + 1]

                        # swap if violation
                        if val_current > val_next:
                            out[b, 0, row_current, c] = val_next
                            out[b, 0, row_next, c + 1] = val_current

        start_col = end_col

    return out


def make_value_before_slightly_lower(x, valid_slots, largest_values, epsilon=0.01):
    """
    For columns with no largest value, replace the current largest value in the column
    with slightly less than the largest value in the next column.
    """
    device = x.device
    valid_slots = valid_slots.to(device)
    out = x.clone().float()
    B, _, H, W = x.shape

    for b in range(B):
        for c in range(W - 1):  # skip last column for now
            # check if current column has no largest values
            if (largest_values[b, 0, :, c] == 0).all():
                # find largest value in current column
                current_col = out[b, 0, :, c]
                max_val, max_row = torch.max(current_col, dim=0)

                # find largest value in next column
                next_max = torch.max(out[b, 0, :, c + 1])

                # replace largest value in current column
                current_col[max_row] = next_max - epsilon

        # last column (optional)
        if (largest_values[b, 0, :, W - 1] == 0).all():
            current_col = out[b, 0, :, W - 1]
            max_val, max_row = torch.max(current_col, dim=0)
            prev_max = torch.max(out[b, 0, :, W - 2])
            current_col[max_row] = prev_max - epsilon

    return out


def when_pred_same_value_make_not_same(x, largest_values, job_lengths, epsilon=0.01):
    """
    For each job, if a column's largest value equals the next column's largest value,
    increase the next column's value by epsilon.
    """
    device = x.device
    out = x.clone().float()
    B, _, H, W = x.shape

    start_col = 0
    for group_len in job_lengths:
        end_col = start_col + group_len

        for b in range(B):
            for c in range(start_col, end_col - 1):
                # find largest-value row in current column
                mask_current = largest_values[b, 0, :, c] > 0
                mask_next = largest_values[b, 0, :, c + 1] > 0

                if mask_current.any() and mask_next.any():
                    row_current = torch.nonzero(mask_current, as_tuple=True)[0][0]
                    row_next = torch.nonzero(mask_next, as_tuple=True)[0][0]

                    val_current = out[b, 0, row_current, c]
                    val_next = out[b, 0, row_next, c + 1]

                    # if same, increase next column's value
                    if torch.isclose(val_current, val_next):
                        out[b, 0, row_next, c + 1] = val_next + epsilon

        start_col = end_col

    return out


def fix_infeasibilities(x: torch.Tensor, valid_slots: torch.Tensor, ops_sequence_order: torch.Tensor, largest_values: torch.Tensor, n_ops: int,
                        huh=False) -> torch.Tensor:
    print("in fix")
    x = when_empty_move_largest(x, valid_slots, largest_values)
    x = keep_largest_when_scheduled_twice(x, valid_slots, largest_values)

    # the two functions above fix absolute most cases where no value in a column
    # this is more of a hard fix for the very few cases when the above dosent work
    largest_values = utils.show_order_clear(x, n_ops, valid_slots)
    x = make_value_before_slightly_lower(x, valid_slots, largest_values)

    job_lengths = get_job_lengths(ops_sequence_order)
    for i in range(2):
        x = when_break_pred_switch_order(x, valid_slots, largest_values, job_lengths)
        x = when_pred_same_value_make_not_same(x, largest_values, job_lengths) # this fixes the most pred wrongs
    x = when_break_pred_switch_order(x, valid_slots, largest_values, job_lengths)

    return x
import torch

import code.inference.inferenced_to_schedule
import code.inference.report_infeasibilities as utils

# example ... Error for how much lastlast  Machine is used
# assumes no order given in schedule
def guiding_function(x): #  is batch of instances
    # x has shape [1, 1, 20, 20]
    # extract the 4×16 region
    x_inside = x[:, :, :4, :16]   # shape [1, 1, 4, 16]

    # select the bottom row of that 4×16 → index 3
    bottom_row = x_inside[:, :, 3, :]  # shape [1, 1, 16]

    # compute mean absolute value of bottom row
    error = torch.abs(bottom_row).mean()

    return error


import torch.nn.functional as F


def increasing_columns_loss(x, sections, valid_h, valid_w):
    x = x[:, :, :valid_h, :valid_w]

    col_sums = x.sum(dim=1).sum(dim=1)

    losses = []
    start = 0

    for i, section_len in enumerate(sections):
        end = start + section_len

        # sanity check (optional but helpful)
        if end > col_sums.shape[1]:
            raise ValueError(f"Section {i} exceeds valid_w")

        if section_len >= 2:
            sec = col_sums[:, start:end]
            diffs = sec[:, :-1] - sec[:, 1:]
            penalty = torch.relu(diffs)
            losses.append(penalty.mean())

        start = end

    if len(losses) == 0:
        # all sections were fillers
        return torch.zeros((), device=x.device, requires_grad=True)

    return torch.stack(losses).mean()



def loss_single_ma(x, valid_h, valid_w, top_k=55):
    # Remove padding
    x = x[:, :, :valid_h, :valid_w]   # (b, 1, h, w)
    b, _, h, w = x.shape

    # Flatten safely
    flat = x.reshape(b, -1)

    # Top-k values
    top_vals, top_idx = torch.topk(flat, k=top_k, dim=1)

    # Convert flat indices → column indices
    col_idx = top_idx % w

    # One-hot encode columns
    col_onehot = F.one_hot(col_idx, num_classes=w).float()

    # Count collisions per column
    col_counts = col_onehot.sum(dim=1)

    # Penalize collisions
    collision_penalty = F.relu(col_counts - 1.0) ** 2

    return collision_penalty.mean()



# error is how much lower some value in front of predecessor us
# job_lengths is a list like [3, 4, 3, 6] .. tells the length of each job
def minimize_infeasibility(x, valid_h, valid_w, job_lengths):
    """
    Loss that penalizes “infeasible” column order where a column has a lower
    max activation than its predecessor within each job group.

    x: (B, C, H, W) raw model output
    valid_h: height without padding
    valid_w: width without padding
    job_lengths: list of ints like [3,4,3,6] describing column group sizes
    """

    # Crop padding
    x = x[:, :, :valid_h, :valid_w]

    # We want to compute max activations per column (remove batch & channel)
    # Shape: (batch, channels, height, cols)
    # We reduce height & channels to a single representative value per column
    # Use max over height and channels:
    # shape → (batch, cols)
    col_max = x.amax(dim=2).amax(dim=1)  # max over height then channel

    # Now col_max[b, col_index] gives max activation for each column

    start_col = 0
    loss_terms = []

    # Loop job groups
    for length in job_lengths:
        group_end = start_col + length

        # Extract this group
        group_vals = col_max[:, start_col:group_end]  # shape: (B, length)

        # Make pairwise comparisons
        # For every consecutive pair (prev, next) apply relu(prev - next)
        if group_vals.shape[1] > 1:  # only if at least 2 columns
            prev_vals = group_vals[:, :-1]  # all except last
            next_vals = group_vals[:, 1:]   # all except first

            # Penalty where next < prev
            diff = torch.relu(prev_vals - next_vals)

            # Average penalty in this group
            loss_terms.append(diff.mean())

        # Move to next group
        start_col = group_end

    # If no losses were generated (e.g., empty job_lengths), just zero
    if not loss_terms:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    # Mean over all job groups
    loss = torch.stack(loss_terms).mean()

    return loss












# ma to use more is a list of row indexes
def use_ma_more(x, valid_h, valid_w, ma_to_use, margin=0.1):
    """
    Loss that *pushes the specified rows (ma_to_use)* to have higher values
    than the other rows.

    x: (B, C, H, W) tensor
    valid_h: height excluding padding
    valid_w: width excluding padding
    ma_to_use: list of row indices from bottom (0 = bottom)
    margin: how much larger selected rows should be
    """
    # Remove padding
    x = x[:, :, :valid_h, :valid_w]

    # Calculate absolute row indices
    selected_inds = []
    for r in ma_to_use:
        if 0 <= r < valid_h:
            selected_inds.append(valid_h - 1 - r)

    # If no rows selected → zero loss
    if len(selected_inds) == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    # Gather selected rows
    selected = x[:, :, selected_inds, :]  # (B, C, len(selected), W)
    other_inds = [
        i for i in range(valid_h) if i not in selected_inds
    ]
    if len(other_inds) == 0:
        # If every row is selected → no comparison → zero loss
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    other = x[:, :, other_inds, :]      # (B, C, len(other), W)

    # Mean activations
    # (B,C) → take mean over rows & width
    sel_mean = selected.mean(dim=[2, 3])
    oth_mean = other.mean(dim=[2, 3])

    # Compute margin loss
    # We want sel_mean > oth_mean by margin
    loss_matrix = torch.relu(oth_mean - sel_mean + margin)

    # Mean over batch & channels
    loss = loss_matrix.mean()
    return loss


# ma to use less is a list of row indexes
def use_ma_less(x, valid_h, valid_w, ma_to_use_less):
    """
    Loss on specified rows counted from the bottom (0 = bottom row).

    x: (B, C, H, W) - raw model output
    valid_h: height without padding
    valid_w: width without padding
    ma_to_use_less: list of rows from bottom, with 0 = bottom
    """
    # 1) Remove padding
    x = x[:, :, :valid_h, :valid_w]

    # 2) Convert row indices from bottom to absolute indices
    # valid_h-1 = bottom, valid_h-2 = one above, etc.
    rows = []
    for r in ma_to_use_less:
        if 0 <= r < valid_h:
            abs_idx = valid_h - 1 - r
            rows.append(abs_idx)

    if len(rows) == 0:
        # No rows specified → zero loss
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    # 3) Gather selected rows
    # Shape: (batch, channels, len(rows), width)
    selected = x[:, :, rows, :]

    # 4) Loss: how far those rows are from empty (zero)
    loss = torch.abs(selected).mean()

    return loss





def amt_errors(x, n_ops, ops_ma_adj, ops_seq_order, valid_h, valid_w, max_allowed_total_errors=30):
    # remove padding
    x = x[:, :, :valid_h, :valid_w]
    ops_ma_adj = ops_ma_adj[:, :, :valid_h, :valid_w]
    #print("hvata")
    #print(x.shape)
    #print(ops_ma_adj.shape)
    #print(ops_seq_order.shape)
    #print("---")
    # make in schedule with order

    #inference_assignments = utils.show_order_clear(inference_assignments, n_ops, ops_ma_adj)

    inference_assignments = code.inference.inferenced_to_schedule.show_order_clear(x, n_ops, ops_ma_adj)
    # get amount of errors
    report, total_errors, error_list = utils.count_infeasibilities(inference_assignments, ops_seq_order, do_print=False)
    # sammenligner med max mengde tillate feil
    error = max_allowed_total_errors - total_errors
    # hvis flere enn max_allowed_total_errors, er error 1
    if error <= total_errors:
        print("is total")
        return torch.tensor(1.0, device=x.device, dtype=x.dtype, requires_grad=True)
    # normaliserer totale feil mot max_allowed_total_errors
    norm_error = (error) / (max_allowed_total_errors)
    # returnerer error

    # Convert float to tensor if needed
    if not isinstance(norm_error, torch.Tensor):
        print("is not total")
        norm_error = torch.tensor(norm_error, device=x.device, dtype=x.dtype, requires_grad=True)

    return norm_error




# få til clear order, men kan også ta binær, binær er sikkert raskere
# map assignments til proctid
# summer proc tider for hver maskin
# ta loss som forskjell i proc tider: 
# beste tilfelle: (alle summer, som en sum) / mengde maskiner
# faktisk tilfelle: SUM( abs(sumN - beste) )
# normaliser: beste tilfelle, værste tilfelle (beste+beste/2)


def similair_MU(x, h, w, n_ops, ops_ma_adj, proc_times, n_ma):
    """
    Machine-utilization similarity loss.
    Lower is better (0 = perfectly balanced).
    """

    # crop to active region
    x = x[:, :, :h, :w]
    proc_times = proc_times[:, :, :h, :w]
    # binary assignment matrix
    x_used_assignments = code.inference.inferenced_to_schedule.round_to_values(x, w, ops_ma_adj)
    # map processing times via assignment
    valid_proc_times = x_used_assignments * proc_times

    # sum processing times per machine
    proc_time_per_machine = valid_proc_times.sum(dim=-1)  # (B, 1, n_ma)
    # total processing time
    total_proc_time = proc_time_per_machine.sum(dim=-1, keepdim=True)  # (B, 1, 1)

    # best possible balanced load
    best_case = total_proc_time / n_ma  # (B, 1, 1)
    # actual deviation from balance
    deviation = torch.abs(proc_time_per_machine - best_case)
    loss_raw = deviation.sum(dim=-1)  # (B, 1)
    # worst reasonable case
    worst_case = best_case * 1.5  # (B, 1, 1)

    # normalized loss
    loss_norm = loss_raw / (worst_case.squeeze(-1) + 1e-8)

    return loss_norm.squeeze(-1)





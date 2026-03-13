import torch


def get_clear_sequence(assignments):
    """
    Given a tensor `assignments` with values between 0 and 1,
    return a tensor of the same shape where:
      - the largest value gets 16,
      - the 2nd largest gets 15,
      - ...
      - the 16th largest gets 1,
      - all others get 0.
    """

    flat = assignments.flatten()
    # get indices of the top 16 values
    top_vals, top_idx = torch.topk(flat, 16)

    # output tensor initialized with zeros
    out = torch.zeros_like(flat, dtype=torch.long)

    # assign values 16 → 1
    for rank, idx in enumerate(top_idx):
        out[idx] = 16 - rank

    return out.view(assignments.shape)


def map_assignemnts_to_actions(assignments, order: bool, n_jobs: int):

    # remove batch/channel dims if present
    if assignments.dim() > 2:
        assignments = assignments.squeeze(0).squeeze(0)  # (H, W)

    H, W = assignments.shape  # e.g. H=4, W=16
    cols_per_section = n_jobs
    num_sections = W // cols_per_section

    actions = []
    machine_assigned_to = []

    if order:
        # ---- ORDERED MODE ----

        # Get all non-zero positions
        nonzero = torch.nonzero(assignments, as_tuple=False)

        # Extract their values
        values = assignments[nonzero[:, 0], nonzero[:, 1]]

        # Sort by value (1 -> 2 -> 3 -> ...)
        sorted_idx = torch.argsort(values)

        for idx in sorted_idx:
            row = nonzero[idx, 0].item()
            col = nonzero[idx, 1].item()

            section_idx = col // cols_per_section
            action = section_idx * cols_per_section + (row +1)
            actions.append(action)



    else:
        # ---- BINARY MODE ----
        for col in range(W):
            section_idx = col // cols_per_section
            for row in range(H):
                if assignments[row, col] == 1:
                    action = section_idx * cols_per_section + (row + 1)
                    actions.append(action)


    return actions




def map_operation_to_machines(ma_assignment):
    # if there’s a leading batch dim, remove it
    if ma_assignment.dim() == 3:
        ma_assignment = ma_assignment.squeeze(0)

    op_to_ma = []
    num_rows, num_cols = ma_assignment.shape

    for col in range(num_cols):
        # find rows where value == 1
        rows = (ma_assignment[:, col] == 1).nonzero(as_tuple=False)

        if rows.shape[0] == 0:
            # no 1 in this column → filler
            op_to_ma.append(99)
        else:
            # take the first row index with a 1
            op_to_ma.append(rows[0, 0].item())

    return op_to_ma


from collections import defaultdict
import math



import torch

import torch


def map_assignments_to_actions_per_machine(assignments, order: bool, n_jobs):
    if assignments.dim() > 2:
        assignments = assignments.squeeze(0)

    H, W = assignments.shape

    # Determine widths for each machine group if n_jobs is list or int
    if isinstance(n_jobs, int):
        widths = [n_jobs] * (W // n_jobs)
    else:
        widths = n_jobs

    stride = max(widths)
    section_starts = torch.cumsum(
        torch.tensor([0] + widths[:-1]), dim=0
    )

    # Prepare one list per machine
    per_machine_actions = [[] for _ in range(H)]

    if order:
        nonzero = torch.nonzero(assignments, as_tuple=False)
        values = assignments[nonzero[:, 0], nonzero[:, 1]]
        sorted_idx = torch.argsort(values)

        for i in sorted_idx:
            row, col = nonzero[i].tolist()
            section_idx = int((section_starts <= col).sum() - 1)
            action = section_idx * stride + (row + 1)
            per_machine_actions[row].append(action)

    else:
        for row in range(H):
            for col in range(W):
                if assignments[row, col] == 1:
                    section_idx = int((section_starts <= col).sum() - 1)
                    action = section_idx * stride + (row + 1)
                    per_machine_actions[row].append(action)

    return per_machine_actions


def map_assignments_to_actions_text(assignments, order: bool, n_jobs):
    if assignments.dim() > 2:
        assignments = assignments.squeeze(0) # .squeeze(0)

    H, W = assignments.shape

    if isinstance(n_jobs, int):
        widths = [n_jobs] * (W // n_jobs)
    else:
        widths = n_jobs

    stride = max(widths)
    section_starts = torch.cumsum(
        torch.tensor([0] + widths[:-1]), dim=0
    )

    actions = []

    if order:
        nonzero = torch.nonzero(assignments, as_tuple=False)
        values = assignments[nonzero[:, 0], nonzero[:, 1]]
        sorted_idx = torch.argsort(values)

        for i in sorted_idx:
            row, col = nonzero[i].tolist()
            section_idx = int((section_starts <= col).sum() - 1)
            action = section_idx * stride + (row + 1)
            actions.append(action)

    else:
        for col in range(W):
            section_idx = int((section_starts <= col).sum() - 1)
            for row in range(H):
                if assignments[row, col] == 1:
                    action = section_idx * stride + (row + 1)
                    actions.append(action)

    # ✅ correct padding
    max_len = sum(widths)
    actions += [0] * (max_len - len(actions))

    return torch.tensor(actions, dtype=torch.int64)


def zero_only_columns(x):
    # (if on CUDA) bring it to CPU for processing
    x_cpu = x.cpu()

    # remove leading batch dimension if present
    x2 = x_cpu.squeeze(0)  # now shape is [num_rows, num_columns]

    # find columns where all rows are zero
    zero_cols = (x2 == 0).all(dim=0).nonzero(as_tuple=True)[0].tolist()
    return zero_cols


def extract_env_actions(ma_seq_matrix: torch.Tensor,
                        ops_sequence_order: torch.Tensor,
                        n_jobs: int,
                        max_ops_per_job: int) -> torch.Tensor:

    """
    Reconstruct the RL4CO FJSP action sequence from a completed schedule matrix.

    Args:
        ma_seq_matrix: Tensor of shape (1,1,n_machines,n_cols) with schedule ranks.
        ops_sequence_order: Tensor of length n_cols, gives op index within job.
        n_jobs: number of jobs
        max_ops_per_job: maximum operations per job

    Returns:
        LongTensor: shape (total_scheduled_ops,) with action indices in env format.
    """
    # Squeeze out batch dims -> shape (n_machines, n_cols)
    ma = ma_seq_matrix.squeeze(0).squeeze(0)
    n_machines, n_cols = ma.shape

    # We'll collect (rank, job, op_idx, machine)
    schedule_entries = []

    for m in range(n_machines):
        for c in range(n_cols):
            rank = int(ma[m, c].item())
            if rank > 0:
                op_idx_in_job = int(ops_sequence_order[c].item())
                job_id = c // max_ops_per_job
                schedule_entries.append((rank, job_id, op_idx_in_job, m))

    # Sort entries by global schedule rank ascending,
    # and tie-break by machine index ascending (as per your mapping rule).
    schedule_entries.sort(key=lambda x: (x[0], x[3]))

    # Convert entries to RL4CO env action IDs
    # RL4CO env actions are flat IDs where:
    # action_id = machine * n_jobs + job_id
    action_seq = []
    for rank, job_id, op_idx_in_job, machine in schedule_entries:
        # Compute the flat action index
        action_id = machine * n_jobs + job_id
        action_seq.append(action_id)

    return torch.tensor(action_seq, dtype=torch.long)

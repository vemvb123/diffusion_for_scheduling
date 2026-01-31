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

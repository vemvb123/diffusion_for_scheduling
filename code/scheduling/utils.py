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


# brukes til å skedulere fra inference
def map_assignemnts_to_actions(assignments, order: bool, n_jobs: int):

    # remove batch/channel dims if present
    if assignments.dim() == n_jobs:
        assignments = assignments.squeeze(0).squeeze(0)  # (4,16)

    H, W = assignments.shape  # H=4, W=16
    section_width = n_jobs # tror skal være samme verdi som mengde jobber
    num_sections = W // section_width
    """
    actions = []
    if order:
        x = assignments

        x = x.squeeze(0).squeeze(0)

        # get all nonzero positions
        rows, cols = torch.nonzero(x, as_tuple=True)

        # get the values at those positions
        vals = x[rows, cols]

        # sort by value (1 → 16)
        order = torch.argsort(vals)
        rows = rows[order]
        cols = cols[order]

        # compute mapped values
        sections = cols // 4
        mapped = sections * 4 + (rows + 1)

        return mapped.tolist()


    # TODO inkluder dette igjen i funksjonen seinere
    """
    if order:
        # order by largest value first
        _, indices = torch.topk(assignments.flatten(), H * W)

        for idx in indices:
            row = idx // W
            col = idx % W
            section = col // section_width
            action = section * H + row + 1
            actions.append(action.item())

    else:
        cols_per_section = n_jobs
        actions = []

        for col in range(assignments.shape[1]):
            section_idx = col // cols_per_section
            for row in range(assignments.shape[0]):
                if assignments[row, col] == 1:
                    value = section_idx * cols_per_section + (row + 1)
                    actions.append(value)

        return actions
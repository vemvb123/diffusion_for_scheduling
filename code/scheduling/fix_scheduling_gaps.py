import numpy as np
import torch


def compress_schedule(start_times, finish_times, machines, job_lengths, td, filler_machine=99):
    """
    Compress schedule by removing unnecessary gaps.
    Works with TensorDict batch dimension (1, N) or plain arrays.
    """

    # -------------------------------------------------
    # 0. Convert to flat numpy arrays
    # -------------------------------------------------
    start = np.asarray(start_times).reshape(-1).astype(float).copy()
    finish = np.asarray(finish_times).reshape(-1).astype(float).copy()
    machines = np.asarray(machines).reshape(-1)

    n = len(start)
    duration = finish - start

    # -------------------------------------------------
    # 1. Build predecessor array from job_lengths
    # -------------------------------------------------
    pred = np.full(n, -1, dtype=int)

    idx = 0
    for job_len in job_lengths:
        for j in range(job_len):
            if idx >= n:
                break
            if j > 0:
                pred[idx] = idx - 1
            idx += 1

    # -------------------------------------------------
    # 2. Machine → operation mapping
    # -------------------------------------------------
    machine_ops = {}
    for i in range(n):
        if machines[i] == filler_machine:
            continue
        machine_ops.setdefault(machines[i], []).append(i)

    # Sort operations on each machine by start time
    for m in machine_ops:
        machine_ops[m].sort(key=lambda i: start[i])

    # -------------------------------------------------
    # 3. Left-shift compression
    # -------------------------------------------------
    changed = True
    while changed:
        changed = False

        for m in machine_ops:
            machine_time = 0

            for op in machine_ops[m]:
                pred_finish = finish[pred[op]] if pred[op] != -1 else 0
                earliest_start = max(machine_time, pred_finish)

                if start[op] > earliest_start:
                    start[op] = earliest_start
                    finish[op] = start[op] + duration[op]
                    changed = True

                machine_time = finish[op]

    # -------------------------------------------------
    # 4. Restore batch dimension (1, N) as Tensor
    # -------------------------------------------------
    start_tensor = torch.tensor(start, dtype=torch.float32).unsqueeze(0)
    finish_tensor = torch.tensor(finish, dtype=torch.float32).unsqueeze(0)

    td["start_times"] = start_tensor
    td["finish_times"] = finish_tensor

    return td
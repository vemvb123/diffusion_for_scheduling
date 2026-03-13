# takes a data representation, and provides a readable representation 




import torch


from typing import List


def get_job_lengths(ops_sequence_order: torch.Tensor) -> List[int]:
    flat = ops_sequence_order.flatten()
    vals = flat.tolist()

    job_lengths = []
    current_len = 0
    expected_next = 0

    for v in vals:
        if v == expected_next:
            current_len += 1
            expected_next += 1
        else:
            # run ended → store and start new run
            job_lengths.append(current_len)
            current_len = 1
            expected_next = v + 1

    # append the final run
    if current_len > 0:
        job_lengths.append(current_len)

    return job_lengths


def infer_n_jobs(ops_sequence_order: torch.Tensor):
    n_jobs = []
    count = 1

    for i in range(1, len(ops_sequence_order)):
        # if sequence continues (0→1→2→...)
        if ops_sequence_order[i] == ops_sequence_order[i - 1] + 1:
            count += 1
        else:
            n_jobs.append(count)
            count = 1

    # append last job
    n_jobs.append(count)

    return n_jobs
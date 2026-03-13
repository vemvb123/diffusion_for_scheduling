from typing import Dict


def get_rl4co_parameters_from_brandimarte_instance(filepath_brandimarte_instance: str) -> Dict:

    with open(filepath_brandimarte_instance, "r") as f:
        lines = [line.strip() for line in f if line.strip()]


    # First line: number of jobs, number of machines
    first = lines[0].split()
    n_jobs = int(first[0])
    n_machines = int(first[1])

    # Stats
    global_min_pt = float("inf")
    global_max_pt = float("-inf")
    min_ops = float("inf")
    max_ops = float("-inf")

    # New stats for machine options per operation
    min_machine_options = float("inf")
    max_machine_options = float("-inf")

    # Loop through job lines
    for i in range(1, 1 + n_jobs):
        parts = list(map(int, lines[i].split()))
        idx = 0

        # Number of operations in this job
        n_ops = parts[idx]
        idx += 1

        # Update min/max number of operations
        min_ops = min(min_ops, n_ops)
        max_ops = max(max_ops, n_ops)

        # Loop through each operation
        for _ in range(n_ops):
            m_count = parts[idx]
            idx += 1

            # Track machine options stats
            min_machine_options = min(min_machine_options, m_count)
            max_machine_options = max(max_machine_options, m_count)

            # m_count pairs of (machine, processing time)
            for _ in range(m_count):
                machine_id = parts[idx]        # machine index (not needed for stats)
                proc_time = parts[idx + 1]     # processing time
                idx += 2

                # Track processing time
                global_min_pt = min(global_min_pt, proc_time)
                global_max_pt = max(global_max_pt, proc_time)


    return {
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "min_processing_time": global_min_pt,
        "max_processing_time": global_max_pt,
        "fewest_operations": min_ops,
        "most_operations": max_ops,
        "min_machine_options": min_machine_options,
        "max_machine_options": max_machine_options
    }
"""
dataset.py contains code for making a dataset of instances
"""

import logging
import os
import torch
from torch import Tensor

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)


import code.scheduling.schedule as schedule

import bisect

from tensordict import TensorDict, from_dict
from typing import Dict, Tuple

import torch.nn.functional as F

import time







def expand_matrix(x: torch.Tensor, shape_to_make: Tuple[int, int]) -> torch.Tensor:
    min_val = x.min()
    max_val = x.max()

    # Avoid divide by zero
    if max_val == min_val:
        x_norm = torch.zeros_like(x)   # or ones_like(x), depending on intent
    else:
        x_norm = (x - min_val) / (max_val - min_val)

    x_norm = x_norm.clamp(0, 1)

    # get height/width
    if x_norm.ndim == 2:
        h, w = x_norm.shape
    else:
        _, h, w = x_norm.shape

    # compute padding
    pad_bottom = shape_to_make[0] - h
    pad_right  = shape_to_make[1] - w

    # pad
    x_padded = F.pad(x_norm, (0, pad_right, 0, pad_bottom), value=0.0)

    return x_padded





def get_feature_adj_from_instance(td: TensorDict, env, order: bool, h: int, w: int) -> Tuple[
        torch.Tensor, # target assignments
        torch.Tensor, # proc times matrix
        torch.Tensor, # jobid matrix
        torch.Tensor, # pos in job matrix
        ]:

    # ASSIGNMENT
    assignments = None
    if order:
        assignments = td["opt_assignment_order"]
        assignments = assignments.unsqueeze(0)
    else:
        assignments = td['opt_assignment']
        assignments = assignments.unsqueeze(0)

    assignments = expand_matrix(assignments, (w, h))
    #logging.info(assignments)

    # PROC TIMES
    proc_times = td['proc_times']
    #logging.info(proc_times)
    proc_times = proc_times.unsqueeze(0)
    proc_times = expand_matrix(proc_times, (w, h))
    #logging.info(proc_times)

    # JOB OPS ADJ
    job_ops_adj = td['job_ops_adj']
    #logging.info(job_ops_adj)
    job_ops_adj = job_ops_adj.unsqueeze(0)
    job_ops_adj = expand_matrix(job_ops_adj, (w, h))
    #logging.info(job_ops_adj)

    # OPS MA ADJ
    ops_ma_adj = td['ops_ma_adj']
    ops_ma_adj = ops_ma_adj.unsqueeze(0)
    ops_ma_adj = expand_matrix(ops_ma_adj, (w, h))
    return assignments, proc_times, job_ops_adj, ops_ma_adj



def print_info_about_dataset(td: TensorDict):
    print(td["opt_assignment_order"].shape)
    print(td["opt_assignment"].shape)
    print(td["opt_assignment"][0])
    print(td["opt_assignment_order"][0])



def make_dataset(n: int, dataset_folder: str, 
                 n_jobs, n_ma, max_op_per_job, min_op_per_job, max_proc_time, min_proc_time, max_eligable_ma_per_op, min_eligable_ma_per_op,
                 target_model: str, order: bool, batch_size: int = 1280):

    logging.info("Making dataset...")

    os.makedirs(dataset_folder, exist_ok=True)
    for i in range(0, n, batch_size):
        # lag instanse
        env, td, generator_params = schedule.make_instance(
            n_ma=n_ma, n_jobs=n_jobs, 
            max_op_per_job=max_op_per_job, 
            min_op_per_job=min_op_per_job, 
            max_proc_time=max_proc_time, 
            min_proc_time=min_proc_time, 
            max_eligable_ma_per_op=max_eligable_ma_per_op, 
            min_eligable_ma_per_op=min_eligable_ma_per_op, 
            batch_size=batch_size)
        # fa target fra instance

        td_target, actions, ordered_assignments = schedule.make_target(env, td.copy(), target_model, order)

        # lagre json med: td, og optimale td koords
        td.set('opt_assignment', td_target['ma_assignment'])
        td.set('opt_actions', torch.tensor(actions))
        if order:
            td.set('opt_assignment_order', ordered_assignments)

        # TODO hvis du vil sjekke data, for testring
        #print_info_about_dataset(td)
        #exit()

        torch.save(td.copy(), f'{dataset_folder}/{i}_{i+batch_size}.pt')

        print(f'Made instance {i} to {i+batch_size}')

    print(f'Made all {i+batch_size} instances. Done making dataset')









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





def get_td_from_path(path: str, instance_idx: int) -> Tensor:
    # Get sorted list of data files
    files = sorted([f for f in os.listdir(path) if f.endswith(".pt")])
    # Build file ranges
    cum_sizes = []
    ranges = []
    total = 0

    for fname in files:
        start, end = map(int, fname.replace(".pt", "").split("_"))
        size = end - start
        cum_sizes.append(total)
        ranges.append((start, end))
        total += size

    # Find which file contains the instance
    file_idx = bisect.bisect_right(cum_sizes, instance_idx) - 1
    if file_idx < 0:
        raise ValueError(f"Instance {instance_idx} not found in {path}")

    file_path = os.path.join(path, files[file_idx])

    # Load with weights_only=False so that TensorDict objects (or other custom objects)
    # can be unpickled properly. Only do this if the file is from a trusted source.
    batch = torch.load(
        file_path,
        map_location="cpu",
        weights_only=False,  # use full pickle, not restricted weights_only loader
    )

    # Compute local index within this batch
    local_idx = instance_idx - cum_sizes[file_idx]
    return batch[local_idx]



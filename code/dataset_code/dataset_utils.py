"""
dataset.py contains code for making a dataset of instances
"""

import logging
import os
import torch
from torch import Tensor

import code.dataset_code.dataset_maker

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)


from code.dataset_code.data_manipulation import expand_matrix

import bisect

from tensordict import TensorDict
from typing import Tuple


import time






# gets values from schedule dataset used to train models
def get_dataset_features(td: TensorDict, env, order: bool, h: int, w: int, include_ops_sequence: bool = False, from_benchmark_instance: bool = False, include_ops_actions: bool = True) -> Tuple[
        torch.Tensor, # target assignments
        torch.Tensor, # proc times matrix
        torch.Tensor, # jobid matrix
        torch.Tensor, # pos in job matrix
        ]:

    # ASSIGNMENT
    assignments = None
    if from_benchmark_instance is False:
        if order:
            assignments = td["opt_assignment_order"]
            assignments = assignments.unsqueeze(0)
        else:
            assignments = td['opt_assignment']
            assignments = assignments.unsqueeze(0)
    else:
        assignments = td['ma_assignment']
        assignments = assignments.unsqueeze(0)

    assignments = expand_matrix(assignments, (h, w))
    #logging.info(assignments)

    # PROC TIMES
    proc_times = td['proc_times']
    proc_times = proc_times.unsqueeze(0)
    proc_times = expand_matrix(proc_times, (h, w))
    #logging.info(proc_times)

    # JOB OPS ADJ
    job_ops_adj = td['job_ops_adj']
    #logging.info(job_ops_adj)
    job_ops_adj = job_ops_adj.unsqueeze(0)
    job_ops_adj = expand_matrix(job_ops_adj, (h, w))
    #logging.info(job_ops_adj)

    # OPS MA ADJ
    ops_ma_adj = td['ops_ma_adj']
    ops_ma_adj = ops_ma_adj.unsqueeze(0)
    ops_ma_adj = expand_matrix(ops_ma_adj, (h, w))

    if include_ops_sequence:
        if include_ops_actions:
            return assignments, proc_times, job_ops_adj, ops_ma_adj, td["ops_sequence_order"], td["opt_actions"]
        else:
            return assignments, proc_times, job_ops_adj, ops_ma_adj, td["ops_sequence_order"], None

    return assignments, proc_times, job_ops_adj, ops_ma_adj



def print_info_about_dataset(td: TensorDict):
    print(td["opt_assignment_order"].shape)
    print(td["opt_assignment"].shape)
    print(td["opt_assignment"][0])
    print(td["opt_assignment_order"][0])



def make_dataset(n: int, dataset_folder: str, 
                 n_jobs, n_ma, max_op_per_job, min_op_per_job, max_proc_time, min_proc_time, max_eligable_ma_per_op, min_eligable_ma_per_op,
                 target_model: str, order: bool, batch_size: int = 1280, startpoint: int = 0):

    logging.info("Making dataset...")

    os.makedirs(dataset_folder, exist_ok=True)
    for i in range(startpoint, n, batch_size):
        # lag instanse
        env, td, generator_params = code.dataset_code.dataset_maker.make_instance(
            n_ma=n_ma, n_jobs=n_jobs, 
            max_op_per_job=max_op_per_job, 
            min_op_per_job=min_op_per_job, 
            max_proc_time=max_proc_time, 
            min_proc_time=min_proc_time, 
            max_eligable_ma_per_op=max_eligable_ma_per_op, 
            min_eligable_ma_per_op=min_eligable_ma_per_op, 
            batch_size=batch_size)
        # fa target fra instance
        # TODO fjern
        # print(td.shape)
        # print(td['ops_ma_adj'].shape)
        # print(type(td))
        # print("exiting")
        # exit()

        td_target, actions, ordered_assignments = code.dataset_code.dataset_maker.make_target(env, td.copy(), target_model, order)

        # lagre json med: td, og optimale td koords
        td.set('opt_assignment', td_target['ma_assignment'])
        td.set('opt_actions', torch.tensor(actions))
        if order:
            td.set('opt_assignment_order', ordered_assignments)

        # TODO hvis du vil sjekke data, for testring
        #print_info_about_dataset(td)
        """
        print("ass order")
        print(td[10]["opt_assignment"])
        print(td[10]["opt_assignment_order"])
        #print(td_target[10]["ops_ma_adj"])
        #print(td[10]["ops_ma_adj"])
        #print(td_target[10]["ops_sequence_order"])
        print(td[10]["ops_sequence_order"])

        print(td[10]["opt_actions"])
        exit()
        """
        torch.save(td.copy(), f'{dataset_folder}/{i}_{i+batch_size}.pt')

        print(f'Made instance {i} to {i+batch_size}')


    print(f'Made all {i+batch_size} instances. Done making dataset')








# gets a dataset instance
def get_dataset_instance(path: str, instance_idx: int) -> Tensor:
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



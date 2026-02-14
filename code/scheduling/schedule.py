"""
scheduling_utils.py contains all code that does scheduling
"""

from joblib import Parallel, delayed

import time
import random
import os
import logging
import bisect

import code.scheduling.utils as utils

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)

import torch
from tensordict import TensorDict, from_dict
from torch import Tensor

from torchtyping import TensorType
from typing import Callable, Dict, List, Tuple

from rl4co.envs import FJSPEnv
from rl4co.models.zoo.l2d import L2DModel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


"""
params:
  0: number of machines
  1: ops per job
  2: number of jobs
  3: minimum proc time
  4: maximum proc time
"""

def make_instance(
    n_ma, n_jobs, max_op_per_job, min_op_per_job, max_proc_time, min_proc_time, max_eligable_ma_per_op, min_eligable_ma_per_op, batch_size
) -> Tuple[FJSPEnv, TensorDict, Dict]:

    

    generator_params = {
        "num_jobs": n_jobs,
        "num_machines": n_ma,
        "min_ops_per_job": min_op_per_job,
        "max_ops_per_job": max_op_per_job,
        "min_processing_time": min_proc_time,
        "max_processing_time": max_proc_time,
        "min_eligible_ma_per_op": min_eligable_ma_per_op,
        "max_eligible_ma_per_op": max_eligable_ma_per_op,
    }

    env = FJSPEnv(
        generator_params=generator_params,
        _torchrl_mode=True,
        stepwise_reward=True
    )
    td = env.reset(batch_size=[batch_size])
    return env, td, generator_params


# kan brukes hvis td ikke inneholder order fra før av
def make_adj_with_order(
        actions: List, td_unscheduled: TensorDict, env: FJSPEnv, order: bool
    )-> TensorDict:

    n_actions = len(actions)

    td_to_actions = td_unscheduled.copy()
    td_to_actions = td_to_actions.unsqueeze(0)
    prev_adj = td_to_actions["ma_assignment"].clone()

    # this stores the sequence / order matrix
    assignment_adj = torch.zeros_like(prev_adj, dtype=torch.float)

    per_row_counts = torch.zeros(prev_adj.size(0), dtype=torch.int)
    for i, action in enumerate(actions):
        td_to_actions["action"] = torch.tensor([action])
        td_to_actions = env.step(td_to_actions)["next"]

        new_adj = td_to_actions["ma_assignment"]

        diff = (new_adj == 1) & (prev_adj == 0)

        if diff.any():
            normalized_order = i / float(n_actions)
            assignment_adj[diff] = normalized_order
        prev_adj = new_adj.clone()


    td_to_actions["ma_assignment"] = assignment_adj
    return td_to_actions




# actions: [batch_size, seq_len]
def schedule_actions_batch(env: FJSPEnv, actions: List, td: TensorDict, order: bool) -> TensorDict:

    n_actions = len(actions)
    prev_adj = td["ma_assignment"].clone()
    ordered_assignments = torch.zeros_like(prev_adj, dtype=torch.float)

    actions_assigned = 0
    for t in range(actions.size(1)):
        td["action"] = actions[:, t] 
        td = env.step(td)["next"]
        
        if order:
            new_adj = td["ma_assignment"]
            diff = (new_adj == 1) & (prev_adj == 0)
            if diff.any():
                actions_assigned += 1
                ordered_assignments[diff] = actions_assigned
                #normalized_order = t / float(n_actions)
                #ordered_assignments[diff] = normalized_order
            prev_adj = new_adj.clone()

    if order:
        return td, ordered_assignments
    else:
        return td, None



def make_target(env: FJSPEnv, td: TensorDict, checkpoint_path: str, order: bool = False) -> Tuple[TensorDict, List]:

    model = L2DModel.load_from_checkpoint(checkpoint_path)
    model = model.to("cpu")

    with torch.inference_mode():
        out = model(td,
                    decode_type="multistart_sampling",
                    num_starts=5,
                    select_best=True,
                    return_actions=True)
    actions = out["actions"]
    td_scheduled, ordered_assignments = schedule_actions_batch(env, actions, td.copy(), order)
    return td_scheduled, actions, ordered_assignments








 

# bruk hvis ordered, for å se klart sekvens
# første operasjon er laveste tallet i return matrisen, det er annerledes enn hvordan det ellers er, der største verdi rett fra modell er første operasjon
def make_step(env, td, action):
    td['action'] = torch.tensor([action])
    td = env.step(td)['next']


    return td

import time
# import code.dataset_code.utils as dataset_utils


# TODO FJERN
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


from functorch import vmap

# busy ... will only schedule action if machine avalible, otherwise wait
# notbusy .. will skip action if machine not avalible at the time

def do_actions(actions, n_machines, td, env):

    for action in actions:
        # machine index that this action refers to
        ma_to_use = (action - 1) % n_machines


        while td["busy_until"][0, ma_to_use].item() > td["time"].item():
            invalid_action = torch.tensor([0])  
            td["action"] = invalid_action

            td = env.step(td)["next"]

        if action != 0:
            td["action"] = torch.tensor([action])
            td = env.step(td)["next"]

    return td



def do_actions_fix_gap(actions, n_machines, td, env):
    
    for action in actions:
        # machine index that this action refers to
        ma_to_use = (action - 1) % n_machines


        while td["busy_until"][0, ma_to_use].item() > td["time"].item():
            invalid_action = torch.tensor([0])  
            td["action"] = invalid_action

            td = env.step(td)["next"]

        if action != 0:
            td["action"] = torch.tensor([action])
            td = env.step(td)["next"]

    return td

import torch



def inferenced_schedule( assignments, order: bool, env, td, path_save_image: str, n_jobs: int, n_machines: int, error_list, ops_sequence_order):
    print(f"assignments shape: {assignments.shape}")
    n_jobs = infer_n_jobs(ops_sequence_order) 




    print(f"making actions for assignments with {os.cpu_count()} processors")
    B = assignments.size(0)
    all_actions = Parallel(n_jobs=os.cpu_count())(
        delayed(utils.map_assignments_to_actions_text)( assignments[b], True, n_jobs )
        for b in range(B)
    )
    all_actions = torch.stack(all_actions)
    all_actions = all_actions.to(torch.int64) 

    print(f"all actions shape: {all_actions.shape}")
    td.del_("opt_assignment")
    td.del_("opt_assignment_order")
    td.del_("opt_actions")
    td = td.unsqueeze(0)

 
    feasible_indecies = [i for i in range(len(error_list)) if error_list[i] == 0] 
    tds = Parallel(n_jobs=os.cpu_count())(
        delayed(do_actions)( all_actions[f_i], n_machines, td.copy(), env )
        for f_i in feasible_indecies
    )

    # filling gaps
    ## mapping operations to machines
    machine_assignments_maps = Parallel(n_jobs=os.cpu_count())(
        delayed(utils.map_operation_to_machines)(td["ma_assignment"])
        for td in tds
    )
    #machine_assignments_maps = [list(m) for m in machine_assignments_maps]

    ## filling gaps
    tds = Parallel(n_jobs=os.cpu_count())(
        delayed(utils.compress_schedule)(td["start_times"], td["finish_times"], ma_op_map, n_jobs, td, filler_machine=99)
        for td, ma_op_map in zip(tds, machine_assignments_maps)
    )
    
    makespans = [
        td["finish_times"][td["finish_times"] != 9999.0].max().item()
        for td in tds
    ]
    td_best = tds[ makespans.index( min(makespans) ) ]

    """
    start_times = td_best["start_times"]
    finish_times = td_best["finish_times"]
    machines = utils.map_operation_to_machines(td_best["ma_assignment"])
    job_lengths = n_jobs
    td_best["start_times"], td_best["finish_times"] = utils.compress_schedule(start_times, finish_times, machines, job_lengths, filler_machine=99)


    makespans = [
        td["busy_until"].max(dim=1).values
        for td in tds
    ]
    """

    print("")
    print(f"makespans: {makespans}")
    print("")
    print(f"lowest makespan: {min(makespans)}")
    print(f"highestmakespan: {max(makespans)}")
    print(f"Average makespan: {sum(makespans) / len(makespans)}")



    env.render(td_best, 0)
    if path_save_image:
        plt.savefig(path_save_image, dpi=150, bbox_inches='tight')
        print(f"Saved scheduled image at path {path_save_image}")



    return td_best





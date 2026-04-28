"""
scheduling_utils.py contains all code that does scheduling
"""

from joblib import Parallel, delayed

import time
import random
import os
import logging
import bisect

from code.inference.data_inform import infer_n_jobs
import code.scheduling.fix_scheduling_gaps
import code.scheduling.utils as utils
from code.scheduling import fix_scheduling_gaps


logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)

import torch
from tensordict import TensorDict, from_dict
from torch import Tensor

from torchtyping import TensorType
from typing import Callable, List

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


import code.dataset_code.benchmark_utils as data_utils


# TODO fortsatt under testing. Forsøk på å lage dataset der man stiller på machine utilization
def schdule_by_utilization():
    """
    print(1)
    checkpoint_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/rl4co_model_0.0001_10j_6ma_6op_mk01.ckpt'
    model = L2DModel.load_from_checkpoint(checkpoint_path)
    model = model.to("cpu")

    print(2)
    instance_idx = 10
    dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_mk01_10j_6ma_6op_mk01'
    td = data_utils.get_td_from_path(dataset_folder, instance_idx)

    td.del_("opt_assignment")
    td.del_("opt_assignment_order")
    td.del_("opt_actions")
    td = td.unsqueeze(0)

    print(3)
    print(td['ops_ma_adj'].shape)

    # reset env with instance
    env = FJSPEnv()
    td = env.reset(td)

    with torch.inference_mode():
        out = model(td,
                    decode_type="multistart_sampling",
                    num_starts=5,
                    select_best=True,
                    return_actions=True)
    actions = out["actions"]

    td_scheduled, ordered_assignments = schedule_actions_batch(env, actions, td.copy(), True)


    path_save_image = f'/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/testing_util.png'
    env.render(td_scheduled, 0)
    if path_save_image:
        plt.savefig(path_save_image, dpi=150, bbox_inches='tight')
        print(f"Saved scheduled image at path {path_save_image}")

    """
       # 1) Load model
    checkpoint_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/rl4co_model_0.0001_10j_6ma_6op_mk01.ckpt'
    model = L2DModel.load_from_checkpoint(checkpoint_path).to("cpu")
    model.eval()

    benchmark_instance_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk01.txt'
    parameters = data_utils.get_rl4co_parameters_from_brandimarte_instance(benchmark_instance_path)
    print(parameters)

    generator_params = {
        "num_jobs": parameters['n_jobs'],
        "num_machines": parameters['n_machines'],
        "min_ops_per_job": parameters['fewest_operations'],
        "max_ops_per_job": parameters['most_operations'],
        "min_processing_time": parameters['min_processing_time'],
        "max_processing_time": parameters['max_processing_time'],
        "min_eligible_ma_per_op": parameters['min_machine_options'],
        "max_eligible_ma_per_op": parameters['max_machine_options'],
    }

    env = FJSPEnv(generator_params=generator_params)
    td = env.reset(batch_size=[1])



    with torch.no_grad():

        done = False
        while not done:

            # Get action logits from policy
            out = model.policy(td)
            logits = out["logits"]
            mask = out["mask"]

            # Greedy action (respect mask)
            logits[~mask] = -torch.inf
            action = logits.argmax(dim=-1)

            # Add action to tensordict
            td["action"] = action

            # Step environment
            td = env.step(td)["next"]

            # Check termination
            done = td["done"].item()

            print("Action chosen:", action.item())












    """
    path_save_image = f'/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/testing_util.png'
    env.render(td_current, 0)
    if path_save_image:
        plt.savefig(path_save_image, dpi=150, bbox_inches='tight')
        print(f"Saved scheduled image at path {path_save_image}")

    """










# Takes schedule made from the model, than schedules it
def schedule_from_inference( assignments, order: bool, env, td, path_save_image: str, n_jobs: int, n_machines: int, error_list, ops_sequence_order, report_file_path, fill_gaps: bool = False):
    print(f"assignments shape: {assignments.shape}")
    n_jobs = infer_n_jobs(ops_sequence_order) 


    jobs_to_make = int(os.cpu_count() / 6)

    print(f"making actions for assignments with {os.cpu_count()} processors")
    B = assignments.size(0)
    all_actions = Parallel(n_jobs=jobs_to_make)(
        delayed(utils.map_assignments_to_actions_text)( assignments[b], True, n_jobs )
        for b in range(B)
    )
    all_actions = torch.stack(all_actions)
    all_actions = all_actions.to(torch.int64) 

    print(f"all actions shape: {all_actions.shape}")
    if any("opt_" in key for key in td):
        td.del_("opt_assignment")
        td.del_("opt_assignment_order")
        td.del_("opt_actions")
    else:
        for k in td.keys():
            td[k] = td[k].unsqueeze(0)  # shape becomes [1, 6, 60] for your tensor

        # Wrap in a TensorDict
        td = TensorDict(td, batch_size=[1])

        print(td['ops_ma_adj'].shape)  # torch.Size([1, 6, 60])

    # Scheduling
    feasible_indecies = [i for i in range(len(error_list)) if error_list[i] == 0] 
    tds = Parallel(n_jobs=jobs_to_make)(
        delayed(do_actions)( all_actions[f_i], n_machines, td.copy(), env )
        for f_i in feasible_indecies
    )

    # filling gaps
    ## mapping operations to machines
    machine_assignments_maps = Parallel(n_jobs=jobs_to_make)(
        delayed(utils.map_operation_to_machines)(td["ma_assignment"])
        for td in tds
    )
    #machine_assignments_maps = [list(m) for m in machine_assignments_maps]

    ## filling gaps
    tds = Parallel(n_jobs=jobs_to_make)(
        delayed(code.scheduling.fix_scheduling_gaps.compress_schedule)(td["start_times"], td["finish_times"], ma_op_map, n_jobs, td, filler_machine=99)
        for td, ma_op_map in zip(tds, machine_assignments_maps)
    )
    '''
    makespans = [
    td["finish_times"][td["finish_times"] != 9999.0].max().item()
    for td in tds
    if td["finish_times"][td["finish_times"] != 9999.0].max().item() >= 30.0
    ]
    '''
    makespans = [
        td["finish_times"][td["finish_times"] != 9999.0].max().item()
        for td in tds
    ]
    print(f'makespans made: {len(makespans)}')
    td_best = tds[ makespans.index( min(makespans) ) ]

    """
    start_times = td_best["start_times"]
    finish_times = td_best["finish_times"]
    machines = utils.map_operation_to_machines(td_best["ma_assignment"])
    job_lengths = n_jobsdd
    td_best["start_times"], td_best["finish_times"] = utils.compress_schedule(start_times, finish_times, machines, job_lengths, filler_machine=99)


    makespans = [
        td["busy_until"].max(dim=1).values
        for td in tds
    ]
    """

    print("")

    min_makespan = min(makespans)
    max_makespan = max(makespans)
    avg_makespan = sum(makespans) / len(makespans)
    print(f"makespans: {makespans}")
    print("")
    print(f"lowest makespan: {min_makespan}")
    print(f"highest makespan: {max_makespan}")
    print(f"Average makespan: {avg_makespan}")



    env.render(td_best, 0)
    if path_save_image:
        plt.savefig(path_save_image, dpi=150, bbox_inches='tight')
        print(f"Saved scheduled image at path {path_save_image}")



    return td_best, min_makespan, max_makespan, avg_makespan





def schedule_randomly(ops_ma_adj, valid_h, valid_w, job_lengths, N=55):

    ops_ma_adj = ops_ma_adj[:, :, :valid_h, :valid_w]
    out = torch.zeros_like(ops_ma_adj, dtype=torch.float32)

    # Global pool of values
    available_values = list(range(1, N+1))

    col = 0
    for length in job_lengths:

        if length <= 1:
            col += length
            continue

        if len(available_values) < length:
            break  # not enough values left

        group_cols = list(range(col, col + length))

        # 🔥 RANDOM UNIQUE VALUES
        group_values = random.sample(available_values, length)

        # ensure increasing inside group
        group_values.sort()

        # remove from global pool
        for v in group_values:
            available_values.remove(v)

        # assign them
        for c, v in zip(group_cols, group_values):

            valid_rows = (ops_ma_adj[0,0,:,c] == 1).nonzero(as_tuple=True)[0].tolist()
            if not valid_rows:
                continue

            row = valid_rows[0]  # or random.choice(valid_rows)
            out[0,0,row,c] = v

        col += length

    return out / 100.0
"""
scheduling_utils.py contains all code that does scheduling
"""
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*failed to set start method to spawn.*"
)

warnings.filterwarnings(
    "ignore",
    message=".*A worker stopped while some jobs were given to the executor.*"
)

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
# kanskje...
# man tracker action som blir gjort, og tida for den actionen
# så tracker man maskinen den blir lagt på
# så bare har man en counter selv, som returnerer maskinen med lengst tid..
# men det kan hende det er noe feil med enviromentet som fucker opp dette

# annen mulighet ... 
# på en eller annen måte får de valide mulige actionene... i lista av actions, velger jeg der den av actionene som er først i lista.
'''
def do_actions(actions, n_machines, td, env, index):
    print('doing actions..')
    invalid_action_counter = 0
    actions_taken = []
    save_dir = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/code/inference/sched_frames'
    frame_idx = 0

    for action in actions:
        # machine index that this action refers to
        ma_to_use = (action - 1) % n_machines

        # If action cant be scheduled because machine is busy, then skip forward in time
        while td["busy_until"][0, ma_to_use].item() > td["time"].item():
            invalid_action = torch.tensor([0])  
            td["action"] = invalid_action

            td = env.step(td)["next"]
            invalid_action_counter += 1

        # Doing valid actions
        if action != 0:
            td["action"] = torch.tensor([action])
            td = env.step(td)["next"]
            actions_taken.append(action)

            if index == 0:

                plt.clf()
                env.render(td, 0)
                save_path = os.path.join(
                    save_dir,
                    f"frame_{frame_idx:04d}_act{action}_time{td['time']}.png"
                )
                plt.savefig(save_path, bbox_inches="tight")
                frame_idx += 1


    return td, invalid_action_counter, actions_taken
'''
import torch
from collections import deque




def batched_schedule_rollout(
    env,
    td,
    all_actions,
    render_idx=None,
    save_dir="frames",
):
    """
    Batched RL4CO rollout with precomputed action sequences.

    Parameters
    ----------
    env:
        RL4CO environment

    td:
        Batched TensorDict
        shape = [batch]

    all_actions:
        Tensor of shape:
            [batch, seq_len]

    render_idx:
        Optional batch index to render/save

    save_dir:
        Where PNG frames are saved

    Returns
    -------
    td:
        final TensorDict

    executed_actions:
        tensor [batch, max_steps]
    """

    device = td.device

    batch_size = all_actions.shape[0]
    seq_len = all_actions.shape[1]

    all_actions = all_actions.to(device)


    # ---------------------------------------------------
    # Per-batch pointer:
    # tells which action each rollout wants next
    # ---------------------------------------------------

    current_ptr = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=device,
    )

    # done tracking
    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=device,
    )

    # executed actions history
    executed_history = []

    skip_counter = torch.zeros(
    batch_size,
    dtype=torch.long,
    device=device,
    )

    # rendering
    if render_idx is not None:
        os.makedirs(save_dir, exist_ok=True)

    frame_idx = 0

    # ---------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------

    while not td["done"].all():

        mask = td["action_mask"]

        batch_ids = torch.arange(
            batch_size,
            device=device,
        )

        # ---------------------------------------------
        # Clamp pointers
        # ---------------------------------------------

        safe_ptr = torch.clamp(
            current_ptr,
            max=seq_len - 1
        )

        proposed_actions = all_actions[
            batch_ids,
            safe_ptr
        ]

        # ---------------------------------------------
        # Check feasibility
        # ---------------------------------------------

        feasible = mask[
            batch_ids,
            proposed_actions
        ]

        # ---------------------------------------------
        # If infeasible:
        # advance pointer until feasible
        # ---------------------------------------------

        max_retries = seq_len

        retries = 0

        while not feasible.all():

            infeasible_idx = (~feasible).nonzero(
                as_tuple=True
            )[0]

            skip_counter[infeasible_idx] += 1

            current_ptr[infeasible_idx] += 1

            safe_ptr = torch.clamp(
                current_ptr,
                max=seq_len - 1
            )

            proposed_actions = all_actions[
                batch_ids,
                safe_ptr
            ]

            feasible = mask[
                batch_ids,
                proposed_actions
            ]

            retries += 1

            if retries > max_retries:
                break

        # ---------------------------------------------
        # fallback for impossible states
        # ---------------------------------------------

        still_bad = ~feasible

        if still_bad.any():

            valid_actions = mask.float().argmax(dim=1)

            proposed_actions[still_bad] = valid_actions[
                still_bad
            ]

        # ---------------------------------------------
        # STEP ENTIRE BATCH
        # ---------------------------------------------

        td["action"] = proposed_actions

        td = env.step(td)["next"]

        executed_history.append(
            proposed_actions.clone()
        )

        # ---------------------------------------------
        # Advance pointers
        # ---------------------------------------------

        current_ptr += 1

        # ---------------------------------------------
        # OPTIONAL RENDERING
        # ---------------------------------------------

        if render_idx is not None:

            plt.clf()

            env.render(td, render_idx)

            action_rendered = proposed_actions[
                render_idx
            ].item()

            current_time = td["time"][
                render_idx
            ].item()

            save_path = os.path.join(
                save_dir,
                f"frame_{frame_idx:04d}"
                f"_act{action_rendered}"
                f"_time{current_time:.2f}.png"
            )

            plt.savefig(
                save_path,
                bbox_inches="tight"
            )

            plt.close()

            frame_idx += 1

    # ---------------------------------------------------
    # STACK HISTORY
    # ---------------------------------------------------

    executed_history = torch.stack(
        executed_history,
        dim=1
    )

    return td, executed_history, skip_counter



def do_actions(actions, n_machines, td, env, index):
    save_dir = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/code/inference/sched_frames'
    """
    Execute a precomputed list of actions safely in RL4CO scheduling envs.

    Invalid/busy actions are postponed until they become feasible.
    """

    # queue of actions still to try
    pending = deque(actions)

    executed_actions = []


    frame_idx = 0
    while not td["done"].all():

        mask = td["action_mask"].squeeze(0)

        scheduled_this_step = False

        # try every currently pending action once
        for _ in range(len(pending)):

            action = pending.popleft()

            # feasible now?
            if mask[action]:

                td["action"] = torch.tensor([action], device=mask.device)

                td = env.step(td)["next"]

                executed_actions.append(action)

                if index == 0:
                    plt.clf()
                    env.render(td, 0)
                    save_path = os.path.join(
                        save_dir,
                        f"frame_{frame_idx:04d}_act{action}_time{td['time']}.png"
                    )
                    plt.savefig(save_path, bbox_inches="tight")
                    frame_idx += 1

                scheduled_this_step = True

                break

            else:
                # machine busy / job not ready
                # postpone action for later
                pending.append(action)

        # IMPORTANT:
        # if nothing could be scheduled,
        # env must advance time somehow
        #
        # RL4CO scheduling envs usually do this internally
        # when only wait/no-op remains feasible.
        #
        # So we execute a valid fallback action.

        if not scheduled_this_step:

            valid_actions = torch.where(mask)[0]

            if len(valid_actions) == 0:
                raise RuntimeError("No valid actions available")

            # choose first feasible fallback action
            fallback = valid_actions[0]

            td["action"] = fallback.unsqueeze(0)

            td = env.step(td)["next"]

            executed_actions.append(fallback.item())
            if index == 0:
                plt.clf()
                env.render(td, 0)
                save_path = os.path.join(
                    save_dir,
                    f"frame_{frame_idx:04d}_act{fallback.unsqueeze(0)}_time{td['time']}.png"
                )
                plt.savefig(save_path, bbox_inches="tight")
                frame_idx += 1



    return td, 0, executed_actions

'''
def do_actions(actions, td, env):
    print('doing actions')

    invalid_action_counter = 0
    actions_taken = []

    remaining_actions = [int(a) for a in actions if int(a) != 0]

    while remaining_actions:

        mask = td["action_mask"][0]

        found_action = False

        # Try to find ANY currently feasible action
        for idx, action in enumerate(remaining_actions):

            if mask[action]:

                td["action"] = torch.tensor(
                    [action],
                    device=td.device
                )

                td = env.step(td)["next"]

                actions_taken.append(action)

                remaining_actions.pop(idx)

                found_action = True

                break

        # No action currently feasible -> WAIT
        # No action currently feasible
        if not found_action:

            current_time = td["time"].item()

            # Are any machines still processing?
            future_events_exist = torch.any(
                td["busy_until"][0] > current_time
            )

            # If nothing is processing anymore,
            # then remaining actions are impossible forever
            if not future_events_exist:

                print("Deadlock reached.")
                print("Remaining actions:", remaining_actions[:20])

                break

            # Otherwise WAIT for next machine completion
            td["action"] = torch.tensor(
                [0],
                device=td.device
            )

            td = env.step(td)["next"]

            invalid_action_counter += 1

    return td, invalid_action_counter, actions_taken
'''
# TODO problem var at hang for lenge...
'''
def do_actions(actions, td, env):

    invalid_action_counter = 0
    actions_taken = []

    for action in actions:

        action = int(action)

        # Skip padding
        if action == 0:
            continue

        while True:

            mask = td["action_mask"][0]

            # Action feasible -> execute
            if mask[action]:

                td["action"] = torch.tensor(
                    [action],
                    device=td.device
                )

                td = env.step(td)["next"]

                actions_taken.append(action)

                break

            # No machines currently processing
            # => action will NEVER become feasible
            busy = td["busy_until"][0]
            current_time = td["time"].item()

            future_events_exist = torch.any(busy > current_time)

            if not future_events_exist:

                print(f"Skipping permanently infeasible action {action}")

                invalid_action_counter += 1

                break

            # Otherwise WAIT until next event
            td["action"] = torch.tensor(
                [0],
                device=td.device
            )

            td = env.step(td)["next"]

            invalid_action_counter += 1

    return td, invalid_action_counter, actions_taken
'''



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
    
    n_jobs = infer_n_jobs(ops_sequence_order) # for mk03 gir denne 10, når den egentlig har 15 jobber
    # mente jeg her å få max prosesser, eller jobber? .. mk03 har 10 max prosesser
    # n_jobs = 15


    jobs_to_make = int(os.cpu_count() / 6)

    print('printing..')
    print(n_jobs)
    print(sum(i for i in n_jobs if i > 1))

    print(f"making actions for assignments with {os.cpu_count()} processors")
    B = assignments.size(0)
    all_actions = Parallel(n_jobs=jobs_to_make)(
        delayed(utils.map_assignments_to_actions_text)( assignments[b], True, n_jobs )
        for b in range(B)
    )
    all_actions = torch.stack(all_actions)
    all_actions = all_actions.to(torch.int64) 
    print(all_actions[0])
    print(len(all_actions[0]))


    if any("opt_" in key for key in td):
        td.del_("opt_assignment")
        td.del_("opt_assignment_order")
        td.del_("opt_actions")
    else:
        for k in td.keys():
            td[k] = td[k].unsqueeze(0)  # shape becomes [1, 6, 60] for your tensor

        # Wrap in a TensorDict
        td = TensorDict(td, batch_size=[1])

    print(td.shape)
    exit()
    # Scheduling
    # raise Exception('Kan ikke kjøre enda, fordi out of memory (node var opptatt).. i metode: do_actions i schedule.py ... se om kan returnere invalid actions, og legge i cond rapport .. Exception raised i schedule.py schedule_from_inference')


    '''
    feasible_indecies = [i for i in range(len(error_list)) if error_list[i] == 0] 
    results = Parallel(n_jobs=jobs_to_make)(
        delayed(do_actions)( all_actions[f_i], n_machines, td.copy(), env , None) # erstatt None med en index for å se hvordan den blir skedulert over tid.
        for f_i in feasible_indecies
    )

    '''

    batch_size = all_actions.shape[0]
    td_batched = td.expand(batch_size).clone()

    print(f'all actions shape: {all_actions.shape}')
    td_final, executed, invalid_action_counters = batched_schedule_rollout(
    env,
    td_batched,
    all_actions,
    render_idx=None,
    save_dir="frames",
    )

    print('done scheduling')
    print(invalid_action_counters.shape)


    tds = [
        td_final[i]
        for i in range(batch_size)
    ]
    invalid_action_counters = (
        invalid_action_counters
        .cpu()
        .tolist()
    )
    actions_taken = [
        executed[i].cpu().tolist()
        for i in range(batch_size)
    ]


    #tds, invalid_action_counters, actions_taken = zip(*results)
    #tds = list(tds)
    #invalid_action_counters = list(invalid_action_counters)
    #actions_taken = list(actions_taken)


    # filling gaps
    ## mapping operations to machines
    '''
    machine_assignments_maps = Parallel(n_jobs=jobs_to_make)(
        delayed(utils.map_operation_to_machines)(td["ma_assignment"])
        for td in tds
    )
    '''
    #machine_assignments_maps = [list(m) for m in machine_assignments_maps]

    ## filling gaps
    '''
    tds = Parallel(n_jobs=jobs_to_make)(
        delayed(code.scheduling.fix_scheduling_gaps.compress_schedule)(td["start_times"], td["finish_times"], ma_op_map, n_jobs, td, filler_machine=99)
        for td, ma_op_map in zip(tds, machine_assignments_maps)
    )
    '''
    makespans = [
    td["finish_times"][td["finish_times"] != 9999.0].max().item()
    for td in tds
    if td["finish_times"][td["finish_times"] != 9999.0].max().item() # >= 30.0
    ]

    best_index = makespans.index( min(makespans) )
    worst_index = makespans.index( max(makespans) )
    td_best = tds[ best_index ]
    best_actions = actions_taken[makespans.index( min(makespans ))]

    # p = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/code/inference/best_actions.pt'
    # torch.save(best_actions, p)


    # TODO ignorer
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


    min_makespan = min(makespans)
    max_makespan = max(makespans)
    avg_makespan = sum(makespans) / len(makespans)
    print(f"makespans: {makespans}")
    print("")
    print(f"lowest makespan: {min_makespan}")
    print(f"highest makespan: {max_makespan}")
    print(f"Average makespan: {avg_makespan}")



    # env.render(td_best) # sto tidligere env.render(td_best, 0)
    env.render(td_best.unsqueeze(0), 0)
    if path_save_image:
        plt.savefig(path_save_image, dpi=150, bbox_inches='tight')
        print(f"Saved scheduled image at path {path_save_image}")


    denom = len(all_actions[0])
    invalid_actions_counter_rate = [
        x / denom if denom > 0 else 0
        for x in invalid_action_counters
    ]

    invalid_act_of_max = invalid_action_counters[ worst_index ]
    invalid_act_of_min = invalid_action_counters[ best_index ]
    invalid_act_avg = sum(invalid_action_counters) / len(invalid_action_counters)

    invalid_act_rate_of_max = invalid_actions_counter_rate[ worst_index ]
    invalid_act_rate_of_min = invalid_actions_counter_rate[ best_index ]
    invalid_act_rate_avg = sum(invalid_actions_counter_rate) / len(invalid_actions_counter_rate)



    # print(f"invalid action counters: {invalid_action_counters}")
    # print(f"invalid action counter rates: {invalid_actions_counter_rate}")

    return td_best, min_makespan, max_makespan, avg_makespan, invalid_act_of_min, invalid_act_of_max, invalid_act_avg, invalid_act_rate_of_min, invalid_act_rate_of_max, invalid_act_rate_avg





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
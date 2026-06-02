"""
scheduling_utils.py contains all code that does scheduling
"""
import warnings

from collections import defaultdict

warnings.filterwarnings(
    "ignore",
    message=".*failed to set start method to spawn.*"
)

warnings.filterwarnings(
    "ignore",
    message=".*A worker stopped while some jobs were given to the executor.*"
)

from joblib import Parallel, delayed

import random
import os
import logging

from code.inference.data_inform import infer_n_jobs
import code.scheduling.utils as utils


import torch
from collections import deque


logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)

from tensordict import TensorDict

from typing import List

from rl4co.envs import FJSPEnv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import deque




def decode_machine(action, h):
    """
    Example with h=6:

    machine 2:
        2, 8, 14, 20
    """

    return ((action - 1) % h) + 1


def decode_job(action, h):

    return (action - 1) // h


def get_operation_column_from_job_counter(
    action,
    job_ops_adj,
    job_use_count,
    h
):

    job = decode_job(
        action,
        h
    )

    # Columns for this job
    job_cols = torch.where(
        job_ops_adj[job] == 1
    )[0]

    # Which operation are we on?
    op_idx = job_use_count[job]

    if op_idx >= len(job_cols):
        return None

    return int(job_cols[op_idx])


def find_new_machine_for_operation(
    action,
    ops_ma_adj,
    col_for_action,
    h
):
    """
    Change machine while preserving job.

    HARD RULE:
    machine 2 is forbidden.
    """

    current_machine = decode_machine(
        action,
        h
    )

    # Valid machines (0-based)
    valid_machines = torch.where(
        ops_ma_adj[:, col_for_action] == 1
    )[0]

    # Convert to 1-based
    valid_machines = valid_machines + 1

    # -----------------------------------
    # REMOVE:
    # - current machine
    # - machine 2
    # -----------------------------------

    valid_machines = valid_machines[
        (valid_machines != current_machine)
        & (valid_machines != 2)
    ]

    # -----------------------------------
    # NO VALID MACHINE EXISTS
    # -----------------------------------

    if len(valid_machines) == 0:
        # FORCE REMOVE MACHINE 2
        return action

    # Random machine
    new_machine = valid_machines[
        torch.randint(
            len(valid_machines),
            (1,)
        )
    ].item()

    # Decode job
    job = decode_job(
        action,
        h
    )

    # Reconstruct action
    new_action = (
        job * h
        + new_machine
    )

    return int(new_action)



def modify_actions_batch_mautil(
    actions,
    td
):
    max_amount = 1
    actions = actions.clone()

    b, h, w = td["ma_assignment"].shape

    # ---------------------------------------------
    # Job counters
    # ---------------------------------------------

    max_jobs = td['job_ops_adj'].shape[1]

    job_use_count = torch.zeros(
        (b, max_jobs),
        dtype=torch.long,
        device=actions.device
    )

    # ---------------------------------------------
    # MODIFY ACTIONS
    # ---------------------------------------------

    for t in range(actions.size(1)):

        for batch_idx in range(b):

            action = actions[
                batch_idx,
                t
            ].item()

            # Skip padding
            if action == 0:
                continue

            # Decode machine
            machine = decode_machine(
                action,
                h
            )

            # Decode job
            job = decode_job(
                action,
                h
            )

            # Find operation column
            col_for_action = (
                get_operation_column_from_job_counter(
                    action,
                    td['job_ops_adj'][batch_idx],
                    job_use_count[batch_idx],
                    h
                )
            )

            if col_for_action is None:
                continue

            # Increment job op counter
            job_use_count[
                batch_idx,
                job
            ] += 1

            # -----------------------------------------
            # CURRENT machine-2 count
            # -----------------------------------------

            current_actions = actions[
                batch_idx
            ]

            current_machines = (
                ((current_actions - 1) % h) + 1
            )

            current_machine_2_count = (
                (current_machines == 2)
                & (current_actions != 0)
            ).sum()

            # -----------------------------------------
            # Too many machine-2 actions
            # -----------------------------------------

            if (
                machine == 2
                and current_machine_2_count > max_amount
            ):

                new_action = (
                    find_new_machine_for_operation(
                        action,
                        td['ops_ma_adj'][batch_idx],
                        col_for_action,
                        h
                    )
                )


                actions[
                    batch_idx,
                    t
                ] = new_action

    return actions


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








def make_queue_mautil(
    actions,
    n_machines
):

    """
    Returns:
    {
        machine: [
            (
                action,
                job,
                precedence_in_job
            ),
            ...
        ]
    }
    """
    # machine queues
    queue_machines = {}

    for machine in range(n_machines):

        queue_machines[machine] = []

    # count occurrences per job
    job_counts = {}

    # process actions in order
    for action in actions.tolist():

        if action == 0:
            continue

        # decode machine
        machine = (
            (action - 1)
            % n_machines
        )

        # decode job
        job = (
            (action - 1)
            // n_machines
        )

        # precedence in job
        if job not in job_counts:
            job_counts[job] = 0
        else:
            job_counts[job] += 1

        precedence = job_counts[job]

        # append
        queue_machines[
            machine
        ].append(
            (
                action,
                job,
                precedence
            )
        )

    return queue_machines





def check_processing_times(
    start_times,
    finish_times,
    min_processing_time,
    max_processing_time
):
    """
    Checks whether all scheduled operations
    have durations within the allowed range.
    """

    durations = finish_times - start_times

    # Remove padded / unscheduled operations
    valid_mask = finish_times < 9999

    valid_durations = durations[valid_mask]

    too_small = valid_durations < min_processing_time
    too_large = valid_durations > max_processing_time

    invalid_mask = too_small | too_large
    if invalid_mask.any():

        print('found invalid processing duration ranges:')
        print(valid_durations)
        print(
            f"\nAny invalid durations? "
            f"{invalid_mask.any()}"
        )
        print("\nInvalid durations:")
        print(
            valid_durations[
                invalid_mask
            ]
        )


def check_duplicate_job_precedence(
    queue_machines
):
    """
    Checks whether the same:
        (job, precedence)
    appears multiple times
    across machine queues.

    queue format:
    {
        machine: [
            (action, job, precedence),
            ...
        ]
    }
    """

    seen = defaultdict(list)

    # Collect occurrences
    for machine, queue in queue_machines.items():

        for action, job, precedence in queue:

            key = (job, precedence)

            seen[key].append(
                (machine, action)
            )

    # Find duplicates
    duplicates = {}

    for key, occurrences in seen.items():

        if len(occurrences) > 1:

            duplicates[key] = occurrences

    # Print result
    if len(duplicates) == 0:

        print(
            "No duplicate "
            "(job, precedence) pairs found."
        )

    else:
        print(
            "Duplicate "
            "(job, precedence) pairs found:"
        )

        for (
            job,
            precedence
        ), occurrences in duplicates.items():

            print(
                f"job={job}, "
                f"precedence={precedence}"
            )

            for machine, action in occurrences:

                print(
                    f"    machine={machine}, "
                    f"action={action}"
                )

    return duplicates




def make_queue_mautil_alt(
    actions,
    n_machines
):

    """
    Returns:

    {
        (
            action,
            job,
            precedence_in_job,
            machine
        ),
        ...
    }
    """

    # machine queues
    queue_machines = []

    # count occurrences per job
    job_counts = {}

    # process actions in order
    for action in actions.tolist():

        if action == 0:
            continue

        # decode machine
        machine = (
            (action - 1)
            % n_machines
        )

        # decode job
        job = (
            (action - 1)
            // n_machines
        )

        # precedence in job
        if job not in job_counts:
            job_counts[job] = 0
        else:
            job_counts[job] += 1

        precedence = job_counts[job]

        # append
        queue_machines.append(
            (
                action,
                job,
                precedence,
                machine
            )
        )

    return queue_machines


def schedule_single_instance_mautil_alt(
    td,
    env,
    actions,
    order=True, 
    busy_times=False 
):
    

    amt_busy = 0
    device = td.device

    ops_ma_adj_before_schedule = td['ops_ma_adj'].clone()
    proc_times_before_schedule = td['proc_times'].clone()
    job_lengths = (
        td["job_ops_adj"][0]
        .sum(dim=1)
        .tolist()
    )

    _, n_machines, _ = (
        td["ma_assignment"].shape
    )


    # Build queues
    queue_machines = (
        make_queue_mautil_alt(
            actions,
            n_machines
        )
    )

    num_jobs = td["job_ops_adj"].shape[1]
    job_counter = [0 for _ in range(num_jobs)]

    # track order of assignments
    prev_adj = td["ma_assignment"].clone()

    ordered_assignments = torch.zeros_like(
        prev_adj,
        dtype=torch.float
    )

    actions_assigned = 0
    job_actions_done = [[] for i in range(num_jobs)]
    is_ma_unavailable = [False for i in range(n_machines)]

    # TODO Jeg lurer på om machine kanskje starter fra 1??? tror starter på 0
    n_ops_on_ma = [0 for i in range(n_machines)]
    for _, _, _, machine in queue_machines:
        n_ops_on_ma[machine] += 1


    # schedule
    while not td['done'].all():

        i = 0
        # while there are remaining operations in queue, and not all machines are unavalible
        while i < len(queue_machines) and not all(is_ma_unavailable):
            

            action_job_precedence_machine = queue_machines[i]
            action = action_job_precedence_machine[0]
            job = action_job_precedence_machine[1]
            precedence = action_job_precedence_machine[2]
            machine = action_job_precedence_machine[3]

            if (
                td["busy_until"][0][machine] > td["time"] # if machine for operation is busy
                or job_counter[job] != precedence # or predecessor of operation not scheduled
                or td["job_in_process"][0][job] # or a predessecor is still processing
                or is_ma_unavailable[machine] # or machine unavailbile (scheduled to, without time having progressed yet)
            ):
                i += 1
                continue



            popped_item = queue_machines.pop(i)
            action = popped_item[0]

            # uncomment in case debugging is needed
            '''
            print(
                f'scheduling action {action} '
                f'for machine {machine} '
                f'busy machines: {td['busy_until'][0]}, '
                f'at time {td["time"]} '
                f'actions left on machine: {action_job_predecence} '
                f', all done? {td['done'].all()} ... {td['done']}'
            )
            print(f'queue machines: {queue_machines}')
            '''

            td["action"] = torch.tensor([action])

            td = env.step(td)["next"]
            i += 1
            scheduled_something = True
            # i = 0 # Trur ikke må starte fra toppen igjen... noe blir jo ikke skedulertbart, bare fordi man har skedulert noe annet på samme tidspunkt
            job_actions_done[ job ].append(popped_item)
            job_counter[ job ] += 1
            n_ops_on_ma[machine] -= 1
            is_ma_unavailable[machine] = True

            # track order of assignments
            if order:
                new_adj = td["ma_assignment"]
                diff = (
                    (new_adj == 1)
                    & (prev_adj == 0)
                )
                if diff.any():
                    actions_assigned += 1
                    ordered_assignments[
                        diff
                    ] = actions_assigned
                prev_adj = new_adj.clone()

            

        td["action"] = torch.tensor([0])
        td = env.step(td)["next"]

        # refreshing list of busy machines
        is_ma_unavailable = [
            td["busy_until"][0][i] > td["time"] # machine is currently busy
            or n_ops_on_ma[i] == 0 # all ops on ma done. noting as scheduling so time can progress
            for i in range(n_machines)
        ]



    # Check if there are any invalid processing times (times less or larger than the enviroment was generated with)
    valid_proc_times = proc_times_before_schedule[ proc_times_before_schedule > 0]
    min_processing_time = valid_proc_times.min().item()
    max_processing_time = valid_proc_times.max().item()
    check_processing_times(
        td['start_times'],
        td['finish_times'],
        min_processing_time=min_processing_time,
        max_processing_time=max_processing_time
    )

    # checking if any operation is assigned to a machine that cannot process it
    invalid = (
        (td['ma_assignment'] == 1)
        & (ops_ma_adj_before_schedule == 0)
    )
    if invalid.any():
        print("scheduled to invalid machine")
        print(invalid.any())
        print(torch.where(invalid))

    '''
    path_save_image = f'/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/code/inference/sched_mautil.png'
    env.render(td, 0)
    if path_save_image:
        plt.savefig(
            path_save_image,
            dpi=150,
            bbox_inches='tight'
        )
    '''
    if order:
        if busy_times: return td, ordered_assignments, amt_busy
        return td, ordered_assignments, amt_busy
    else:
        if busy_times: return td, None, amt_busy
        return td, None, amt_busy






def schedule_batch_instances_mautil(
    env,
    td,
    all_actions,
    order=True,
    busy_times=False
):

    batch_size = all_actions.shape[0]

    td_scheduled_list = []

    ordered_assignments_list = []

    actions_taken_b0 = None

    # -------------------------------------------------
    # Schedule ONE batch instance at a time
    # -------------------------------------------------
    busy_count = 0
    for b in range(batch_size):

        print(
            f'\nScheduling batch {b}'
        )

        td_single = td[
            b:b+1
        ].clone()

        actions_single = (
            all_actions[b]
        )

        (
            td_single,
            ordered_assignments,
            busy
        ) = schedule_single_instance_mautil_alt(
            td_single,
            env,
            actions_single,
            order,
            busy_times
        )
        busy_count += busy

        td_scheduled_list.append(
            td_single
        )

        ordered_assignments_list.append(
            ordered_assignments
        )


    # -------------------------------------------------
    # Concatenate outputs
    # -------------------------------------------------

    td_scheduled = torch.cat(
        td_scheduled_list,
        dim=0
    )

    ordered_assignments = None
    if order:
        ordered_assignments = torch.cat(
            ordered_assignments_list,
            dim=0
        )

    return (
        td_scheduled,
        ordered_assignments,
        busy_count
    )





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








# Takes schedule made from the model, than schedules it
def schedule_from_inference(assignments, env, td, ops_sequence_order):
    
    n_jobs = infer_n_jobs(ops_sequence_order) 
    jobs_to_make = int(os.cpu_count() / 6)

    # Making actions from assignments produced by model
    B = assignments.size(0)
    all_actions = Parallel(n_jobs=jobs_to_make)(
        delayed(utils.map_assignments_to_actions_text)( assignments[b], True, n_jobs )
        for b in range(B)
    )
    all_actions = torch.stack(all_actions)
    all_actions = all_actions.to(torch.int64) 


    if any("opt_" in key for key in td.keys()):
        td.del_("opt_assignment")
        td.del_("opt_assignment_order")
        td.del_("opt_actions")
    else:
        for k in td.keys():
            td[k] = td[k].unsqueeze(0) 
        td = TensorDict(td, batch_size=[1])


    batch_size = all_actions.shape[0]
    td_batched = td.expand(batch_size).clone()

    # Scheduling actions
    td_final, ordered_assignments, busy_count = schedule_batch_instances_mautil(
        env,
        td_batched,
        all_actions,
        order=False
    )
    tds = [
        td_final[i]
        for i in range(batch_size)
    ]


    # noting machine utilization
    ma_all_counts = []
    for td_check in tds:
        ma_assignment_check = td_check['ma_assignment']
        ma_counts = ma_assignment_check.sum(dim=-1).squeeze()
        ma_all_counts.append(ma_counts)

    ma_all_counts = torch.stack(ma_all_counts)
    avg_per_row = ma_all_counts.float().mean(dim=0)

    logging.info('MA USAGE')
    logging.info(avg_per_row)

    makespans = [
    td["finish_times"][td["finish_times"] != 9999.0].max().item()
    for td in tds
    if td["finish_times"][td["finish_times"] != 9999.0].max().item()
    ]

    best_index = makespans.index( min(makespans) )
    worst_index = makespans.index( max(makespans) )
    td_best = tds[ best_index ]
    td_worst = tds[worst_index]

    min_makespan = min(makespans)
    max_makespan = max(makespans)
    avg_makespan = sum(makespans) / len(makespans)

    logging.info(f"makespans: {makespans}")
    logging.info(f"lowest makespan: {min_makespan}")
    logging.info(f"highest makespan: {max_makespan}")
    logging.info(f"Average makespan: {avg_makespan}")

    return td_best, min_makespan, max_makespan, avg_makespan, busy_count, avg_per_row





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

        # RANDOM UNIQUE VALUES
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
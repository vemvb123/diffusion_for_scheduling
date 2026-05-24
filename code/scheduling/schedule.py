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

from collections import deque


import torch


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

# this does not restrict to only 2 actions, as it might not be possible
# but it rather limits the usage of a machine, to something less for the machin, than it otherwise wouldve been
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


def batched_schedule_rollout_keep_order(
    env,
    td,
    all_actions
):

    device = td.device

    i = 0

    actions_taken = []

    batch_size = all_actions.shape[0]

    seq_len = all_actions.shape[1]

    all_actions = all_actions.to(device)

    # -------------------------------------------------
    # Number of machines
    # -------------------------------------------------

    _, h, _ = td["ma_assignment"].shape

    # -------------------------------------------------
    # Build machine-local queues
    # -------------------------------------------------

    machine_queues = []

    for b in range(batch_size):

        queues = [[] for _ in range(h)]

        for t in range(seq_len):

            action = all_actions[b, t].item()

            if action == 0:
                continue

            machine = (
                (action - 1) % h
            )

            queues[machine].append(action)

        machine_queues.append(queues)

    # -------------------------------------------------
    # Queue pointers
    # -------------------------------------------------

    queue_ptr = torch.zeros(
        (batch_size, h),
        dtype=torch.long,
        device=device
    )

    # -------------------------------------------------
    # Track assignments
    # -------------------------------------------------

    prev_adj = td["ma_assignment"].clone()

    ordered_assignments = torch.zeros_like(
        prev_adj,
        dtype=torch.float
    )

    actions_assigned = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=device
    )

    # -------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------

    while not td["done"].all():

        mask = td["action_mask"]

        proposed_actions = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=device
        )

        found_action = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=device
        )

        # -------------------------------------------------
        # FOR EACH MACHINE:
        # only consider FRONT of queue
        # -------------------------------------------------

        for machine in range(h):

            candidate_batches = []

            candidate_actions = []

            for b in range(batch_size):

                if found_action[b]:
                    continue

                ptr = queue_ptr[
                    b,
                    machine
                ].item()

                queue = machine_queues[
                    b
                ][machine]

                if ptr >= len(queue):
                    continue

                action = queue[ptr]

                candidate_batches.append(b)

                candidate_actions.append(action)

            if len(candidate_batches) == 0:
                continue

            candidate_batches = torch.tensor(
                candidate_batches,
                device=device
            )

            candidate_actions = torch.tensor(
                candidate_actions,
                device=device
            )

            feasible = mask[
                candidate_batches,
                candidate_actions
            ]

            feasible_batches = candidate_batches[
                feasible
            ]

            feasible_actions = candidate_actions[
                feasible
            ]

            proposed_actions[
                feasible_batches
            ] = feasible_actions

            found_action[
                feasible_batches
            ] = True

            # advance queue ptr
            for idx in range(
                len(feasible_batches)
            ):

                b = feasible_batches[
                    idx
                ].item()

                queue_ptr[
                    b,
                    machine
                ] += 1

        # -------------------------------------------------
        # NEXT TD
        # -------------------------------------------------

        next_td = td.clone()

        # -------------------------------------------------
        # FEASIBLE ACTIONS
        # -------------------------------------------------

        feasible_idx = found_action.nonzero(
            as_tuple=True
        )[0]

        if len(feasible_idx) > 0:

            td_feasible = td[
                feasible_idx
            ].clone()

            td_feasible["action"] = (
                proposed_actions[
                    feasible_idx
                ]
            )

            td_feasible = env.step(
                td_feasible
            )["next"]

            # ---------------------------------------------
            # SAFE UPDATE
            # ---------------------------------------------

            for key in td.keys():

                target = next_td[key]

                source = td_feasible[key]

                # SAME SHAPE
                if (
                    target[
                        feasible_idx
                    ].shape
                    ==
                    source.shape
                ):

                    target[
                        feasible_idx
                    ] = source

                # TARGET [B]
                # SOURCE [B,1]
                elif (
                    target[
                        feasible_idx
                    ].dim() == 1
                    and source.dim() == 2
                    and source.shape[-1] == 1
                ):

                    target[
                        feasible_idx
                    ] = source.squeeze(-1)

                # GENERAL CASE
                else:

                    target[
                        feasible_idx,
                        ...
                    ] = source

        # -------------------------------------------------
        # NO-OP / TIME ADVANCE
        # -------------------------------------------------

        no_op_idx = (
            (~found_action)
            & (~td["done"].view(-1))
        ).nonzero(as_tuple=True)[0]

        if len(no_op_idx) > 0:

            td_noop = td[
                no_op_idx
            ].clone()

            td_noop["action"] = torch.zeros(
                len(no_op_idx),
                dtype=torch.long,
                device=device
            )

            td_noop = env.step(
                td_noop
            )["next"]

            # ---------------------------------------------
            # SAFE UPDATE
            # ---------------------------------------------

            for key in td.keys():

                target = next_td[key]

                source = td_noop[key]

                # SAME SHAPE
                if (
                    target[
                        no_op_idx
                    ].shape
                    ==
                    source.shape
                ):

                    target[
                        no_op_idx
                    ] = source

                # TARGET [B]
                # SOURCE [B,1]
                elif (
                    target[
                        no_op_idx
                    ].dim() == 1
                    and source.dim() == 2
                    and source.shape[-1] == 1
                ):

                    target[
                        no_op_idx
                    ] = source.squeeze(-1)

                # GENERAL CASE
                else:

                    target[
                        no_op_idx,
                        ...
                    ] = source

        td = next_td

        # -------------------------------------------------
        # RECORD batch 0
        # -------------------------------------------------

        actions_taken.append(
            proposed_actions[0].item()
        )


        # -------------------------------------------------
        # OPTIONAL RENDER
        # -------------------------------------------------

        path_save_image = (
            f'/cluster/datastore/'
            f'vemundvb/diffusion/'
            f'diff_project/'
            f'mindre_prosjekt/code/'
            f'dataset_code/steps/'
            f'sched_{i}_'
            f'{proposed_actions[0].item()}.png'
        )

        env.render(td, 0)

        plt.savefig(
            path_save_image,
            dpi=150,
            bbox_inches='tight'
        )

        plt.close()

        i += 1

        # -------------------------------------------------
        # TRACK ORDER OF ASSIGNMENTS
        # -------------------------------------------------

        new_adj = td["ma_assignment"]

        diff = (
            (new_adj == 1)
            & (prev_adj == 0)
        )

        if diff.any():

            actions_assigned += (
                diff.any(dim=(1, 2)).long()
            )

            for b in range(batch_size):

                batch_diff = diff[b]

                if batch_diff.any():

                    ordered_assignments[b][
                        batch_diff
                    ] = actions_assigned[b]

        prev_adj = new_adj.clone()

    # -------------------------------------------------
    # Convert actions_taken
    # -------------------------------------------------

    actions_taken = torch.tensor(
        actions_taken,
        device=device
    )

    return (
        td,
        ordered_assignments,
        actions_taken
    )


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



import torch


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

    # -----------------------------------
    # machine queues
    # -----------------------------------

    queue_machines = {}

    for machine in range(n_machines):

        queue_machines[machine] = []

    # -----------------------------------
    # count occurrences per job
    # -----------------------------------

    job_counts = {}

    # -----------------------------------
    # process actions in order
    # -----------------------------------

    for action in actions.tolist():

        if action == 0:
            continue

        # -------------------------------
        # decode machine
        # -------------------------------

        machine = (
            (action - 1)
            % n_machines
        )

        # -------------------------------
        # decode job
        # -------------------------------

        job = (
            (action - 1)
            // n_machines
        )

        # -------------------------------
        # precedence in job
        # -------------------------------

        if job not in job_counts:

            job_counts[job] = 0

        else:

            job_counts[job] += 1

        precedence = job_counts[job]

        # -------------------------------
        # append
        # -------------------------------

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



def get_job_counts_mautil(
    actions,
    td,
    n_machines
):
    """
    Returns:

    1.
    How many actions belong
    to each job

    2.
    TRUE job lengths from env
    """

    num_jobs = td[
        "job_ops_adj"
    ].shape[1]

    # actions per job
    job_counts = torch.zeros(
        num_jobs,
        dtype=torch.long
    )

    for action in actions:

        action = action.item()

        if action == 0:
            continue

        # RL4CO:
        # action = job * n_machines + machine
        job = (
            (action - 1)
            // n_machines
        )

        job_counts[job] += 1

    # TRUE env job lengths
    job_lengths = (
        td["job_ops_adj"][0]
        .sum(dim=1)
        .long()
    )

    return (
        job_counts,
        job_lengths
    )

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
from collections import defaultdict

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

    # -----------------------------------
    # Collect occurrences
    # -----------------------------------

    for machine, queue in queue_machines.items():

        for action, job, precedence in queue:

            key = (job, precedence)

            seen[key].append(
                (machine, action)
            )

    # -----------------------------------
    # Find duplicates
    # -----------------------------------

    duplicates = {}

    for key, occurrences in seen.items():

        if len(occurrences) > 1:

            duplicates[key] = occurrences

    # -----------------------------------
    # Print result
    # -----------------------------------

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



def validate_actions_mautil(
    actions,
    ops_ma_adj,
    job_lengths,
    n_machines
):

    # -----------------------------------
    # operation count per job
    # -----------------------------------

    job_counter = [0] * len(job_lengths)

    for action in actions:

        action = action.item()

        if action == 0:
            continue

        # -----------------------------------
        # decode machine/job
        # -----------------------------------

        machine = (
            (action - 1)
            % n_machines
        )

        job = (
            (action - 1)
            // n_machines
        )

        # -----------------------------------
        # where does this job start
        # in operation columns?
        # -----------------------------------

        job_start = sum(
            job_lengths[:job]
        )

        # -----------------------------------
        # operation index inside job
        # -----------------------------------

        precedence = job_counter[job]

        # -----------------------------------
        # actual operation column
        # -----------------------------------

        op_col = (
            job_start
            + precedence
        )

        # -----------------------------------
        # validity
        # -----------------------------------

        valid = (
            ops_ma_adj[
                machine,
                op_col
            ].item()
            == 1
        )

        print(
            f'action={action}, '
            f'job={job}, '
            f'precedence={precedence}, '
            f'machine={machine}, '
            f'op_col={op_col}, '
            f'valid={valid}'
        )

        # increment AFTER checking
        job_counter[job] += 1

# TODO men tar det nå hensyn til at oppgavene har forskjellig prioritet selv om de er på forskjellige maskiner?
# Nå ordrer jeg kanskje bare ut ifra prioritet på forskjellige maskiner, men ikke global prioritet?
# men burde jeg? hva hvis maskin er busy, burde man da nødvendigvis vente med å skedulere neste oppgave?
# det vil nok øke kjøretida... men av oppgaver, burde kanskje prøve å ta den som er globalt først..?
# kan alternativt lage en annen funksjon, som tar hensyn til slikt, bare for å se
# da ser jeg på neste oppgave i på hver maskin, og plukker først den med høyest prioritet, så hvis ikke kan sked, så bare går jeg til neste i prioritet
# når en oppg er skedulert, reordrer jeg etter prioritete neste oppgaver på hver maskin på nytt
def schedule_single_instance_mautil(
    td,
    env,
    actions,
    order=True, # include ordered assignments ma_assignments, specifying in what order the assignments were done
    busy_times=False # record times machine busy when scheduling
):
    

    busy_times = 0
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

    # used to debug if all actions are on valid machines for the operations
    '''
    print(ops_ma_adj_before_schedule)
    print(job_lengths)
    print('..')
    validate_actions_mautil(actions, ops_ma_adj_before_schedule[0], job_lengths, n_machines)
    '''


    # -------------------------------------------------
    # Build queues
    # -------------------------------------------------

    queue_machines = (
        make_queue_mautil(
            actions,
            n_machines
        )
    )
    duplicates = check_duplicate_job_precedence(queue_machines)

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

    # schedule
    while not td['done'].all():

        scheduled = False

        for machine, action_job_predecence in queue_machines.items():
            
            # if all operations for machine scheduled already
            if len(action_job_predecence) == 0:
                continue

            # if the machine for the operation is not busy
            if td["busy_until"][0][machine] > td["time"]:
                busy_times += 1
                continue

            next_item = action_job_predecence[0]
            action = next_item[0]
            job = next_item[1]
            precedence = next_item[2]

            # if the predesecor of the operation has been scheduled
            if job_counter[ job ] != precedence:
                continue
            
            # if another operation on the same job is currently being processed, but is still not done
            if td["job_in_process"][0][job]:
                continue


            popped_item = action_job_predecence.pop(0)
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
            job_actions_done[ job ].append(popped_item)
            job_counter[ job ] += 1
            scheduled = True

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
            


        # IMPORTANT:
        # nothing scheduled -> let env advance time
        if not scheduled:
            td["action"] = torch.tensor([0])
            td = env.step(td)["next"]


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
        if busy_times: return td, ordered_assignments, busy_times
        return td, ordered_assignments, None
    else:
        if busy_times: return td, None, busy_times
        return td, None, None




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

    # -----------------------------------
    # machine queues
    # -----------------------------------
    queue_machines = []

    '''
    for machine in range(n_machines):

        queue_machines[machine] = []

    '''
    # -----------------------------------
    # count occurrences per job
    # -----------------------------------

    job_counts = {}

    # -----------------------------------
    # process actions in order
    # -----------------------------------

    for action in actions.tolist():

        if action == 0:
            continue

        # -------------------------------
        # decode machine
        # -------------------------------

        machine = (
            (action - 1)
            % n_machines
        )

        # -------------------------------
        # decode job
        # -------------------------------

        job = (
            (action - 1)
            // n_machines
        )

        # -------------------------------
        # precedence in job
        # -------------------------------

        if job not in job_counts:

            job_counts[job] = 0

        else:

            job_counts[job] += 1

        precedence = job_counts[job]

        # -------------------------------
        # append
        # -------------------------------

        queue_machines.append(
            (
                action,
                job,
                precedence,
                machine
            )
        )

    return queue_machines



# TODO:
# X at det merkes når alle actions på en maskin gjort (kan kanskje telle på forhånd, til ta minus hver gang sked til en maskin)
# X kun går fram i tid når alle maskiner er skedulert til
# X når alle actions for en maskin er gjort, skal man kunne progressere selv om noe ikke er skedulert til maskin
# X starter fra toppen av liste når en action skedulert
# X når action skedulert, tar man den ut fra lista i queue
# X hvis action ikke kan skeduleres, går man bare til neste action
def schedule_single_instance_mautil_alt(
    td,
    env,
    actions,
    order=True, # include ordered assignments ma_assignments, specifying in what order the assignments were done
    busy_times=False # record times machine busy when scheduling
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

    # used to debug if all actions are on valid machines for the operations
    '''
    print(ops_ma_adj_before_schedule)
    print(job_lengths)
    print('..')
    validate_actions_mautil(actions, ops_ma_adj_before_schedule[0], job_lengths, n_machines)
    '''


    # -------------------------------------------------
    # Build queues
    # -------------------------------------------------

    queue_machines = (
        make_queue_mautil_alt(
            actions,
            n_machines
        )
    )
    # duplicates = check_duplicate_job_precedence(queue_machines)

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


    # for debugging wether assignments are made to only valid machines for each operation
    '''
    padding = torch.zeros(*assignments.shape[:-1],5,device=assignments.device)
    assignments_padded = torch.cat([assignments, padding],dim=-1)

    ops_ma_adj = td["ops_ma_adj"].to(assignments_padded.device)

    print(assignments_padded.shape)
    print(ops_ma_adj.shape)

    invalid = ((assignments_padded != 0) & (ops_ma_adj == 0))

    print('her')
    print(invalid.any())
    print(torch.nonzero(invalid))

    exit()
    '''


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


    # if any("opt_" in key for key in td):
    if any("opt_" in key for key in td.keys()):
        td.del_("opt_assignment")
        td.del_("opt_assignment_order")
        td.del_("opt_actions")
    else:
        for k in td.keys():
            td[k] = td[k].unsqueeze(0)  # shape becomes [1, 6, 60] for your tensor

        # Wrap in a TensorDict
        td = TensorDict(td, batch_size=[1])



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

    '''
    print(f'all actions shape: {all_actions.shape}')
    td_final, executed, invalid_action_counters = batched_schedule_rollout(
    env,
    td_batched,
    all_actions,
    render_idx=None,
    save_dir="frames",
    )
    '''
    
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

    # machine utilization
    ma_all_counts = []
    for td_check in tds:
        ma_assignment_check = td_check['ma_assignment']
        ma_counts = ma_assignment_check.sum(dim=-1).squeeze()
        ma_all_counts.append(ma_counts)

    ma_all_counts = torch.stack(ma_all_counts)
    avg_per_row = ma_all_counts.float().mean(dim=0)
    print('MA USAGE')
    print(avg_per_row)

    # amount valid operations for each machine
    '''
    valids = td['ops_ma_adj']
    # Count 1s per row
    # Sum across width dimension (last dim)
    row_counts = valids.sum(dim=-1)

    print(row_counts.shape)  # [1, 1, H]
    print(row_counts)
    print('amount valids')
    exit()
    '''


    # TODO fjern etterpå
    '''
    print('done scheduling')
    print(invalid_action_counters.shape)

    invalid_action_counters = (
        invalid_action_counters
        .cpu()
        .tolist()
    )
    actions_taken = [
        executed[i].cpu().tolist()
        for i in range(batch_size)
    ]
    '''

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
    td_worst = tds[worst_index]
    # best_actions = actions_taken[makespans.index( min(makespans ))]

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
    fold = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/scheds'
    name_sched = 'new_sched_best.png'
    path_save_image = f'{fold}/{name_sched}'
    env.render(td_best.unsqueeze(0), 0)
    if path_save_image:
        plt.savefig(path_save_image, dpi=150, bbox_inches='tight')
        print(f"Saved scheduled image at path {path_save_image}")

    name_sched = 'new_sched_worst.png'
    path_save_image = f'{fold}/{name_sched}'
    env.render(td_best.unsqueeze(0), 0)
    if path_save_image:
        plt.savefig(path_save_image, dpi=150, bbox_inches='tight')
        print(f"Saved scheduled image at path {path_save_image}")



    # TODO ukkomenter etterpå
    '''
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
    '''

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
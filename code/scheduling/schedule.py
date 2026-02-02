"""
scheduling_utils.py contains all code that does scheduling
"""



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


def inferenced_schedule( assignments, order: bool, env, td, path_save_image: str, n_jobs: int, n_machines: int, ops_sequence_order ):
    #actions = utils.map_assignemnts_to_actions(assignments, order, n_jobs)

    n_jobs = infer_n_jobs(ops_sequence_order) # example [6,5,6,5,6,5,6,5,6,5, 5]
    actions = utils.map_assignments_to_actions_text(assignments, True, n_jobs)

    print("herskjekkda")
    print(actions)
    print(td["opt_actions"])

    td.del_("opt_assignment")
    td.del_("opt_assignment_order")
    td.del_("opt_actions")

    td = td.unsqueeze(0)
    env.render(td, 0)



    # antar at order assignments da inneholder større og større verdi for hver order
    #if order:
        # TODO hvordan håndtere/se skeduleringsfeil i seq    
        # se på verdier fra inferenced
        # map hver verdi til en action, i sekvensen assignments gir
        # .. alt det gjør du fr loopen
        # så når loopen starter, looper du igjennom lista av actions

        # ops: td gir ikke oppgaver klare, men maskiner,
        #  og det gis i format som teller opp, så når ma1 er klar første gang sier den 0, så neste gang sier den 1
        # men når det er ops, så trur jeg den teller ned... så eks første kolonne er 1,2,3,4 osv...

        # looper action i actions
        # ser på is ready, som gir klare maskiner
        # mapper ops og actions
        # - har spenn for hver maskin
        # - 0-3,4-7,8-11,12-5
        # - hvis ac er innenfor en av de tilgjengelige spenna, så kan den skeduleres, eller må den hoppe i tid
        # - da hopper den i tid til den maskinen er klar
        # tar action, og skedulerer den ut

        # får klare maskiner (rl4co gir format 0,4,8,12), så teller den 1 opp etter en skedulering, til eks 1,4,8,12
    # if order:

    print(actions)
    for action in actions:
        # machine index that this action refers to
        ma_to_use = (action - 1) % n_machines

        while td["busy_until"][0, ma_to_use].item() > td["time"].item():
            invalid_action = torch.tensor([0])  
            td["action"] = invalid_action

            td = env.step(td)["next"]
            env.render(td, 0)

        if action != 0:
            td["action"] = torch.tensor([action])
            td = env.step(td)["next"]
            env.render(td, 0)


    """
    else:
        while not td["done"].all():
            ready_ops = torch.nonzero(td["is_ready"], as_tuple=True)[1]

            ma_indices_for_actions = [((v - 1) % n_jobs) for v in actions]  # 0‑based
            # print("machine indices:", ma_indices_for_actions)

            for op in ready_ops:
                op = op.item()
                machine_idx = ma_indices_for_actions[op]
                busy_val = td["busy_until"][0, machine_idx]
                current_time = td["time"][0]

                # td["time"] = busy_val.unsqueeze(0)

                if busy_val <= current_time:
                    td['action'] = torch.tensor([actions[op]])
                    td = env.step(td)['next']
                    env.render(td, 0)
                else:
                    td["time"] = busy_val.unsqueeze(0)

    """
    print("done")
    print(td["ma_assignment"])
    if path_save_image:
        plt.savefig(path_save_image, dpi=150, bbox_inches='tight')
        print(f"Saved scheduled image at path {path_save_image}")
    return td


# inferenced_schedule()








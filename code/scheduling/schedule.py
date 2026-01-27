"""
scheduling_utils.py contains all code that does scheduling
"""



import time
import random
import os
import logging
import bisect

from code.scheduling.utils import map_assignemnts_to_actions

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


def inferenced_schedule(assignments, order: bool, env, td, path_save_image: str, n_jobs: int):
    actions = map_assignemnts_to_actions(assignments, order, n_jobs)
    # print(assignments)
    # print(td["opt_assignment"])
    # print(actions)
    # print(td["opt_actions"])
    # exit()
    td.del_("opt_assignment")
    td.del_("opt_actions")

    #print(td.shape)
    #td = TensorDict.from_dict(td, auto_batch_size=True)
    #print(td.shape)
    #td = TensorDict(td, batch_size=[1])
    #print(td.shape)

    td = td.unsqueeze(0)
    env.render(td, 0)


    #---
    while not td["done"].all():
        # loop true indexer
        # for en true index, se at maskinen den oppgaven skal skeduleres til ikke er opptatt
        # det ses ved at: mapped = [((v - 1) % 4) + 1 for v in actions]  ... fra ctions
        # hvis opptatt, gå til neste gå til neste true.
        # så den oppgaven endelig kn skeduleres, skeduleres den, så starter du å loope true fra starten av
        # time.sleep(10)
        #print(td["time"])
        #print(td["busy_until"])
        #print(td["is_ready"])
        #print(assignments)

        if order:
            pass

        else:
            ready_ops = torch.nonzero(td["is_ready"], as_tuple=True)[1]

            ma_indices_for_actions = [((v - 1) % n_jobs) for v in actions]  # 0‑based
            # print("machine indices:", ma_indices_for_actions)

            for op in ready_ops:
                op = op.item()
                machine_idx = ma_indices_for_actions[op]
                busy_val = td["busy_until"][0, machine_idx]
                current_time = td["time"][0]

                if busy_val <= current_time:
                    td['action'] = torch.tensor([actions[op]])
                    td = env.step(td)['next']
                    env.render(td, 0)
                else:
                    td["time"] = busy_val.unsqueeze(0)

    if path_save_image:
        plt.savefig(path_save_image, dpi=150, bbox_inches='tight')
    return td










